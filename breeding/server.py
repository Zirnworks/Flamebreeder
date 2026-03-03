"""FastAPI inference server for fractal flame GAN breeding.

Runs as a sidecar process alongside the Tauri desktop app.
Provides endpoints for generation, breeding, interpolation, gene editing,
and genome management. All latent operations happen in W-space for smooth,
disentangled results. Class labels (30 clusters) steer the aesthetic family.
"""

import base64
import io
import os
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .inference import FlameGenerator
from .breeding import BREEDING_METHODS, blend_class_labels, mutate, truncate_w
from .interpolation import interpolation_strip
from .genome import Genome, GenomeStore

app = FastAPI(title="Fractal Flame Breeder", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state — initialized on startup
generator: FlameGenerator | None = None
store: GenomeStore | None = None
images_dir: Path | None = None


def image_to_base64(img) -> str:
    """Convert PIL Image to base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_image(genome_id: str, img) -> None:
    """Save a PIL Image to disk alongside the genome metadata."""
    if images_dir is not None:
        img.save(images_dir / f"{genome_id}.png")


# --- Request/Response Models ---

class GenerateRequest(BaseModel):
    count: int = Field(default=4, ge=1, le=16)
    truncation_psi: float = Field(default=0.7, ge=0.0, le=1.0)
    class_label: list[float] | None = None

class GenomeResponse(BaseModel):
    id: str
    image_base64: str
    genome: dict

class BreedRequest(BaseModel):
    parent_a_id: str
    parent_b_id: str
    method: str = "average"
    params: dict = Field(default_factory=dict)
    count: int = Field(default=1, ge=1, le=8)

class InterpolateRequest(BaseModel):
    genome_a_id: str
    genome_b_id: str
    steps: int = Field(default=10, ge=2, le=64)
    method: str = "slerp"

class MutateRequest(BaseModel):
    genome_id: str
    rate: float = Field(default=0.1, ge=0.0, le=1.0)
    strength: float = Field(default=0.3, ge=0.0, le=2.0)

class RemapRequest(BaseModel):
    genome_id: str
    class_label: list[float]
    truncation_psi: float = Field(default=0.7, ge=0.0, le=1.0)

class UpdateGenomeRequest(BaseModel):
    tags: list[str] | None = None
    favorite: bool | None = None


# --- Endpoints ---

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": generator is not None,
        "latent_space": "w",
        "w_dim": generator.w_dim if generator else None,
        "c_dim": generator.c_dim if generator else None,
        "num_classes": generator.c_dim if generator else None,
    }


@app.post("/generate", response_model=list[GenomeResponse])
def generate(req: GenerateRequest):
    """Generate random fractal flame images with optional class label."""
    if generator is None:
        raise HTTPException(503, "Model not loaded")

    images, ws_batch, z_batch = generator.generate_random(
        req.count,
        truncation_psi=req.truncation_psi,
        class_label=req.class_label,
    )
    results = []

    for img, ws, z in zip(images, ws_batch, z_batch):
        # Store the first layer's W for flat-vector breeding operations
        w_flat = ws[0] if ws.ndim >= 2 else ws
        genome = Genome(
            latent_vector=w_flat.tolist(),
            latent_space="w",
            truncation_psi=req.truncation_psi,
            class_label=req.class_label,
            seed_z=z.tolist(),
            generation=0,
        )
        store.add(genome)
        save_image(genome.id, img)
        results.append(GenomeResponse(
            id=genome.id,
            image_base64=image_to_base64(img),
            genome=genome.to_dict(),
        ))

    return results


@app.post("/breed", response_model=list[GenomeResponse])
def breed(req: BreedRequest):
    """Breed two parent genomes to produce offspring."""
    if generator is None:
        raise HTTPException(503, "Model not loaded")

    parent_a = store.get(req.parent_a_id)
    parent_b = store.get(req.parent_b_id)
    if parent_a is None or parent_b is None:
        raise HTTPException(404, "Parent genome not found")

    if req.method not in BREEDING_METHODS:
        raise HTTPException(400, f"Unknown method: {req.method}. Options: {list(BREEDING_METHODS)}")

    breed_fn = BREEDING_METHODS[req.method]
    w_a = torch.tensor(parent_a.latent_vector, dtype=torch.float32)
    w_b = torch.tensor(parent_b.latent_vector, dtype=torch.float32)

    # Extract breeding-specific params (filter out server-level params)
    params = dict(req.params)
    psi = params.pop("truncation_psi", 0.7)
    ratio = params.get("ratio", 0.5)

    # For style_mix, inject num_ws from the generator
    if req.method == "style_mix" and "num_ws" not in params:
        params["num_ws"] = generator.num_ws

    # Blend class labels from both parents
    child_label = blend_class_labels(
        parent_a.class_label, parent_b.class_label, ratio=ratio
    )

    results = []
    parent_gen = max(parent_a.generation, parent_b.generation)

    for _ in range(req.count):
        w_child = breed_fn(w_a, w_b, **params)

        # W-space truncation
        if w_child.ndim == 1:
            w_child = generator.truncate_w(w_child, psi=psi)
            images = generator.generate_from_w(w_child.unsqueeze(0))
            store_vector = w_child.tolist()
        else:
            # Style mixing returns (num_ws, w_dim) — truncate each layer
            w_child = generator.truncate_w(w_child, psi=psi)
            images = generator.generate_from_w(w_child.unsqueeze(0))
            store_vector = w_child[0].tolist()

        genome = Genome(
            latent_vector=store_vector,
            latent_space="w",
            truncation_psi=psi,
            class_label=child_label,
            parents=(parent_a.id, parent_b.id),
            breeding_method=req.method,
            breeding_params=req.params,
            generation=parent_gen + 1,
        )
        store.add(genome)
        save_image(genome.id, images[0])
        results.append(GenomeResponse(
            id=genome.id,
            image_base64=image_to_base64(images[0]),
            genome=genome.to_dict(),
        ))

    return results


@app.post("/interpolate", response_model=list[GenomeResponse])
def interpolate(req: InterpolateRequest):
    """Generate an interpolation strip between two genomes."""
    if generator is None:
        raise HTTPException(503, "Model not loaded")

    genome_a = store.get(req.genome_a_id)
    genome_b = store.get(req.genome_b_id)
    if genome_a is None or genome_b is None:
        raise HTTPException(404, "Genome not found")

    w_a = torch.tensor(genome_a.latent_vector, dtype=torch.float32)
    w_b = torch.tensor(genome_b.latent_vector, dtype=torch.float32)

    w_strip = interpolation_strip(w_a, w_b, req.steps, method=req.method)
    w_batch = torch.stack(w_strip)
    images = generator.generate_from_w(w_batch)

    results = []
    for i, (img, w) in enumerate(zip(images, w_strip)):
        t = i / (req.steps - 1)
        child_label = blend_class_labels(
            genome_a.class_label, genome_b.class_label, ratio=t
        )
        genome = Genome(
            latent_vector=w.tolist(),
            latent_space="w",
            class_label=child_label,
            parents=(genome_a.id, genome_b.id),
            breeding_method=f"interpolate_{req.method}",
            breeding_params={"t": t},
            generation=max(genome_a.generation, genome_b.generation),
        )
        store.add(genome)
        save_image(genome.id, img)
        results.append(GenomeResponse(
            id=genome.id,
            image_base64=image_to_base64(img),
            genome=genome.to_dict(),
        ))

    return results


@app.post("/mutate", response_model=GenomeResponse)
def mutate_endpoint(req: MutateRequest):
    """Mutate a genome to produce a variant."""
    if generator is None:
        raise HTTPException(503, "Model not loaded")

    parent = store.get(req.genome_id)
    if parent is None:
        raise HTTPException(404, "Genome not found")

    w = torch.tensor(parent.latent_vector, dtype=torch.float32)
    w_child = mutate(w, mutation_rate=req.rate, mutation_strength=req.strength)
    w_child = generator.truncate_w(w_child, psi=0.7)

    images = generator.generate_from_w(w_child.unsqueeze(0))
    genome = Genome(
        latent_vector=w_child.tolist(),
        latent_space="w",
        class_label=parent.class_label,
        parents=(parent.id, parent.id),
        breeding_method="mutate",
        breeding_params={"rate": req.rate, "strength": req.strength},
        generation=parent.generation + 1,
    )
    store.add(genome)
    save_image(genome.id, images[0])

    return GenomeResponse(
        id=genome.id,
        image_base64=image_to_base64(images[0]),
        genome=genome.to_dict(),
    )


@app.post("/remap", response_model=GenomeResponse)
def remap(req: RemapRequest):
    """Re-map a genome's z-vector with a new class label (gene editing).

    Takes the original z-vector stored in the genome, passes it through
    the mapping network with a new class label to get a new w-vector,
    then generates the image. This preserves structure while shifting
    the aesthetic family.
    """
    if generator is None:
        raise HTTPException(503, "Model not loaded")

    parent = store.get(req.genome_id)
    if parent is None:
        raise HTTPException(404, "Genome not found")
    if parent.seed_z is None:
        raise HTTPException(
            400, "Genome has no seed z-vector (bred genomes can't be remapped)"
        )

    z = torch.tensor([parent.seed_z], dtype=torch.float32)
    images, ws = generator.generate_from_z(
        z, class_label=req.class_label, truncation_psi=req.truncation_psi
    )

    w_flat = ws[0, 0] if ws.ndim >= 3 else ws[0]
    genome = Genome(
        latent_vector=w_flat.tolist(),
        latent_space="w",
        truncation_psi=req.truncation_psi,
        class_label=req.class_label,
        seed_z=parent.seed_z,
        parents=(parent.id, parent.id),
        breeding_method="remap",
        breeding_params={"class_label": req.class_label},
        generation=parent.generation,
    )
    store.add(genome)
    save_image(genome.id, images[0])

    return GenomeResponse(
        id=genome.id,
        image_base64=image_to_base64(images[0]),
        genome=genome.to_dict(),
    )


@app.get("/genome/{genome_id}")
def get_genome(genome_id: str):
    """Retrieve a genome by ID and regenerate its image."""
    genome = store.get(genome_id)
    if genome is None:
        raise HTTPException(404, "Genome not found")

    w = torch.tensor(genome.latent_vector, dtype=torch.float32).unsqueeze(0)
    images = generator.generate_from_w(w)

    return GenomeResponse(
        id=genome.id,
        image_base64=image_to_base64(images[0]),
        genome=genome.to_dict(),
    )


@app.patch("/genome/{genome_id}")
def update_genome(genome_id: str, req: UpdateGenomeRequest):
    """Update genome metadata (tags, favorite)."""
    genome = store.get(genome_id)
    if genome is None:
        raise HTTPException(404, "Genome not found")

    if req.tags is not None:
        genome.tags = req.tags
    if req.favorite is not None:
        genome.favorite = req.favorite

    store.update(genome)
    return genome.to_dict()


@app.get("/genome/{genome_id}/image")
def get_genome_image(genome_id: str):
    """Serve a saved genome image from disk as base64."""
    genome = store.get(genome_id)
    if genome is None:
        raise HTTPException(404, "Genome not found")

    img_path = images_dir / f"{genome_id}.png"
    if not img_path.exists():
        raise HTTPException(404, "Image not found on disk")

    img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return GenomeResponse(
        id=genome.id,
        image_base64=img_b64,
        genome=genome.to_dict(),
    )


@app.get("/genomes")
def list_genomes():
    """List all saved genomes with metadata and image availability."""
    results = []
    for g in store.all():
        d = g.to_dict()
        d["has_image"] = (images_dir / f"{g.id}.png").exists() if images_dir else False
        results.append(d)
    return results


def start_server(
    checkpoint_path: str,
    data_dir: str = "~/.fractal-breeder",
    host: str = "127.0.0.1",
    port: int = 8420,
    device: str = "mps",
):
    """Start the inference server."""
    global generator, store, images_dir

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    data_path = Path(data_dir).expanduser()
    data_path.mkdir(parents=True, exist_ok=True)
    images_dir = data_path / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"Loading StyleGAN2-ADA model from {checkpoint_path}...")
    generator = FlameGenerator(checkpoint_path, device=device)
    store = GenomeStore(data_path / "genomes.json")

    print(f"Model loaded: z_dim={generator.z_dim}, c_dim={generator.c_dim}, "
          f"w_dim={generator.w_dim}, num_ws={generator.num_ws}")
    print(f"Server starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m breeding.server <checkpoint_path> [--port PORT] [--device DEVICE]")
        sys.exit(1)

    checkpoint = sys.argv[1]
    port = 8420
    device = "mps"
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
        if arg == "--device" and i + 1 < len(sys.argv):
            device = sys.argv[i + 1]

    start_server(checkpoint, port=port, device=device)
