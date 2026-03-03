"""Timeform: 3D volumetric export from interpolation strips.

Stacks interpolation frames as alpha-transparent textured planes along
the Z-axis, producing a wispy 3D sculpture exported as GLB for game engines.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
from PIL import Image

from .interpolation import multi_keyframe_strip

if TYPE_CHECKING:
    from .genome import Genome
    from .inference import FlameGenerator


def rgb_to_rgba(img: Image.Image) -> Image.Image:
    """Convert RGB image to RGBA with luminance-based alpha.

    Black pixels become transparent, bright pixels become opaque.
    Alpha = max(R, G, B) per pixel.
    """
    rgb = np.array(img)
    alpha = np.max(rgb, axis=2)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def compute_adaptive_spacing(images: list[Image.Image]) -> list[float]:
    """Compute Z positions proportional to visual change between frames.

    Returns normalized Z values in [0, 1].
    """
    if len(images) < 2:
        return [0.0]

    diffs = []
    for i in range(len(images) - 1):
        a = np.array(images[i], dtype=np.float32)
        b = np.array(images[i + 1], dtype=np.float32)
        mse = np.mean((a - b) ** 2)
        diffs.append(max(mse, 1e-6))

    cumulative = np.cumsum([0.0] + diffs)
    return (cumulative / cumulative[-1]).tolist()


def compute_uniform_spacing(n: int) -> list[float]:
    """Compute evenly-spaced Z positions in [0, 1]."""
    if n < 2:
        return [0.0]
    return [i / (n - 1) for i in range(n)]


def generate_frames_batched(
    generator: FlameGenerator,
    w_strip: list[torch.Tensor],
    batch_size: int = 16,
) -> list[Image.Image]:
    """Generate frames from W vectors in memory-safe sub-batches."""
    images = []
    for i in range(0, len(w_strip), batch_size):
        chunk = torch.stack(w_strip[i : i + batch_size])
        images.extend(generator.generate_from_w(chunk))
    return images


def build_timeform_glb(
    rgba_images: list[Image.Image],
    z_positions: list[float],
    total_depth: float = 10.0,
    quad_size: float = 5.0,
) -> bytes:
    """Assemble RGBA textured quads into a GLB binary.

    Each image becomes a transparent quad at the corresponding Z position.
    """
    scene = trimesh.Scene()
    half = quad_size / 2.0

    for i, (img, z_norm) in enumerate(zip(rgba_images, z_positions)):
        z = z_norm * total_depth

        vertices = np.array(
            [
                [-half, -half, z],
                [half, -half, z],
                [half, half, z],
                [-half, half, z],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)

        # Convert PIL RGBA to PNG bytes for trimesh texture
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        texture_img = Image.open(buf)

        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture_img,
            alphaMode="BLEND",
            doubleSided=True,
        )
        color_visuals = trimesh.visual.TextureVisuals(uv=uv, material=material)

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            visual=color_visuals,
            process=False,
        )
        scene.add_geometry(mesh, node_name=f"frame_{i:04d}")

    return scene.export(file_type="glb")


def create_timeform(
    generator: FlameGenerator,
    keyframes: list[Genome],
    total_frames: int = 128,
    spacing: str = "uniform",
    texture_size: int = 256,
    total_depth: float = 10.0,
    quad_size: float = 5.0,
    method: str = "slerp",
    batch_size: int = 16,
) -> bytes:
    """Full pipeline: keyframes → interpolation → generation → GLB.

    Args:
        generator: The FlameGenerator instance.
        keyframes: Ordered list of Genome objects (at least 2).
        total_frames: Number of frames to generate (16-512).
        spacing: "uniform" or "adaptive".
        texture_size: Resize textures to this size (e.g., 256).
        total_depth: Z-axis extent of the timeform.
        quad_size: XY size of each quad plane.
        method: Interpolation method ("slerp" or "lerp").
        batch_size: Sub-batch size for generation.

    Returns:
        GLB binary bytes.
    """
    # Extract W-vectors from keyframes
    ws = [torch.tensor(kf.latent_vector, dtype=torch.float32) for kf in keyframes]

    # Generate multi-keyframe interpolation strip
    w_strip = multi_keyframe_strip(ws, total_frames, method=method)

    # Batch-generate frames
    images = generate_frames_batched(generator, w_strip, batch_size=batch_size)

    # Resize textures
    if texture_size != 512:
        images = [
            img.resize((texture_size, texture_size), Image.LANCZOS) for img in images
        ]

    # Convert to RGBA (black → transparent)
    rgba_images = [rgb_to_rgba(img) for img in images]

    # Compute Z spacing
    if spacing == "adaptive":
        z_positions = compute_adaptive_spacing(images)
    else:
        z_positions = compute_uniform_spacing(len(images))

    # Build and export GLB
    return build_timeform_glb(rgba_images, z_positions, total_depth, quad_size)
