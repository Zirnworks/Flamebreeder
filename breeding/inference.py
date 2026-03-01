"""Inference pipeline for conditional StyleGAN2-ADA generator.

Loads NVIDIA's .pkl checkpoint directly and wraps it for W-space
breeding operations with conditional class label support.

The 30 conditional labels correspond to aesthetic clusters (fractal
flame families). Blending labels produces smooth interpolated aesthetics.
"""

import os
import sys
from pathlib import Path

import torch
from PIL import Image

# NVIDIA's code is needed to unpickle the checkpoint
_VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "stylegan2-ada-pytorch"

NUM_CLASSES = 30


def _ensure_nvidia_imports():
    """Add NVIDIA stylegan2-ada-pytorch to sys.path for unpickling."""
    vendor_str = str(_VENDOR_DIR)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)


class FlameGenerator:
    """Wraps the trained conditional StyleGAN2-ADA generator.

    The generator has two stages:
        1. Mapping network: z (512) + embed(c) (30→512) → w (512)
        2. Synthesis network: w → 512x512 RGB image

    For breeding, all operations happen in W-space:
        - generate_random() returns W vectors, Z vectors, and class labels
        - generate_from_w() renders images from W vectors
        - generate_from_z() re-maps a Z vector with a new class label (gene editing)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "mps",
    ):
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.model = self._load_nvidia_pkl(str(checkpoint_path))
        self.model.to(self.device)
        self.model.eval()

        self.z_dim = self.model.z_dim
        self.c_dim = self.model.c_dim
        self.w_dim = self.model.w_dim
        self.num_ws = self.model.num_ws
        self.img_resolution = self.model.img_resolution

    @staticmethod
    def _load_nvidia_pkl(pkl_path: str):
        """Load G_ema from NVIDIA's .pkl checkpoint."""
        _ensure_nvidia_imports()
        import dnnlib
        import legacy

        with dnnlib.util.open_url(pkl_path) as f:
            data = legacy.load_network_pkl(f)
        return data["G_ema"]

    def _make_class_label(self, class_label: list[float] | None) -> torch.Tensor:
        """Convert a class label list to a tensor for the model.

        Args:
            class_label: 30-dim vector (will be normalized to sum to 1),
                         or None for a uniform label.

        Returns:
            Tensor of shape (1, c_dim) on the correct device.
        """
        if class_label is None:
            # Uniform across all classes
            c = torch.ones(1, self.c_dim, device=self.device) / self.c_dim
        else:
            c = torch.tensor([class_label], dtype=torch.float32, device=self.device)
            # Normalize to sum to 1 (consistent class signal strength)
            total = c.sum(dim=1, keepdim=True)
            if total.item() > 0:
                c = c / total
        return c

    @property
    def w_avg(self) -> torch.Tensor:
        """Mean W vector for truncation trick."""
        return self.model.mapping.w_avg

    @torch.no_grad()
    def map_z_to_w(
        self,
        z: torch.Tensor,
        class_label: list[float] | None = None,
        truncation_psi: float = 0.7,
    ) -> torch.Tensor:
        """Map z-space vectors to w-space via the mapping network.

        Args:
            z: Latent vectors (B, z_dim).
            class_label: 30-dim class label vector (shared across batch).
            truncation_psi: Truncation strength. 1.0 = no truncation.

        Returns:
            W vectors (B, num_ws, w_dim).
        """
        z = z.to(self.device)
        c = self._make_class_label(class_label)
        c = c.expand(z.shape[0], -1)
        ws = self.model.mapping(z, c, truncation_psi=truncation_psi)
        return ws.cpu()

    @torch.no_grad()
    def generate_from_w(self, ws: torch.Tensor) -> list[Image.Image]:
        """Generate images from W-space vectors.

        Args:
            ws: W vectors, either:
                - (B, w_dim): broadcast to all style layers
                - (B, num_ws, w_dim): per-layer styles for style mixing

        Returns:
            List of PIL Images.
        """
        ws = ws.to(self.device)
        if ws.ndim == 2:
            ws = ws.unsqueeze(1).expand(-1, self.num_ws, -1)
        images = self.model.synthesis(ws)
        return self._to_pil(images)

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        class_label: list[float] | None = None,
        truncation_psi: float = 0.7,
    ) -> list[Image.Image]:
        """Generate images from z-space (full pipeline through mapping network)."""
        z = z.to(self.device)
        c = self._make_class_label(class_label)
        c = c.expand(z.shape[0], -1)
        images = self.model(z, c, truncation_psi=truncation_psi)
        return self._to_pil(images)

    def random_latent(self, count: int = 1) -> torch.Tensor:
        """Generate random z-space vectors."""
        return torch.randn(count, self.z_dim)

    @torch.no_grad()
    def generate_random(
        self,
        count: int = 1,
        truncation_psi: float = 0.7,
        class_label: list[float] | None = None,
    ) -> tuple[list[Image.Image], torch.Tensor, torch.Tensor]:
        """Generate random images with class label conditioning.

        Returns:
            (images, w_vectors, z_vectors): PIL images, W vectors for
            breeding, and Z vectors for gene editing.
        """
        z = self.random_latent(count)
        ws = self.map_z_to_w(z, class_label=class_label, truncation_psi=truncation_psi)
        images = self.generate_from_w(ws)
        return images, ws, z

    @torch.no_grad()
    def generate_from_z(
        self,
        z: torch.Tensor,
        class_label: list[float] | None = None,
        truncation_psi: float = 0.7,
    ) -> tuple[list[Image.Image], torch.Tensor]:
        """Re-map a z-vector with a (possibly new) class label.

        This is the gene editing path: same seed z, different class label c
        produces a different w, yielding a structurally related but
        aesthetically shifted image.

        Args:
            z: Latent vectors (B, z_dim).
            class_label: New class label vector.
            truncation_psi: Truncation strength.

        Returns:
            (images, w_vectors): The generated images and the new W vectors.
        """
        ws = self.map_z_to_w(z, class_label=class_label, truncation_psi=truncation_psi)
        images = self.generate_from_w(ws)
        return images, ws

    def truncate_w(self, w: torch.Tensor, psi: float = 0.7) -> torch.Tensor:
        """Apply truncation trick to a W vector.

        Pulls the W vector toward the learned mean, trading diversity
        for quality. psi=1.0 means no change; psi=0.0 collapses to mean.
        """
        w_avg = self.w_avg.to(w.device)
        return w_avg + psi * (w - w_avg)

    def _to_pil(self, images: torch.Tensor) -> list[Image.Image]:
        """Convert model output tensor [-1, 1] to PIL images."""
        images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)
        images = (images * 255).byte().cpu().numpy()
        return [
            Image.fromarray(img.transpose(1, 2, 0), mode="RGB")
            for img in images
        ]
