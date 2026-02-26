"""Inference pipeline for StyleGAN2-ADA generator.

Supports W-space operations: mapping (z->w), synthesis (w->image),
truncation, and style mixing. All breeding operates in W-space for
smooth, disentangled interpolation.
"""

from pathlib import Path

import torch
from PIL import Image

from .stylegan2.generator import StyleGAN2Generator


class FlameGenerator:
    """Wraps the trained StyleGAN2 generator for W-space inference.

    The generator has two stages:
        1. Mapping network: z (512-dim Gaussian) -> w (512-dim learned manifold)
        2. Synthesis network: w -> 512x512 RGB image

    For breeding, all operations happen in W-space:
        - generate_random() returns W vectors (not Z)
        - generate_from_w() renders images from W vectors
        - truncate_w() applies the truncation trick in W-space
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

        self.model = StyleGAN2Generator.load_from_nvidia(
            str(checkpoint_path), device=device
        )
        self.model.to(self.device)
        self.model.eval()

        self.z_dim = self.model.z_dim
        self.w_dim = self.model.w_dim
        self.num_ws = self.model.num_ws

    @property
    def w_avg(self) -> torch.Tensor:
        """Mean W vector for truncation trick."""
        return self.model.mapping.w_avg

    @torch.no_grad()
    def map_z_to_w(
        self, z: torch.Tensor, truncation_psi: float = 0.7
    ) -> torch.Tensor:
        """Map z-space vectors to w-space via the mapping network.

        Args:
            z: Latent vectors (B, z_dim).
            truncation_psi: Truncation strength. 1.0 = no truncation.

        Returns:
            W vectors (B, w_dim).
        """
        z = z.to(self.device)
        return self.model.mapping(z, truncation_psi=truncation_psi).cpu()

    @torch.no_grad()
    def generate_from_w(self, w: torch.Tensor) -> list[Image.Image]:
        """Generate images from W-space vectors.

        Args:
            w: W vectors. Either:
                - (B, w_dim): broadcast to all style layers
                - (B, num_ws, w_dim): per-layer styles for style mixing

        Returns:
            List of PIL Images.
        """
        w = w.to(self.device)
        images = self.model.forward_from_w(w)
        return self._to_pil(images)

    @torch.no_grad()
    def generate(
        self, z: torch.Tensor, truncation_psi: float = 0.7
    ) -> list[Image.Image]:
        """Generate images from z-space (full pipeline through mapping network).

        For breeding, prefer generate_from_w() to avoid re-mapping.
        """
        z = z.to(self.device)
        images = self.model(z, truncation_psi=truncation_psi)
        return self._to_pil(images)

    def random_latent(self, count: int = 1) -> torch.Tensor:
        """Generate random z-space vectors."""
        return torch.randn(count, self.z_dim)

    @torch.no_grad()
    def generate_random(
        self, count: int = 1, truncation_psi: float = 0.7
    ) -> tuple[list[Image.Image], torch.Tensor]:
        """Generate random images, returning images and their W vectors.

        This is the primary generation entry point. It:
        1. Samples random z vectors
        2. Maps through the mapping network to get w vectors
        3. Generates images from w
        4. Returns both images and w (for storage in genomes)

        Returns:
            (images, w_vectors): PIL images and the W vectors used.
        """
        z = self.random_latent(count)
        w = self.map_z_to_w(z, truncation_psi=truncation_psi)
        images = self.generate_from_w(w)
        return images, w

    def truncate_w(self, w: torch.Tensor, psi: float = 0.7) -> torch.Tensor:
        """Apply truncation trick to a W vector.

        Pulls the W vector toward the learned mean, trading diversity
        for quality. psi=1.0 means no change; psi=0.0 collapses to mean.

        Args:
            w: W vector(s) to truncate.
            psi: Truncation strength.

        Returns:
            Truncated W vector(s).
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
