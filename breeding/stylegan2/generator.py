"""StyleGAN2 Generator: z -> w -> image.

Composes the MappingNetwork and SynthesisNetwork into a complete generator.
Provides both z-to-image and w-to-image paths for breeding/interpolation.
"""

import torch
import torch.nn as nn

from .mapping import MappingNetwork
from .synthesis import SynthesisNetwork


class StyleGAN2Generator(nn.Module):
    """Complete StyleGAN2 generator for inference.

    Two forward paths:
        forward(z, truncation_psi) — full pipeline: z -> mapping -> w -> synthesis -> image
        forward_from_w(w) — breeding path: w -> synthesis -> image
    """

    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        img_resolution: int = 512,
        img_channels: int = 3,
        mapping_num_layers: int = 8,
        channel_schedule: dict[int, int] | None = None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.mapping = MappingNetwork(z_dim, w_dim, mapping_num_layers)
        self.synthesis = SynthesisNetwork(
            w_dim, img_resolution, img_channels, channel_schedule
        )
        self.num_ws = self.synthesis.num_ws

    def forward(
        self,
        z: torch.Tensor,
        truncation_psi: float = 1.0,
    ) -> torch.Tensor:
        """Full pipeline: z -> w -> image.

        Args:
            z: Latent vectors (B, z_dim).
            truncation_psi: Truncation strength. 1.0 = no truncation.

        Returns:
            RGB images (B, 3, H, W) in [-1, 1].
        """
        w = self.mapping(z, truncation_psi=truncation_psi)
        ws = w.unsqueeze(1).expand(-1, self.num_ws, -1)
        return self.synthesis(ws)

    def forward_from_w(
        self,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Breeding path: w -> image (bypasses mapping network).

        Args:
            w: W vectors. Either:
                - (B, w_dim): broadcast to all style layers
                - (B, num_ws, w_dim): per-layer styles (for style mixing)

        Returns:
            RGB images (B, 3, H, W) in [-1, 1].
        """
        if w.ndim == 2:
            ws = w.unsqueeze(1).expand(-1, self.num_ws, -1)
        else:
            ws = w
        return self.synthesis(ws)

    @classmethod
    def load_from_nvidia(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> "StyleGAN2Generator":
        """Load from a portable .pt checkpoint (output of convert_checkpoint.py).

        The conversion script extracts weights from NVIDIA's .pkl format and
        remaps them to match our naming conventions.

        Args:
            checkpoint_path: Path to the converted .pt file.
            device: Device to load the model onto.

        Returns:
            Initialized StyleGAN2Generator with loaded weights.
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        meta = ckpt["metadata"]

        # Reconstruct channel schedule if provided
        channel_schedule = meta.get("channel_schedule", None)

        model = cls(
            z_dim=meta["z_dim"],
            w_dim=meta["w_dim"],
            img_resolution=meta["img_resolution"],
            img_channels=meta["img_channels"],
            mapping_num_layers=meta["mapping_num_layers"],
            channel_schedule=channel_schedule,
        )

        # Load state dict — the conversion script already remaps keys to match
        # our naming convention
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()

        return model
