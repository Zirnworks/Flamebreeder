"""StyleGAN2 Mapping Network: z -> w.

Maps random Gaussian noise vectors to a learned intermediate latent space
that has smoother, more disentangled properties for interpolation and breeding.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate.

    NVIDIA StyleGAN2 initializes weights from N(0,1) and applies a runtime
    scaling factor of lr_mul / sqrt(fan_in). This keeps the effective learning
    rate equal across layers regardless of fan-in, improving training stability.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, lr_mul: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = lr_mul / math.sqrt(in_features)
        self.lr_mul = lr_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.scale
        b = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, w, b)


class MappingNetwork(nn.Module):
    """StyleGAN2 Mapping Network.

    Architecture: z (z_dim) -> pixel_norm -> 8x [FC + LeakyReLU] -> w (w_dim)

    The mapping network transforms raw Gaussian noise into a learned latent
    space W that is much better structured for interpolation. This is the key
    architectural difference from FastGAN that enables Artbreeder-style breeding.

    Tracks a running average of w vectors (w_avg) for the truncation trick.
    """

    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        num_layers: int = 8,
        lr_mul: float = 0.01,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.append(EqualizedLinear(in_dim, w_dim, lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

        # Running average of W for truncation trick
        self.register_buffer("w_avg", torch.zeros(w_dim))

    def forward(
        self,
        z: torch.Tensor,
        truncation_psi: float = 1.0,
    ) -> torch.Tensor:
        """Map z to w with optional truncation.

        Args:
            z: Latent vectors (B, z_dim).
            truncation_psi: Truncation strength. 1.0 = no truncation,
                0.0 = collapse to mean w. Typical range: 0.5-1.0.

        Returns:
            W vectors (B, w_dim).
        """
        # Pixel normalization: normalize z to unit variance
        x = z * (z.square().mean(dim=1, keepdim=True) + 1e-8).rsqrt()

        w = self.net(x)

        # Truncation trick: pull toward the mean
        if truncation_psi < 1.0:
            w = self.w_avg.lerp(w, truncation_psi)

        return w
