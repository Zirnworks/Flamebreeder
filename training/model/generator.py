"""FastGAN Generator with Skip-Layer Excitation (SLE) blocks.

Architecture based on "Towards Faster and Stabilized GAN Training
for High-fidelity Few-shot Image Synthesis" (Liu et al., ICLR 2021).

z ∈ R^256 → Linear → Reshape(256, 4, 4) → Upsample blocks → 512x512 RGB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    """Gated Linear Unit activation (channel-wise)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nc = x.size(1)
        assert nc % 2 == 0
        return x[:, :nc // 2] * torch.sigmoid(x[:, nc // 2:])


class SkipLayerExcitation(nn.Module):
    """Skip-Layer Excitation (SLE) module.

    Takes a low-resolution feature map and uses it to modulate
    (excite) a high-resolution feature map via channel-wise scaling.
    """

    def __init__(self, low_channels: int, high_channels: int):
        super().__init__()
        self.transform = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(low_channels, high_channels, 4, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(high_channels, high_channels, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, low_res: torch.Tensor, high_res: torch.Tensor) -> torch.Tensor:
        excitation = self.transform(low_res)
        return high_res * excitation


class UpsampleBlock(nn.Module):
    """Upsample + Conv + BatchNorm + GLU block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # GLU halves channels, so we produce 2x out_channels then gate
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            GLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class Generator(nn.Module):
    """FastGAN Generator for 512x512 output.

    Architecture:
        z (256) → FC → Reshape (256, 4, 4)
        → Up blocks: 4→8→16→32→64→128→256→512
        → SLE connections from low-res to high-res features
        → Conv 1x1 → 3ch → Tanh

    SLE connections:
        8x8   → excites 64x64
        16x16 → excites 128x128
        32x32 → excites 256x256
        64x64 → excites 512x512
    """

    def __init__(self, latent_dim: int = 256, base_channels: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # Initial projection: z → (base_channels, 4, 4)
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, base_channels * 4 * 4, bias=False),
            nn.BatchNorm1d(base_channels * 4 * 4),
            nn.GLU(dim=1),
        )
        # After GLU, channels are halved: base_channels * 4 * 4 / 2
        # But we want base_channels * 4 * 4 output, so double the linear
        # Actually, let's use a simpler approach:
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, base_channels * 16, bias=False),
        )
        self.initial_bn = nn.BatchNorm2d(base_channels)

        ch = base_channels  # 256

        # Upsample blocks: 4→8→16→32→64→128→256→512
        self.up_4_to_8 = UpsampleBlock(ch, ch)          # 256→256, 4→8
        self.up_8_to_16 = UpsampleBlock(ch, ch)          # 256→256, 8→16
        self.up_16_to_32 = UpsampleBlock(ch, ch)          # 256→256, 16→32
        self.up_32_to_64 = UpsampleBlock(ch, ch // 2)     # 256→128, 32→64
        self.up_64_to_128 = UpsampleBlock(ch // 2, ch // 4)  # 128→64, 64→128
        self.up_128_to_256 = UpsampleBlock(ch // 4, ch // 8)  # 64→32, 128→256
        self.up_256_to_512 = UpsampleBlock(ch // 8, ch // 16)  # 32→16, 256→512

        # SLE connections
        self.sle_8_to_64 = SkipLayerExcitation(ch, ch // 2)       # 256→128
        self.sle_16_to_128 = SkipLayerExcitation(ch, ch // 4)      # 256→64
        self.sle_32_to_256 = SkipLayerExcitation(ch, ch // 8)      # 256→32
        self.sle_64_to_512 = SkipLayerExcitation(ch // 2, ch // 16)  # 128→16

        # Final output: 16ch → 3ch RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(ch // 16, 3, 1, 1, 0),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)

        # Project and reshape to (B, 256, 4, 4)
        x = self.initial(z)
        x = x.view(batch_size, -1, 4, 4)
        x = F.leaky_relu(self.initial_bn(x), 0.2)

        # Upsample with SLE connections
        feat_8 = self.up_4_to_8(x)         # 8x8, 256ch
        feat_16 = self.up_8_to_16(feat_8)   # 16x16, 256ch
        feat_32 = self.up_16_to_32(feat_16)  # 32x32, 256ch

        feat_64 = self.up_32_to_64(feat_32)  # 64x64, 128ch
        feat_64 = self.sle_8_to_64(feat_8, feat_64)

        feat_128 = self.up_64_to_128(feat_64)  # 128x128, 64ch
        feat_128 = self.sle_16_to_128(feat_16, feat_128)

        feat_256 = self.up_128_to_256(feat_128)  # 256x256, 32ch
        feat_256 = self.sle_32_to_256(feat_32, feat_256)

        feat_512 = self.up_256_to_512(feat_256)  # 512x512, 16ch
        feat_512 = self.sle_64_to_512(feat_64, feat_512)

        return self.to_rgb(feat_512)
