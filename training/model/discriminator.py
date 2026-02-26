"""FastGAN Discriminator with self-supervised reconstruction head.

Multi-scale discriminator that processes images at full resolution.
Includes an auxiliary decoder that reconstructs a downscaled version
of the input for self-supervised regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    """Downsample + Conv + BatchNorm + LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class DownBlockComp(nn.Module):
    """Compressed downsample block with two paths (skip + main) for stability."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.skip = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.main(x) + self.skip(x)) / 2.0


class SimpleDecoder(nn.Module):
    """Simple decoder for self-supervised reconstruction.

    Takes intermediate features and reconstructs a 128x128 RGB image
    that should match a downscaled version of the input.
    """

    def __init__(self, in_channels: int, target_channels: int = 3):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.GLU(dim=1),
        )  # channels: in -> in//4 (GLU halves)
        c1 = in_channels // 4

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c1, c1 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c1 // 2),
            nn.GLU(dim=1),
        )
        c2 = c1 // 4

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c2, c2 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c2 // 2),
            nn.GLU(dim=1),
        )
        c3 = c2 // 4

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c3, c3 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c3 // 2),
            nn.GLU(dim=1),
        )
        c4 = c3 // 4

        self.to_rgb = nn.Sequential(
            nn.Conv2d(c4, target_channels, 1, 1, 0),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.to_rgb(x)


class Discriminator(nn.Module):
    """FastGAN Discriminator for 512x512 images.

    Architecture:
        512x512 → encode path → 8x8 features
        Two heads:
        1. Classification head → real/fake logit
        2. Reconstruction head → reconstruct 128x128 image
           (self-supervised: compared against downscaled input)

    Encoder channels: 3→16→32→64→128→256→256→256→256
    """

    def __init__(self, base_channels: int = 16):
        super().__init__()
        ch = base_channels

        # Initial: 512→256
        self.from_rgb = nn.Sequential(
            nn.Conv2d(3, ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Encode path: 512→256→128→64→32→16→8
        self.down_256 = DownBlockComp(ch, ch * 2)       # 16→32
        self.down_128 = DownBlockComp(ch * 2, ch * 4)    # 32→64
        self.down_64 = DownBlockComp(ch * 4, ch * 8)     # 64→128
        self.down_32 = DownBlockComp(ch * 8, ch * 16)    # 128→256
        self.down_16 = DownBlock(ch * 16, ch * 16)        # 256→256
        self.down_8 = DownBlock(ch * 16, ch * 16)         # 256→256

        # Classification head: 8x8 → logit
        self.classify = nn.Sequential(
            nn.Conv2d(ch * 16, ch * 16, 4, 1, 0),
            nn.BatchNorm2d(ch * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 16, 1, 1, 1, 0),
        )

        # Self-supervised reconstruction head
        # Takes features from the 32x32 level and reconstructs 128x128
        # (32x32 with 4 upsample steps × 2 = 512, but SimpleDecoder
        # with GLU does 4 upsamples: 32→64→128→256→512... too big.
        # Instead, take 16x16 features: 16→32→64→128 (4 ups with GLU halving))
        # Actually let's take 8x8 features: 8→16→32→64→128
        self.decoder = SimpleDecoder(ch * 16)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: Input image tensor (B, 3, 512, 512) in [-1, 1].

        Returns:
            (logits, reconstruction):
                logits: (B, 1, 1, 1) real/fake prediction
                reconstruction: (B, 3, 128, 128) reconstructed downscaled image or None
        """
        feat = self.from_rgb(x)     # 512, 16ch
        feat = self.down_256(feat)    # 256, 32ch
        feat = self.down_128(feat)    # 128, 64ch
        feat = self.down_64(feat)     # 64, 128ch
        feat_32 = self.down_32(feat)  # 32, 256ch
        feat_16 = self.down_16(feat_32)  # 16, 256ch
        feat_8 = self.down_8(feat_16)    # 8, 256ch

        # Classification
        logits = self.classify(feat_8)  # (B, 1, 5, 5) → need to flatten
        logits = logits.view(logits.size(0), -1).mean(dim=1, keepdim=True)

        # Reconstruction from 8x8 features
        recon = self.decoder(feat_8)

        return logits, recon
