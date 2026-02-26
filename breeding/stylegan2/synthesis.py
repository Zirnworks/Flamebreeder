"""StyleGAN2 Synthesis Network: w -> image.

Transforms W-space latent vectors into images through style-modulated
convolutions with weight demodulation, noise injection, and progressive
upsampling with skip connections.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mapping import EqualizedLinear


class ModulatedConv2d(nn.Module):
    """Style-modulated convolution with weight demodulation.

    The core StyleGAN2 operation:
    1. A style vector (from W) modulates the conv weights per input channel
    2. Weights are demodulated (normalized to unit variance) for stability
    3. Standard convolution is applied

    This allows the style to control "what" is generated while the conv
    weights control "how" — enabling smooth, meaningful latent traversals.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        demodulate: bool = True,
        upsample: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.padding = kernel_size // 2

        # Conv weight: [out_ch, in_ch, kH, kW]
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.scale = 1.0 / math.sqrt(in_channels * kernel_size ** 2)

        # Style affine transform: w -> per-channel modulation
        self.style_linear = EqualizedLinear(style_dim, in_channels, bias=True)
        # Initialize bias to 1 so modulation starts as identity
        self.style_linear.bias.data.fill_(1.0)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch, in_ch, h, w_size = x.shape

        # Style modulation: (B, in_ch)
        s = self.style_linear(style)

        # Scale weight and modulate
        # weight: (1, out, in, k, k) * style: (B, 1, in, 1, 1) -> (B, out, in, k, k)
        weight = self.weight.unsqueeze(0) * self.scale
        weight = weight * s.reshape(batch, 1, -1, 1, 1)

        # Demodulate: normalize each output channel to unit std
        if self.demodulate:
            dcoef = (weight.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
            weight = weight * dcoef.reshape(batch, -1, 1, 1, 1)

        # Upsample input if needed
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            h *= 2
            w_size *= 2

        # Grouped convolution: treat batch as groups
        # Reshape: (1, B*in_ch, H, W) with groups=B
        x = x.reshape(1, batch * in_ch, h, w_size)
        weight = weight.reshape(
            batch * self.out_channels, in_ch, self.kernel_size, self.kernel_size
        )
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        return out.reshape(batch, self.out_channels, h, w_size)


class NoiseInjection(nn.Module):
    """Per-pixel noise injection with learned per-channel scaling.

    Adds stochastic detail (e.g., fine texture variation) that doesn't
    depend on the latent code. This decorrelates fine details from the
    overall image structure, improving quality.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        return x + self.weight * noise


class StyleBlock(nn.Module):
    """One styled convolution block: ModConv -> Noise -> Bias -> LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        upsample: bool = False,
    ):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels, out_channels, kernel_size, style_dim, upsample=upsample
        )
        self.noise = NoiseInjection(out_channels)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.conv(x, style)
        x = self.noise(x, noise)
        x = x + self.bias
        x = self.act(x)
        return x


class ToRGB(nn.Module):
    """Modulated 1x1 convolution to RGB output (no demodulation)."""

    def __init__(self, in_channels: int, style_dim: int):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels, 3, kernel_size=1, style_dim=style_dim, demodulate=False
        )
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return self.conv(x, style) + self.bias


class SynthesisBlock(nn.Module):
    """One resolution block of the synthesis network.

    Each block contains:
    - Two StyleBlocks (with optional upsample on the first)
    - A ToRGB layer whose output is accumulated via skip connections

    For the first block (4x4), the input is a learned constant and the
    first conv does NOT upsample.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        is_first: bool = False,
    ):
        super().__init__()
        self.is_first = is_first

        if is_first:
            # First block: no upsample, input comes from learned constant
            self.conv0 = StyleBlock(in_channels, out_channels, 3, style_dim, upsample=False)
        else:
            # Subsequent blocks: upsample on first conv
            self.conv0 = StyleBlock(in_channels, out_channels, 3, style_dim, upsample=True)

        self.conv1 = StyleBlock(out_channels, out_channels, 3, style_dim, upsample=False)
        self.to_rgb = ToRGB(out_channels, style_dim)

    def forward(
        self,
        x: torch.Tensor,
        rgb: torch.Tensor | None,
        styles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Feature tensor from previous block.
            rgb: Accumulated RGB image (None for first block).
            styles: (B, 2, w_dim) — two style vectors for this block's two convs.
                    Actually uses 3 styles: conv0, conv1, and toRGB.
                    But NVIDIA packs it as 2 per block + toRGB uses the second.

        Returns:
            (features, rgb): Updated feature tensor and accumulated RGB.
        """
        # Style for conv0 and conv1
        x = self.conv0(x, styles[:, 0])
        x = self.conv1(x, styles[:, 1])

        # toRGB uses the second style (same as conv1 in NVIDIA's impl)
        rgb_new = self.to_rgb(x, styles[:, 1])

        # Accumulate RGB via skip: upsample previous + add new
        if rgb is not None:
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear", align_corners=False)
            rgb = rgb + rgb_new
        else:
            rgb = rgb_new

        return x, rgb


class SynthesisNetwork(nn.Module):
    """StyleGAN2 Synthesis Network.

    Transforms a sequence of W vectors (one per style injection) into
    a 512x512 RGB image through style-modulated convolutions.

    Architecture:
        Learned constant (4x4) -> SynthesisBlock(4x4) -> ... -> SynthesisBlock(512x512)
        With skip connections accumulating RGB at each resolution.

    Channel schedule for 512x512 (NVIDIA cfg=auto default):
        4:512, 8:512, 16:512, 32:512, 64:256, 128:128, 256:64, 512:32
    """

    # Default channel schedule matching NVIDIA's auto config for 512x512
    DEFAULT_CHANNELS = {
        4: 512, 8: 512, 16: 512, 32: 512,
        64: 256, 128: 128, 256: 64, 512: 32,
    }

    def __init__(
        self,
        w_dim: int = 512,
        img_resolution: int = 512,
        img_channels: int = 3,
        channel_schedule: dict[int, int] | None = None,
    ):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        channels = channel_schedule or self.DEFAULT_CHANNELS

        # Resolution progression: 4, 8, 16, ..., img_resolution
        log2_res = int(math.log2(img_resolution))
        self.block_resolutions = [2 ** i for i in range(2, log2_res + 1)]

        # Each block consumes 2 style vectors (conv0 + conv1)
        self.num_ws = len(self.block_resolutions) * 2

        # Learned constant input at 4x4
        self.const = nn.Parameter(torch.randn(1, channels[4], 4, 4))

        # Build synthesis blocks
        self.blocks = nn.ModuleDict()
        for i, res in enumerate(self.block_resolutions):
            in_ch = channels[4] if res == 4 else channels[res // 2]
            out_ch = channels[res]
            self.blocks[str(res)] = SynthesisBlock(
                in_ch, out_ch, w_dim, is_first=(res == 4)
            )

    def forward(self, ws: torch.Tensor) -> torch.Tensor:
        """Generate image from style vectors.

        Args:
            ws: Style vectors (B, num_ws, w_dim). If (B, w_dim), broadcast
                to all style layers.

        Returns:
            RGB image tensor (B, 3, img_resolution, img_resolution) in [-1, 1].
        """
        if ws.ndim == 2:
            ws = ws.unsqueeze(1).expand(-1, self.num_ws, -1)

        batch = ws.shape[0]
        x = self.const.expand(batch, -1, -1, -1)
        rgb = None
        w_idx = 0

        for res in self.block_resolutions:
            block = self.blocks[str(res)]
            # Each block gets 2 style vectors
            block_styles = ws[:, w_idx:w_idx + 2]
            x, rgb = block(x, rgb, block_styles)
            w_idx += 2

        # Clamp output to [-1, 1]
        return rgb.clamp(-1, 1)
