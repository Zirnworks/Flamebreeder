#!/usr/bin/env python3
"""Convert NVIDIA StyleGAN2-ADA .pkl checkpoint to portable .pt format.

Runs on the cloud instance where dnnlib/torch_utils are available.
Extracts G_ema weights and remaps them to match our clean local
StyleGAN2 generator's naming conventions.

Usage:
    python convert_checkpoint.py <input.pkl> <output.pt>

The output .pt file contains:
    - state_dict: Weights remapped to our naming convention
    - metadata: Architecture config (z_dim, w_dim, resolution, etc.)
    - source: "stylegan2-ada-pytorch"
"""

import sys
import math
import re
from pathlib import Path

import torch


def load_nvidia_generator(pkl_path: str):
    """Load G_ema from NVIDIA's .pkl checkpoint."""
    # NVIDIA's pickles need their module path
    sys.path.insert(0, "/workspace/stylegan2-ada-pytorch")
    import dnnlib
    import legacy

    print(f"Loading {pkl_path}...")
    with dnnlib.util.open_url(pkl_path) as f:
        data = legacy.load_network_pkl(f)

    G_ema = data["G_ema"]
    print(f"  z_dim: {G_ema.z_dim}, w_dim: {G_ema.w_dim}")
    print(f"  img_resolution: {G_ema.img_resolution}")
    print(f"  img_channels: {G_ema.img_channels}")
    return G_ema


def extract_channel_schedule(G_ema) -> dict:
    """Extract the actual channel schedule from the trained model."""
    channels = {}
    for name, module in G_ema.synthesis.named_children():
        if hasattr(module, "conv1") and hasattr(module.conv1, "out_channels"):
            # Block names are like 'b4', 'b8', ..., 'b512'
            res = int(name.lstrip("b"))
            channels[res] = module.conv1.out_channels
    return channels


def remap_mapping_weights(G_ema) -> dict:
    """Extract and remap mapping network weights.

    NVIDIA naming:  mapping.fc0.weight, mapping.fc0.bias, ...
    Our naming:     mapping.net.0.weight, mapping.net.0.bias, ...
                    (Sequential: [Linear, LeakyReLU, Linear, LeakyReLU, ...])
    """
    state = {}
    num_layers = G_ema.mapping.num_layers

    for i in range(num_layers):
        nvidia_prefix = f"fc{i}"

        # Get the NVIDIA layer
        nvidia_layer = getattr(G_ema.mapping, nvidia_prefix)

        # Our Sequential index: each layer is (EqualizedLinear, LeakyReLU)
        # So fc0 -> net.0, fc1 -> net.2, fc2 -> net.4, etc.
        our_idx = i * 2

        state[f"mapping.net.{our_idx}.weight"] = nvidia_layer.weight.data.cpu()
        state[f"mapping.net.{our_idx}.bias"] = nvidia_layer.bias.data.cpu()

    # w_avg buffer
    if hasattr(G_ema.mapping, "w_avg"):
        state["mapping.w_avg"] = G_ema.mapping.w_avg.cpu()

    return state


def remap_synthesis_weights(G_ema) -> dict:
    """Extract and remap synthesis network weights.

    NVIDIA's synthesis network uses named blocks like:
        synthesis.b4.conv1.weight
        synthesis.b4.conv1.noise_strength
        synthesis.b8.conv0.weight
        synthesis.b8.conv0.up_filter  (for upsampling blocks)
        etc.

    Our naming uses resolution-keyed ModuleDict:
        synthesis.blocks.4.conv0.conv.weight
        synthesis.blocks.4.conv0.noise.weight
        synthesis.blocks.4.conv1.conv.weight
        etc.
    """
    state = {}
    log2_res = int(math.log2(G_ema.img_resolution))
    resolutions = [2 ** i for i in range(2, log2_res + 1)]

    # Learned constant
    state["synthesis.const"] = G_ema.synthesis.b4.const.data.cpu()

    for res in resolutions:
        nvidia_block = getattr(G_ema.synthesis, f"b{res}")

        # conv0 (first conv, upsample for res > 4)
        _remap_style_conv(
            nvidia_block.conv0 if hasattr(nvidia_block, "conv0") else nvidia_block.conv1,
            f"synthesis.blocks.{res}.conv0",
            state,
        )

        # conv1 (second conv, no upsample)
        _remap_style_conv(
            nvidia_block.conv1,
            f"synthesis.blocks.{res}.conv1",
            state,
        )

        # toRGB
        _remap_to_rgb(nvidia_block.torgb, f"synthesis.blocks.{res}.to_rgb", state)

    return state


def _remap_style_conv(nvidia_conv, our_prefix: str, state: dict):
    """Remap a single styled convolution block.

    NVIDIA conv has: weight, bias, noise_const, noise_strength, affine.weight, affine.bias
    Our StyleBlock has: conv (ModulatedConv2d), noise (NoiseInjection), bias

    NVIDIA's ModulatedConv2d: weight, affine (for style)
    Our ModulatedConv2d: weight, style_linear (for style)
    """
    # Main convolution weight
    state[f"{our_prefix}.conv.weight"] = nvidia_conv.weight.data.cpu()

    # Style affine transform
    state[f"{our_prefix}.conv.style_linear.weight"] = nvidia_conv.affine.weight.data.cpu()
    state[f"{our_prefix}.conv.style_linear.bias"] = nvidia_conv.affine.bias.data.cpu()

    # Noise injection weight (noise_strength in NVIDIA)
    if hasattr(nvidia_conv, "noise_strength"):
        # NVIDIA stores as scalar; we store as (1, channels, 1, 1)
        noise_w = nvidia_conv.noise_strength.data.cpu()
        if noise_w.ndim == 0:
            channels = nvidia_conv.weight.shape[0]
            noise_w = noise_w.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, channels, 1, 1).clone()
        state[f"{our_prefix}.noise.weight"] = noise_w

    # Bias
    if hasattr(nvidia_conv, "bias") and nvidia_conv.bias is not None:
        bias = nvidia_conv.bias.data.cpu()
        if bias.ndim == 1:
            bias = bias.reshape(1, -1, 1, 1)
        state[f"{our_prefix}.bias"] = bias


def _remap_to_rgb(nvidia_torgb, our_prefix: str, state: dict):
    """Remap a toRGB layer."""
    state[f"{our_prefix}.conv.weight"] = nvidia_torgb.weight.data.cpu()
    state[f"{our_prefix}.conv.style_linear.weight"] = nvidia_torgb.affine.weight.data.cpu()
    state[f"{our_prefix}.conv.style_linear.bias"] = nvidia_torgb.affine.bias.data.cpu()

    if hasattr(nvidia_torgb, "bias") and nvidia_torgb.bias is not None:
        bias = nvidia_torgb.bias.data.cpu()
        if bias.ndim == 1:
            bias = bias.reshape(1, -1, 1, 1)
        state[f"{our_prefix}.bias"] = bias


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_checkpoint.py <input.pkl> <output.pt>")
        sys.exit(1)

    pkl_path = sys.argv[1]
    output_path = sys.argv[2]

    G_ema = load_nvidia_generator(pkl_path)

    print("Extracting and remapping weights...")

    state = {}
    state.update(remap_mapping_weights(G_ema))
    state.update(remap_synthesis_weights(G_ema))

    # Extract architecture metadata
    channel_schedule = extract_channel_schedule(G_ema)
    metadata = {
        "z_dim": G_ema.z_dim,
        "w_dim": G_ema.w_dim,
        "img_resolution": G_ema.img_resolution,
        "img_channels": G_ema.img_channels,
        "mapping_num_layers": G_ema.mapping.num_layers,
        "num_ws": G_ema.synthesis.num_ws,
        "channel_schedule": channel_schedule,
    }

    checkpoint = {
        "state_dict": state,
        "metadata": metadata,
        "source": "stylegan2-ada-pytorch",
    }

    print(f"Saving portable checkpoint to {output_path}...")
    torch.save(checkpoint, output_path)

    # Summary
    total_params = sum(p.numel() for p in state.values())
    print(f"\nExported {total_params:,} parameters")
    print(f"Architecture: z_dim={metadata['z_dim']}, w_dim={metadata['w_dim']}, "
          f"resolution={metadata['img_resolution']}")
    print(f"Channel schedule: {channel_schedule}")
    print(f"num_ws: {metadata['num_ws']}")
    print(f"\nPortable checkpoint ready: {output_path}")


if __name__ == "__main__":
    main()
