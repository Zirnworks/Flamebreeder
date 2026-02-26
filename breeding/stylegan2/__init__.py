"""Clean StyleGAN2 generator for local inference (MPS-compatible).

Pure PyTorch implementation matching NVIDIA's stylegan2-ada-pytorch architecture.
No custom CUDA kernels, no dnnlib, no torch_utils dependencies.
"""

from .generator import StyleGAN2Generator
from .mapping import MappingNetwork
from .synthesis import SynthesisNetwork

__all__ = ["StyleGAN2Generator", "MappingNetwork", "SynthesisNetwork"]
