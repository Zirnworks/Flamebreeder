"""MPS (Apple Silicon) compatibility utilities for PyTorch.

Handles known MPS pitfalls including:
- Environment setup for fallback ops
- Gradient validation
- Memory management
- Contiguity guards
"""

import os
import logging

import torch

logger = logging.getLogger(__name__)


def setup_mps_env():
    """Set environment variables for MPS compatibility."""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def get_device(preferred: str = "mps") -> torch.device:
    """Get the best available device.

    Prefers MPS on Apple Silicon, falls back to CPU.
    """
    if preferred == "mps" and torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon) device")
        return torch.device("mps")
    elif preferred == "cuda" and torch.cuda.is_available():
        logger.info("Using CUDA device")
        return torch.device("cuda")
    else:
        logger.info("Using CPU device")
        return torch.device("cpu")


def validate_gradients(
    model: torch.nn.Module, model_name: str, step: int
) -> bool:
    """Check that model parameters have non-zero gradients.

    Critical for MPS: there's a known bug where non-contiguous tensors
    can cause silent gradient zeros on PyTorch < 2.4.

    Returns True if gradients look healthy.
    """
    total = 0
    zero_grad = 0

    for name, p in model.named_parameters():
        if p.requires_grad:
            total += 1
            if p.grad is None or p.grad.abs().sum() == 0:
                zero_grad += 1

    if zero_grad > total * 0.1:  # More than 10% zero gradients is concerning
        logger.warning(
            f"Step {step}: {model_name} has {zero_grad}/{total} "
            f"parameters with zero gradients!"
        )
        return False

    return True


def mps_sync():
    """Synchronize MPS operations (useful for timing/debugging)."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def mps_empty_cache():
    """Free MPS memory cache."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
