"""Differentiable Augmentation for GAN training (DiffAugment).

Based on "Differentiable Augmentation for Data-Efficient GAN Training"
(Zhao et al., 2020). Augmentations are applied to both real and fake
images before feeding to the discriminator.
"""

import torch
import torch.nn.functional as F


def diff_augment(
    x: torch.Tensor,
    ops: list[str] | None = None,
    prob: float = 0.5,
) -> torch.Tensor:
    """Apply differentiable augmentations.

    Args:
        x: Image tensor (B, C, H, W) in [-1, 1].
        ops: List of augmentation operations. Options:
            "color", "translation", "cutout".
            If None, applies all three.
        prob: Probability of applying each augmentation.

    Returns:
        Augmented tensor (same shape, same device).
    """
    if ops is None:
        ops = ["color", "translation", "cutout"]

    for op in ops:
        if torch.rand(1).item() < prob:
            if op == "color":
                x = _aug_color(x)
            elif op == "translation":
                x = _aug_translation(x)
            elif op == "cutout":
                x = _aug_cutout(x)

    return x


def _aug_color(x: torch.Tensor) -> torch.Tensor:
    """Random brightness, saturation, and contrast adjustment."""
    batch_size = x.size(0)

    # Random brightness
    brightness = torch.rand(batch_size, 1, 1, 1, device=x.device) - 0.5
    x = x + brightness

    # Random saturation
    mean = x.mean(dim=1, keepdim=True)
    sat_factor = torch.rand(batch_size, 1, 1, 1, device=x.device) * 2
    x = mean + (x - mean) * sat_factor

    # Random contrast
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    con_factor = torch.rand(batch_size, 1, 1, 1, device=x.device) + 0.5
    x = mean + (x - mean) * con_factor

    return x


def _aug_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """Random translation with zero padding."""
    batch_size, _, h, w = x.shape
    shift_h = int(h * ratio)
    shift_w = int(w * ratio)

    # Random shift amounts
    th = torch.randint(-shift_h, shift_h + 1, (batch_size, 1, 1), device=x.device)
    tw = torch.randint(-shift_w, shift_w + 1, (batch_size, 1, 1), device=x.device)

    # Create grid for grid_sample
    grid_h = torch.arange(h, device=x.device).float()
    grid_w = torch.arange(w, device=x.device).float()
    grid_h = grid_h.view(1, h, 1).expand(batch_size, h, w)
    grid_w = grid_w.view(1, 1, w).expand(batch_size, h, w)

    grid_h = 2.0 * (grid_h - th.float()) / (h - 1) - 1.0
    grid_w = 2.0 * (grid_w - tw.float()) / (w - 1) - 1.0
    grid = torch.stack([grid_w, grid_h], dim=-1)

    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=True)


def _aug_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Random rectangular cutout (set region to zero)."""
    batch_size, _, h, w = x.shape
    cut_h = int(h * ratio)
    cut_w = int(w * ratio)

    # Random offset
    offset_h = torch.randint(0, h - cut_h + 1, (batch_size, 1, 1, 1), device=x.device)
    offset_w = torch.randint(0, w - cut_w + 1, (batch_size, 1, 1, 1), device=x.device)

    # Create mask
    grid_h = torch.arange(h, device=x.device).view(1, 1, h, 1).expand(batch_size, 1, h, w)
    grid_w = torch.arange(w, device=x.device).view(1, 1, 1, w).expand(batch_size, 1, h, w)

    mask_h = (grid_h >= offset_h) & (grid_h < offset_h + cut_h)
    mask_w = (grid_w >= offset_w) & (grid_w < offset_w + cut_w)
    mask = (~(mask_h & mask_w)).float()

    return x * mask
