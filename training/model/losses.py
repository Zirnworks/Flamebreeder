"""Loss functions for FastGAN training.

Hinge loss for adversarial training + MSE reconstruction loss
for the self-supervised discriminator head.
"""

import torch
import torch.nn.functional as F


def hinge_loss_d(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Discriminator hinge loss.

    D wants: real_logits > 1, fake_logits < -1
    """
    loss_real = F.relu(1.0 - real_logits).mean()
    loss_fake = F.relu(1.0 + fake_logits).mean()
    return loss_real + loss_fake


def hinge_loss_g(fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator hinge loss.

    G wants: fake_logits to be as large as possible.
    """
    return -fake_logits.mean()


def reconstruction_loss(
    recon: torch.Tensor, target: torch.Tensor, target_size: int = 128
) -> torch.Tensor:
    """Self-supervised reconstruction loss.

    Compares discriminator's reconstruction against a downscaled input.
    Uses MSE loss.

    Args:
        recon: Reconstructed image from discriminator decoder (B, 3, H, W).
        target: Original input image (B, 3, H_orig, W_orig).
        target_size: Size to downscale target to match reconstruction.
    """
    # Downscale target to match reconstruction size
    if target.size(-1) != target_size:
        target_down = F.interpolate(
            target, size=target_size, mode="bilinear", align_corners=False
        )
    else:
        target_down = target

    # Match sizes if reconstruction differs
    if recon.size(-1) != target_down.size(-1):
        recon = F.interpolate(
            recon, size=target_down.size(-1), mode="bilinear", align_corners=False
        )

    return F.mse_loss(recon, target_down)
