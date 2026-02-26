"""Latent space interpolation methods for GAN breeding."""

import torch


def lerp(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
    """Linear interpolation between two latent vectors.

    Args:
        z1: First latent vector.
        z2: Second latent vector.
        t: Interpolation factor in [0, 1]. Returns z1 at t=0, z2 at t=1.

    Returns:
        Interpolated latent vector.
    """
    return (1.0 - t) * z1 + t * z2


def slerp(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two latent vectors.

    Interpolates along the great circle on the hypersphere.
    Preferred over lerp for GAN latent spaces because it maintains
    constant angular velocity and avoids low-density regions.

    Args:
        z1: First latent vector.
        z2: Second latent vector.
        t: Interpolation factor in [0, 1].

    Returns:
        Interpolated latent vector.
    """
    z1_norm = z1 / (torch.norm(z1) + 1e-8)
    z2_norm = z2 / (torch.norm(z2) + 1e-8)

    dot = torch.clamp(torch.dot(z1_norm.flatten(), z2_norm.flatten()), -1.0, 1.0)
    omega = torch.acos(dot)

    if omega.abs() < 1e-6:
        return lerp(z1, z2, t)

    sin_omega = torch.sin(omega)
    coeff1 = torch.sin((1.0 - t) * omega) / sin_omega
    coeff2 = torch.sin(t * omega) / sin_omega

    return coeff1 * z1 + coeff2 * z2


def interpolation_strip(
    z1: torch.Tensor, z2: torch.Tensor, steps: int, method: str = "slerp"
) -> list[torch.Tensor]:
    """Generate a strip of interpolated latent vectors.

    Args:
        z1: First latent vector.
        z2: Second latent vector.
        steps: Number of interpolation steps (including endpoints).
        method: "slerp" or "lerp".

    Returns:
        List of interpolated latent vectors.
    """
    fn = slerp if method == "slerp" else lerp
    return [fn(z1, z2, t) for t in torch.linspace(0, 1, steps)]
