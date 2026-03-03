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


def multi_keyframe_strip(
    ws: list[torch.Tensor], total_steps: int, method: str = "slerp"
) -> list[torch.Tensor]:
    """Generate interpolation strip through multiple keyframes.

    Distributes steps proportionally across segments based on W-space
    arc length. Adjacent segments share their boundary keyframe
    (deduplicated).

    Args:
        ws: List of keyframe W-vectors (at least 2).
        total_steps: Total number of output frames (including endpoints).
        method: "slerp" or "lerp".

    Returns:
        List of interpolated latent vectors.
    """
    if len(ws) < 2:
        raise ValueError("Need at least 2 keyframes")
    if len(ws) == 2:
        return interpolation_strip(ws[0], ws[1], total_steps, method)

    # Compute arc lengths between consecutive keyframes
    arc_lengths = []
    for i in range(len(ws) - 1):
        arc_lengths.append(torch.norm(ws[i + 1] - ws[i]).item())

    total_arc = sum(arc_lengths)
    if total_arc < 1e-8:
        # All keyframes are identical — distribute evenly
        arc_lengths = [1.0] * len(arc_lengths)
        total_arc = len(arc_lengths)

    # Distribute steps proportionally (minimum 2 per segment)
    n_segments = len(ws) - 1
    remaining = total_steps - n_segments  # reserve 1 per segment for dedup
    segment_steps = []
    for arc in arc_lengths:
        s = max(2, round((arc / total_arc) * remaining) + 1)
        segment_steps.append(s)

    # Adjust to hit exact total
    while sum(segment_steps) - (n_segments - 1) > total_steps:
        idx = max(range(n_segments), key=lambda i: segment_steps[i])
        segment_steps[idx] -= 1
    while sum(segment_steps) - (n_segments - 1) < total_steps:
        idx = min(range(n_segments), key=lambda i: segment_steps[i])
        segment_steps[idx] += 1

    # Generate each segment and concatenate (dedup boundary keyframes)
    result = []
    for i in range(n_segments):
        strip = interpolation_strip(ws[i], ws[i + 1], segment_steps[i], method)
        if i == 0:
            result.extend(strip)
        else:
            result.extend(strip[1:])  # skip first (duplicate of prev last)

    return result
