"""Breeding operators for combining parent latent vectors.

Genetic algorithm-inspired operators that work in the GAN's latent space
to produce offspring with traits from both parents.
"""

import torch

from .interpolation import slerp


def breed_average(
    parent_a: torch.Tensor, parent_b: torch.Tensor, ratio: float = 0.5
) -> torch.Tensor:
    """Weighted average breeding via slerp.

    The simplest breeding method — smooth blend between parents.

    Args:
        parent_a: First parent latent vector.
        parent_b: Second parent latent vector.
        ratio: Blend ratio. 0.0 = pure parent_a, 1.0 = pure parent_b.
    """
    return slerp(parent_a, parent_b, ratio)


def breed_crossover(
    parent_a: torch.Tensor, parent_b: torch.Tensor, crossover_rate: float = 0.5
) -> torch.Tensor:
    """Uniform crossover — each dimension independently chosen from one parent.

    Creates more distinct offspring than averaging, with sharper feature
    mixing between parents.

    Args:
        parent_a: First parent latent vector.
        parent_b: Second parent latent vector.
        crossover_rate: Probability of choosing parent_b for each dimension.
    """
    mask = torch.rand_like(parent_a) < crossover_rate
    return torch.where(mask, parent_b, parent_a)


def breed_block_crossover(
    parent_a: torch.Tensor, parent_b: torch.Tensor, num_blocks: int = 4
) -> torch.Tensor:
    """Block crossover — divide latent vector into blocks, alternate parents.

    Contiguous blocks of dimensions come from the same parent, producing
    offspring that inherit "chunks" of structure.

    Args:
        parent_a: First parent latent vector.
        parent_b: Second parent latent vector.
        num_blocks: Number of blocks to divide the latent vector into.
    """
    dim = parent_a.shape[0]
    block_size = dim // num_blocks
    child = parent_a.clone()

    for i in range(num_blocks):
        if torch.rand(1).item() < 0.5:
            start = i * block_size
            end = start + block_size if i < num_blocks - 1 else dim
            child[start:end] = parent_b[start:end]

    return child


def mutate(
    z: torch.Tensor,
    mutation_rate: float = 0.1,
    mutation_strength: float = 0.3,
) -> torch.Tensor:
    """Random perturbation of latent dimensions.

    Args:
        z: Latent vector to mutate.
        mutation_rate: Probability of mutating each dimension.
        mutation_strength: Standard deviation of mutation noise.
    """
    mask = (torch.rand_like(z) < mutation_rate).float()
    noise = torch.randn_like(z) * mutation_strength
    return z + mask * noise


def breed_guided(
    parent_a: torch.Tensor,
    parent_b: torch.Tensor,
    alpha: float = 0.5,
    noise_strength: float = 0.1,
    max_norm: float = 2.5,
) -> torch.Tensor:
    """Guided breeding: slerp with added noise and truncation.

    Produces offspring near the midpoint of the parents but with
    random variation for diversity.

    Args:
        parent_a: First parent latent vector.
        parent_b: Second parent latent vector.
        alpha: Slerp interpolation factor.
        noise_strength: Standard deviation of added Gaussian noise.
        max_norm: Maximum L2 norm (truncation trick).
    """
    child = slerp(parent_a, parent_b, alpha)
    child = child + torch.randn_like(child) * noise_strength
    return truncate(child, max_norm)


def truncate(z: torch.Tensor, max_norm: float = 2.5) -> torch.Tensor:
    """Clamp latent vector L2 norm to prevent artifacts (Z-space).

    Points far from the origin in the Gaussian latent space are rare
    and often produce low-quality images. This truncation keeps vectors
    in the well-explored region.

    For W-space, use truncate_w() instead.

    Args:
        z: Latent vector.
        max_norm: Maximum allowed L2 norm.
    """
    norm = torch.norm(z)
    if norm > max_norm:
        z = z * (max_norm / norm)
    return z


def truncate_w(
    w: torch.Tensor,
    w_avg: torch.Tensor,
    psi: float = 0.7,
) -> torch.Tensor:
    """W-space truncation trick.

    Pulls W vectors toward the learned mean of the W distribution.
    This trades diversity for quality — outputs cluster around the
    "average" look of the training data.

    Args:
        w: W vector(s) to truncate.
        w_avg: Mean W vector from the mapping network.
        psi: Truncation strength. 1.0 = no change, 0.0 = collapse to mean.
    """
    return w_avg + (w - w_avg) * psi


def breed_style_mix(
    parent_a: torch.Tensor,
    parent_b: torch.Tensor,
    num_ws: int = 16,
    crossover_layer: int = 4,
) -> torch.Tensor:
    """Style mixing: coarse layers from parent A, fine layers from parent B.

    A uniquely StyleGAN2 breeding operator. The synthesis network injects
    style at each resolution layer. By using one parent's W for coarse
    layers (4x4 through ~32x32) and the other's for fine layers (~64x64
    through 512x512), we get offspring that inherit structure from one
    parent and texture/color from the other.

    Args:
        parent_a: W vector (w_dim,) for coarse structure.
        parent_b: W vector (w_dim,) for fine details.
        num_ws: Number of style layers (16 for 512x512 StyleGAN2).
        crossover_layer: Layer index where mixing switches parents.
            0-3: coarse (overall shape), 4-7: medium (patterns),
            8-15: fine (texture, color).

    Returns:
        Per-layer style tensor (num_ws, w_dim).
    """
    ws = parent_a.unsqueeze(0).expand(num_ws, -1).clone()
    ws[crossover_layer:] = parent_b
    return ws


def blend_class_labels(
    c_a: list[float] | None,
    c_b: list[float] | None,
    ratio: float = 0.5,
) -> list[float] | None:
    """Blend two class label vectors for breeding offspring.

    Args:
        c_a: Parent A's class label (30 floats), or None.
        c_b: Parent B's class label (30 floats), or None.
        ratio: Blend ratio. 0.0 = pure A, 1.0 = pure B.

    Returns:
        Blended class label normalized to sum to 1, or None if both are None.
    """
    if c_a is None and c_b is None:
        return None
    if c_a is None:
        return list(c_b)
    if c_b is None:
        return list(c_a)

    blended = [(1 - ratio) * a + ratio * b for a, b in zip(c_a, c_b)]
    total = sum(blended)
    if total > 0:
        blended = [v / total for v in blended]
    return blended


# Registry of breeding methods for the API
BREEDING_METHODS = {
    "average": breed_average,
    "crossover": breed_crossover,
    "block_crossover": breed_block_crossover,
    "guided": breed_guided,
    "style_mix": breed_style_mix,
}
