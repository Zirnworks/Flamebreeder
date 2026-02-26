"""Random flame parameter generation.

Creates randomized IFS parameters suitable for the chaos game engine.
Each generated parameter set defines a unique fractal flame.
"""

import numpy as np

from .ifs import MAX_VARIATIONS_PER_XFORM, MAX_XFORMS
from .variations import NUM_VARIATIONS
from .coloring import random_palette

# Variation boundedness categories for vignette-friendly generation.
# Strongly bounded variations produce compact, centered shapes.
# Weakly bounded ones tend to fill the frame or produce streaks to edges.
_STRONGLY_BOUNDED = [2, 8, 9, 13, 14, 15, 16]  # spherical, disc, spiral, julia, fisheye, eyefish, bubble
_MODERATELY_BOUNDED = [3, 4, 5, 7, 10, 11, 12]  # swirl, horseshoe, polar, heart, hyperbolic, diamond, ex
_WEAKLY_BOUNDED = [0, 1, 6, 17]  # linear, sinusoidal, handkerchief, cylinder


def _pick_weighted_variations(rng: np.random.Generator, count: int) -> np.ndarray:
    """Pick variations with probability weighted toward bounded ones.

    Strongly bounded (50%), moderately bounded (35%), weakly bounded (15%).
    This biases generation toward compact, centered shapes suitable for vignettes.
    """
    pool = []
    for _ in range(count):
        r = rng.random()
        if r < 0.50:
            category = _STRONGLY_BOUNDED
        elif r < 0.85:
            category = _MODERATELY_BOUNDED
        else:
            category = _WEAKLY_BOUNDED
        choice = rng.choice(category)
        # Avoid duplicates
        while choice in pool:
            r = rng.random()
            if r < 0.50:
                category = _STRONGLY_BOUNDED
            elif r < 0.85:
                category = _MODERATELY_BOUNDED
            else:
                category = _WEAKLY_BOUNDED
            choice = rng.choice(category)
        pool.append(choice)
    return np.array(pool, dtype=np.int64)


def random_flame_params(
    rng: np.random.Generator,
    min_xforms: int = 2,
    max_xforms: int = 6,
    final_xform_prob: float = 0.3,
    symmetry_prob: float = 0.2,
) -> dict:
    """Generate a complete set of random flame parameters.

    Args:
        rng: NumPy random generator.
        min_xforms: Minimum number of transforms.
        max_xforms: Maximum number of transforms.
        final_xform_prob: Probability of adding a final transform.
        symmetry_prob: Probability of adding rotational symmetry.

    Returns:
        Dictionary of parameters ready for chaos_game().
    """
    num_xforms = rng.integers(min_xforms, max_xforms + 1)

    # Affine pre-transform coefficients [a, b, c, d, e, f]
    affine_coeffs = np.zeros((MAX_XFORMS, 6), dtype=np.float64)
    for i in range(num_xforms):
        affine_coeffs[i] = _random_affine(rng)

    # Variation selection and weights per transform
    variation_ids = np.zeros((MAX_XFORMS, MAX_VARIATIONS_PER_XFORM), dtype=np.int64)
    variation_weights = np.zeros((MAX_XFORMS, MAX_VARIATIONS_PER_XFORM), dtype=np.float64)
    num_variations = np.zeros(MAX_XFORMS, dtype=np.int64)

    for i in range(num_xforms):
        n_vars = rng.integers(1, 4)  # 1-3 variations per transform
        num_variations[i] = n_vars
        chosen = _pick_weighted_variations(rng, n_vars)
        weights = rng.random(n_vars)
        weights /= weights.sum()
        for j in range(n_vars):
            variation_ids[i, j] = chosen[j]
            variation_weights[i, j] = weights[j]

    # Transform probability weights
    xform_weights = np.zeros(MAX_XFORMS, dtype=np.float64)
    raw_weights = rng.random(num_xforms)
    raw_weights /= raw_weights.sum()
    xform_weights[:num_xforms] = raw_weights

    # Color values per transform
    xform_colors = np.zeros(MAX_XFORMS, dtype=np.float64)
    xform_colors[:num_xforms] = np.linspace(0.0, 1.0, num_xforms)
    rng.shuffle(xform_colors[:num_xforms])

    # Post-transforms (optional, ~30% chance per transform)
    has_post = np.zeros(MAX_XFORMS, dtype=np.bool_)
    post_coeffs = np.zeros((MAX_XFORMS, 6), dtype=np.float64)
    for i in range(num_xforms):
        if rng.random() < 0.3:
            has_post[i] = True
            post_coeffs[i] = _random_affine(rng, scale=0.8)

    # Final transform
    has_final = rng.random() < final_xform_prob
    final_affine = np.zeros(6, dtype=np.float64)
    final_var_ids = np.zeros(MAX_VARIATIONS_PER_XFORM, dtype=np.int64)
    final_var_weights = np.zeros(MAX_VARIATIONS_PER_XFORM, dtype=np.float64)
    final_num_vars = 0

    if has_final:
        final_affine[:] = _random_affine(rng, scale=0.6)
        final_num_vars = rng.integers(1, 3)
        chosen = rng.choice(NUM_VARIATIONS, size=final_num_vars, replace=False)
        weights = rng.random(final_num_vars)
        weights /= weights.sum()
        for j in range(final_num_vars):
            final_var_ids[j] = chosen[j]
            final_var_weights[j] = weights[j]

    # Add rotational symmetry transforms
    if rng.random() < symmetry_prob and num_xforms < MAX_XFORMS - 1:
        sym_order = rng.integers(2, 7)  # 2-6 fold symmetry
        num_xforms = _add_symmetry(
            sym_order, num_xforms,
            affine_coeffs, variation_ids, variation_weights,
            num_variations, xform_weights, xform_colors,
            has_post, post_coeffs,
        )

    # Camera parameters — tuned for vignette output (centered object, black edges)
    cam_x_center = rng.uniform(-0.03, 0.03)
    cam_y_center = rng.uniform(-0.03, 0.03)
    cam_scale = rng.uniform(0.25, 0.55)
    cam_rotation = rng.uniform(0, 2 * np.pi)

    # Color palette
    palette = random_palette(rng)

    # Rendering parameters
    gamma = rng.uniform(2.5, 5.0)
    brightness = rng.uniform(2.0, 6.0)
    vibrancy = rng.uniform(0.5, 1.0)

    return {
        "num_xforms": num_xforms,
        "affine_coeffs": affine_coeffs,
        "variation_ids": variation_ids,
        "variation_weights": variation_weights,
        "num_variations": num_variations,
        "xform_weights": xform_weights,
        "xform_colors": xform_colors,
        "has_post": has_post,
        "post_coeffs": post_coeffs,
        "has_final": has_final,
        "final_affine": final_affine,
        "final_var_ids": final_var_ids,
        "final_var_weights": final_var_weights,
        "final_num_vars": final_num_vars,
        "cam_x_center": cam_x_center,
        "cam_y_center": cam_y_center,
        "cam_scale": cam_scale,
        "cam_rotation": cam_rotation,
        "palette": palette,
        "gamma": gamma,
        "brightness": brightness,
        "vibrancy": vibrancy,
    }


def _random_affine(rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Generate random affine coefficients [a, b, c, d, e, f].

    Uses a strategy that tends to produce contractive transforms
    (necessary for IFS convergence) while allowing interesting distortions.
    """
    coeffs = np.zeros(6, dtype=np.float64)

    # Method: generate a random 2x2 matrix with bounded singular values
    # This ensures the transform is contractive on average
    angle = rng.uniform(0, 2 * np.pi)
    scale_x = rng.uniform(0.1, 0.9) * scale
    scale_y = rng.uniform(0.1, 0.9) * scale
    skew = rng.uniform(-0.5, 0.5)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Rotation * scale with skew
    coeffs[0] = scale_x * cos_a + skew * sin_a  # a
    coeffs[1] = -scale_x * sin_a + skew * cos_a  # b
    coeffs[2] = rng.uniform(-0.5, 0.5) * scale  # c (translation x)
    coeffs[3] = scale_y * sin_a  # d
    coeffs[4] = scale_y * cos_a  # e
    coeffs[5] = rng.uniform(-0.5, 0.5) * scale  # f (translation y)

    return coeffs


def _add_symmetry(
    order: int,
    num_xforms: int,
    affine_coeffs: np.ndarray,
    variation_ids: np.ndarray,
    variation_weights: np.ndarray,
    num_variations: np.ndarray,
    xform_weights: np.ndarray,
    xform_colors: np.ndarray,
    has_post: np.ndarray,
    post_coeffs: np.ndarray,
) -> int:
    """Add rotational symmetry transforms.

    Adds one transform that applies a rotation by 2*pi/order.
    This is the standard way to add symmetry to fractal flames.
    """
    if num_xforms >= MAX_XFORMS:
        return num_xforms

    angle = 2.0 * np.pi / order
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    i = num_xforms
    # Pure rotation affine
    affine_coeffs[i] = [cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0]
    # Linear variation only
    variation_ids[i, 0] = 0  # V_LINEAR
    variation_weights[i, 0] = 1.0
    num_variations[i] = 1
    # High weight so symmetry is applied frequently
    xform_weights[i] = xform_weights[:num_xforms].mean() * 2.0
    xform_colors[i] = 0.5
    has_post[i] = False

    return num_xforms + 1
