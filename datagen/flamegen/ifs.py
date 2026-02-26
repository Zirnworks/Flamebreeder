"""Core IFS (Iterated Function System) chaos game algorithm.

Implements the fractal flame algorithm per Draves & Reckase.
The inner loop is Numba JIT-compiled for performance.
"""

import numpy as np
import numba as nb

from .variations import apply_variation


# Maximum number of variations per transform and transforms per flame
MAX_VARIATIONS_PER_XFORM = 6
MAX_XFORMS = 8


@nb.njit(cache=True)
def affine_transform(x: float, y: float, coeffs: np.ndarray) -> tuple[float, float]:
    """Apply 2D affine transform: (a*x + b*y + c, d*x + e*y + f).

    coeffs: array of [a, b, c, d, e, f]
    """
    ax = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    ay = coeffs[3] * x + coeffs[4] * y + coeffs[5]
    return ax, ay


@nb.njit(cache=True)
def chaos_game(
    num_iterations: int,
    size: int,
    # Transform parameters (padded to MAX_XFORMS)
    num_xforms: int,
    affine_coeffs: np.ndarray,       # (MAX_XFORMS, 6) - affine pre-transform
    variation_ids: np.ndarray,       # (MAX_XFORMS, MAX_VARIATIONS_PER_XFORM) - which variations
    variation_weights: np.ndarray,   # (MAX_XFORMS, MAX_VARIATIONS_PER_XFORM) - blend weights
    num_variations: np.ndarray,      # (MAX_XFORMS,) - how many variations per xform
    xform_weights: np.ndarray,       # (MAX_XFORMS,) - probability weights (normalized)
    xform_colors: np.ndarray,        # (MAX_XFORMS,) - color index [0,1]
    # Post-transform (optional, applied if has_post is True)
    has_post: np.ndarray,            # (MAX_XFORMS,) bool
    post_coeffs: np.ndarray,         # (MAX_XFORMS, 6) - affine post-transform
    # Final transform (optional)
    has_final: bool,
    final_affine: np.ndarray,        # (6,)
    final_var_ids: np.ndarray,       # (MAX_VARIATIONS_PER_XFORM,)
    final_var_weights: np.ndarray,   # (MAX_VARIATIONS_PER_XFORM,)
    final_num_vars: int,
    # Camera transform
    cam_x_center: float,
    cam_y_center: float,
    cam_scale: float,
    cam_rotation: float,
    # Output histograms
    hist_count: np.ndarray,          # (size, size) float64 - hit count
    hist_color: np.ndarray,          # (size, size, 3) float64 - accumulated RGB
    # Color palette
    palette: np.ndarray,             # (256, 3) float64 - RGB palette
    # Random seed
    seed: int,
) -> None:
    """Run the chaos game, accumulating into the histogram buffers.

    This is the hot inner loop — everything here must be Numba-compatible.
    """
    np.random.seed(seed)

    # Build cumulative weight distribution for transform selection
    cum_weights = np.zeros(num_xforms, dtype=np.float64)
    total_w = 0.0
    for i in range(num_xforms):
        total_w += xform_weights[i]
        cum_weights[i] = total_w
    # Normalize
    for i in range(num_xforms):
        cum_weights[i] /= total_w

    # Precompute camera rotation
    cos_rot = np.cos(cam_rotation)
    sin_rot = np.sin(cam_rotation)

    # Initialize point
    x = np.random.uniform(-1.0, 1.0)
    y = np.random.uniform(-1.0, 1.0)
    c = np.random.uniform(0.0, 1.0)

    warmup = 20

    for iteration in range(num_iterations + warmup):
        # Select transform via weighted random choice
        r = np.random.random()
        xi = 0
        for i in range(num_xforms):
            if r <= cum_weights[i]:
                xi = i
                break

        # Apply affine pre-transform
        ax, ay = affine_transform(x, y, affine_coeffs[xi])

        # Apply blended variations
        vx = 0.0
        vy = 0.0
        rng_val = np.random.random()
        for vi in range(num_variations[xi]):
            var_id = variation_ids[xi, vi]
            weight = variation_weights[xi, vi]
            vvx, vvy = apply_variation(var_id, ax, ay, rng_val)
            vx += weight * vvx
            vy += weight * vvy

        x = vx
        y = vy

        # Apply post-transform if present
        if has_post[xi]:
            x, y = affine_transform(x, y, post_coeffs[xi])

        # Update color (blend toward transform's color)
        c = (c + xform_colors[xi]) / 2.0

        # Apply final transform (does not affect iteration state)
        fx, fy = x, y
        if has_final:
            fax, fay = affine_transform(fx, fy, final_affine)
            fvx = 0.0
            fvy = 0.0
            for vi in range(final_num_vars):
                var_id = final_var_ids[vi]
                weight = final_var_weights[vi]
                vvx, vvy = apply_variation(var_id, fax, fay, rng_val)
                fvx += weight * vvx
                fvy += weight * vvy
            fx = fvx
            fy = fvy

        # Skip warmup iterations
        if iteration < warmup:
            continue

        # Camera transform: center, scale, rotate
        cx = fx - cam_x_center
        cy = fy - cam_y_center
        rx = cx * cos_rot - cy * sin_rot
        ry = cx * sin_rot + cy * cos_rot

        # Map to pixel coordinates
        px_f = (rx * cam_scale + 0.5) * size
        py_f = (ry * cam_scale + 0.5) * size
        px = int(px_f)
        py = int(py_f)

        # Bounds check
        if 0 <= px < size and 0 <= py < size:
            hist_count[py, px] += 1.0
            # Look up palette color
            cidx = int(c * 255.0)
            if cidx < 0:
                cidx = 0
            elif cidx > 255:
                cidx = 255
            hist_color[py, px, 0] += palette[cidx, 0]
            hist_color[py, px, 1] += palette[cidx, 1]
            hist_color[py, px, 2] += palette[cidx, 2]
