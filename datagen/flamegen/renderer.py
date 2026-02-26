"""Histogram rendering with log-density display and tone mapping.

Converts raw histogram accumulations from the chaos game into
a displayable RGB image using the fractal flame log-density method.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image


def render_flame(
    hist_count: np.ndarray,
    hist_color: np.ndarray,
    gamma: float = 4.0,
    brightness: float = 4.0,
    vibrancy: float = 1.0,
    blur_sigma: float = 0.5,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Image.Image:
    """Render histogram buffers to a PIL Image using log-density display.

    Args:
        hist_count: (H, W) hit count histogram.
        hist_color: (H, W, 3) accumulated RGB color histogram.
        gamma: Gamma correction exponent (higher = brighter midtones).
        brightness: Overall brightness multiplier.
        vibrancy: Blend between log-density coloring (1.0) and flat coloring (0.0).
        blur_sigma: Gaussian blur sigma for anti-aliasing (0 = no blur).
        background: Background RGB color, each channel in [0, 1].

    Returns:
        PIL Image in RGB mode, uint8.
    """
    h, w = hist_count.shape

    # Avoid log(0)
    max_count = hist_count.max()
    if max_count == 0:
        # Empty histogram — return background
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            img[:, :, c] = int(background[c] * 255)
        return Image.fromarray(img)

    # Log-density alpha channel
    # alpha = log(1 + count) / log(1 + max_count)
    log_count = np.log1p(hist_count)
    log_max = np.log1p(max_count)
    alpha = log_count / log_max

    # Apply gamma correction to alpha
    inv_gamma = 1.0 / gamma
    alpha_gamma = np.power(alpha, inv_gamma)

    # Compute average color per pixel (where count > 0)
    avg_color = np.zeros_like(hist_color)
    mask = hist_count > 0
    for c in range(3):
        avg_color[:, :, c] = np.where(
            mask,
            hist_color[:, :, c] / np.maximum(hist_count, 1.0),
            0.0,
        )

    # Blend: vibrancy controls mix between log-density and flat coloring
    # Full vibrancy (1.0): color * alpha_gamma^(1/vibrancy_power)
    # The standard flame algorithm: pixel = color_avg * alpha^(1/gamma) * brightness
    output = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        # Log-density path: modulate color by alpha
        log_color = avg_color[:, :, c] * alpha_gamma
        # Flat path: just the average color
        flat_color = avg_color[:, :, c] * alpha
        # Blend
        output[:, :, c] = vibrancy * log_color + (1.0 - vibrancy) * flat_color

    # Apply brightness
    output *= brightness

    # Anti-aliasing blur
    if blur_sigma > 0:
        for c in range(3):
            output[:, :, c] = gaussian_filter(output[:, :, c], sigma=blur_sigma)

    # Composite over background using alpha as opacity
    alpha_3d = alpha_gamma[:, :, np.newaxis]
    bg = np.array(background).reshape(1, 1, 3)
    output = output * alpha_3d + bg * (1.0 - alpha_3d)

    # Clamp and convert to uint8
    output = np.clip(output * 255.0, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(output, mode="RGB")
