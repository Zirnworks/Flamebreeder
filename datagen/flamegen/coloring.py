"""Color palette generation for fractal flames.

Provides methods to generate random color palettes that map
the [0, 1] color index to RGB values.
"""

import numpy as np


def random_gradient_palette(rng: np.random.Generator, num_anchors: int = 5) -> np.ndarray:
    """Generate a random gradient palette by interpolating between anchor colors.

    Args:
        rng: NumPy random generator.
        num_anchors: Number of anchor colors to interpolate between.

    Returns:
        (256, 3) float64 array with RGB values in [0, 1].
    """
    anchors = rng.random((num_anchors, 3))
    positions = np.sort(rng.random(num_anchors))
    positions[0] = 0.0
    positions[-1] = 1.0

    palette = np.zeros((256, 3), dtype=np.float64)
    for i in range(256):
        t = i / 255.0
        # Find the two anchors this t falls between
        for j in range(num_anchors - 1):
            if positions[j] <= t <= positions[j + 1]:
                local_t = (t - positions[j]) / (positions[j + 1] - positions[j] + 1e-10)
                palette[i] = (1 - local_t) * anchors[j] + local_t * anchors[j + 1]
                break

    return palette


def cubehelix_palette(
    rng: np.random.Generator,
    start: float | None = None,
    rotations: float | None = None,
    hue: float | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Generate a cubehelix-style palette (Dave Green's scheme).

    Good for fractal flames because it has monotonically increasing luminance.

    Args:
        rng: NumPy random generator.
        start: Start color (0-3). Random if None.
        rotations: Number of R->G->B rotations. Random if None.
        hue: Saturation. Random if None.
        gamma: Gamma correction.

    Returns:
        (256, 3) float64 array with RGB values in [0, 1].
    """
    if start is None:
        start = rng.uniform(0, 3)
    if rotations is None:
        rotations = rng.uniform(-2, 2)
    if hue is None:
        hue = rng.uniform(0.5, 2.0)

    palette = np.zeros((256, 3), dtype=np.float64)
    for i in range(256):
        fract = i / 255.0
        angle = 2.0 * np.pi * (start / 3.0 + 1.0 + rotations * fract)
        fract_g = fract**gamma
        amp = hue * fract_g * (1.0 - fract_g) / 2.0
        palette[i, 0] = fract_g + amp * (-0.14861 * np.cos(angle) + 1.78277 * np.sin(angle))
        palette[i, 1] = fract_g + amp * (-0.29227 * np.cos(angle) - 0.90649 * np.sin(angle))
        palette[i, 2] = fract_g + amp * (1.97294 * np.cos(angle))

    return np.clip(palette, 0, 1)


def monochrome_palette(rng: np.random.Generator) -> np.ndarray:
    """Generate a monochrome palette that varies in brightness.

    Args:
        rng: NumPy random generator.

    Returns:
        (256, 3) float64 array with RGB values in [0, 1].
    """
    base_color = rng.random(3)
    # Normalize to reasonable brightness
    base_color = base_color / (base_color.max() + 0.1)

    palette = np.zeros((256, 3), dtype=np.float64)
    for i in range(256):
        t = i / 255.0
        palette[i] = base_color * t

    return palette


def fire_palette() -> np.ndarray:
    """Classic fire palette: black -> red -> orange -> yellow -> white."""
    anchors = np.array([
        [0.0, 0.0, 0.0],    # black
        [0.5, 0.0, 0.0],    # dark red
        [1.0, 0.2, 0.0],    # red-orange
        [1.0, 0.6, 0.0],    # orange
        [1.0, 1.0, 0.3],    # yellow
        [1.0, 1.0, 1.0],    # white
    ], dtype=np.float64)
    positions = np.array([0.0, 0.2, 0.4, 0.6, 0.85, 1.0])

    palette = np.zeros((256, 3), dtype=np.float64)
    for i in range(256):
        t = i / 255.0
        for j in range(len(positions) - 1):
            if positions[j] <= t <= positions[j + 1]:
                local_t = (t - positions[j]) / (positions[j + 1] - positions[j])
                palette[i] = (1 - local_t) * anchors[j] + local_t * anchors[j + 1]
                break

    return palette


def ocean_palette() -> np.ndarray:
    """Ocean palette: deep blue -> teal -> cyan -> white."""
    anchors = np.array([
        [0.0, 0.0, 0.15],
        [0.0, 0.1, 0.4],
        [0.0, 0.3, 0.6],
        [0.1, 0.6, 0.7],
        [0.5, 0.9, 0.9],
        [1.0, 1.0, 1.0],
    ], dtype=np.float64)
    positions = np.array([0.0, 0.2, 0.4, 0.6, 0.85, 1.0])

    palette = np.zeros((256, 3), dtype=np.float64)
    for i in range(256):
        t = i / 255.0
        for j in range(len(positions) - 1):
            if positions[j] <= t <= positions[j + 1]:
                local_t = (t - positions[j]) / (positions[j + 1] - positions[j])
                palette[i] = (1 - local_t) * anchors[j] + local_t * anchors[j + 1]
                break

    return palette


def random_palette(rng: np.random.Generator) -> np.ndarray:
    """Select a random palette generation method.

    Returns:
        (256, 3) float64 array with RGB values in [0, 1].
    """
    choice = rng.integers(0, 5)
    if choice == 0:
        return random_gradient_palette(rng, num_anchors=rng.integers(3, 8))
    elif choice == 1:
        return cubehelix_palette(rng)
    elif choice == 2:
        return monochrome_palette(rng)
    elif choice == 3:
        return fire_palette()
    else:
        return ocean_palette()
