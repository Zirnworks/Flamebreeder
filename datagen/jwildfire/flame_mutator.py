"""Mutate JWildfire flame parameters for dataset generation.

Operations:
- Randomize color palette from a diverse library
- Adjust camera/zoom for vignette framing
- Perturb xform parameters slightly for variation
"""

import colorsys
import math
import random
from copy import deepcopy


# ---------------------------------------------------------------------------
# Palette generation
# ---------------------------------------------------------------------------

def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def _lerp_color(
    c1: tuple[int, int, int], c2: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _gradient_palette(rng: random.Random, anchors: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Interpolate between anchor colors to make a 256-entry palette."""
    n = len(anchors)
    if n < 2:
        return [anchors[0]] * 256
    palette = []
    for i in range(256):
        t = i / 255.0 * (n - 1)
        idx = min(int(t), n - 2)
        frac = t - idx
        palette.append(_lerp_color(anchors[idx], anchors[idx + 1], frac))
    return palette


def _banded_palette(
    rng: random.Random,
    anchors: list[tuple[int, int, int]],
    transition_width: float = 0.08,
) -> list[tuple[int, int, int]]:
    """Create a stripy/banded palette with sharp transitions between colors.

    Instead of smooth gradients, each color occupies a flat band with narrow
    transition zones between them. This produces the stripy look where
    warm and cool colors sit next to each other.

    Args:
        rng: Random number generator.
        anchors: List of colors for the bands.
        transition_width: Fraction of each band that is a blend zone (0.0 = razor sharp).
    """
    n = len(anchors)
    if n < 2:
        return [anchors[0]] * 256

    palette = []
    band_size = 256.0 / n
    half_trans = transition_width * band_size / 2.0

    for i in range(256):
        band_idx = min(int(i / band_size), n - 1)
        band_center = (band_idx + 0.5) * band_size

        # Distance from center of this band, normalized
        dist_from_edge = min(
            i - band_idx * band_size,
            (band_idx + 1) * band_size - i,
        )

        if dist_from_edge > half_trans or half_trans < 1:
            # Inside the flat region
            palette.append(anchors[band_idx])
        else:
            # In the transition zone — blend with neighbor
            t = 0.5 + 0.5 * (half_trans - dist_from_edge) / max(half_trans, 1)
            if i < band_center and band_idx > 0:
                palette.append(_lerp_color(anchors[band_idx - 1], anchors[band_idx], 1 - t))
            elif i >= band_center and band_idx < n - 1:
                palette.append(_lerp_color(anchors[band_idx], anchors[band_idx + 1], t))
            else:
                palette.append(anchors[band_idx])

    return palette


def palette_warm_fire(rng: random.Random) -> list[tuple[int, int, int]]:
    """Deep reds, oranges, yellows, white-hot core."""
    base_hue = rng.uniform(0.0, 0.08)  # red-orange range
    anchors = [
        _hsl_to_rgb(base_hue, 1.0, 0.05),
        _hsl_to_rgb(base_hue + 0.02, 1.0, 0.25),
        _hsl_to_rgb(base_hue + 0.05, 1.0, 0.45),
        _hsl_to_rgb(base_hue + 0.10, 0.9, 0.60),
        _hsl_to_rgb(base_hue + 0.12, 0.7, 0.80),
        _hsl_to_rgb(base_hue + 0.13, 0.3, 0.95),
    ]
    return _gradient_palette(rng, anchors)


def palette_cool_ocean(rng: random.Random) -> list[tuple[int, int, int]]:
    """Deep blues, teals, cyans, white foam."""
    base_hue = rng.uniform(0.5, 0.6)
    anchors = [
        _hsl_to_rgb(base_hue - 0.05, 1.0, 0.05),
        _hsl_to_rgb(base_hue, 1.0, 0.20),
        _hsl_to_rgb(base_hue + 0.05, 0.9, 0.40),
        _hsl_to_rgb(base_hue + 0.10, 0.8, 0.55),
        _hsl_to_rgb(base_hue + 0.12, 0.6, 0.75),
        _hsl_to_rgb(base_hue + 0.13, 0.3, 0.92),
    ]
    return _gradient_palette(rng, anchors)


def palette_aurora(rng: random.Random) -> list[tuple[int, int, int]]:
    """Greens, teals, purples — aurora borealis."""
    anchors = [
        _hsl_to_rgb(0.30, 1.0, 0.10),
        _hsl_to_rgb(0.35, 0.9, 0.30),
        _hsl_to_rgb(0.45, 0.8, 0.45),
        _hsl_to_rgb(0.55, 0.9, 0.40),
        _hsl_to_rgb(0.70, 1.0, 0.35),
        _hsl_to_rgb(0.80, 0.8, 0.50),
    ]
    # Shuffle slightly for variation
    for i in range(len(anchors)):
        h, s, l = colorsys.rgb_to_hls(anchors[i][0] / 255, anchors[i][1] / 255, anchors[i][2] / 255)
        h += rng.uniform(-0.03, 0.03)
        anchors[i] = _hsl_to_rgb(h % 1.0, s, l)
    return _gradient_palette(rng, anchors)


def palette_neon(rng: random.Random) -> list[tuple[int, int, int]]:
    """High-saturation neon colors with dark gaps."""
    n_colors = rng.randint(3, 6)
    anchors = []
    for i in range(n_colors):
        h = rng.random()
        s = rng.uniform(0.85, 1.0)
        l = rng.uniform(0.45, 0.65)
        anchors.append(_hsl_to_rgb(h, s, l))
        # Insert dark gap
        if i < n_colors - 1 and rng.random() < 0.5:
            anchors.append(_hsl_to_rgb(h, 0.5, 0.05))
    return _gradient_palette(rng, anchors)


def palette_monochrome(rng: random.Random) -> list[tuple[int, int, int]]:
    """Single hue, varying brightness and saturation."""
    hue = rng.random()
    anchors = [
        _hsl_to_rgb(hue, 1.0, 0.03),
        _hsl_to_rgb(hue, 0.9, 0.15),
        _hsl_to_rgb(hue, 0.8, 0.30),
        _hsl_to_rgb(hue, 0.7, 0.50),
        _hsl_to_rgb(hue, 0.5, 0.70),
        _hsl_to_rgb(hue, 0.3, 0.90),
    ]
    return _gradient_palette(rng, anchors)


def palette_sunset(rng: random.Random) -> list[tuple[int, int, int]]:
    """Warm sunset: deep purple through red, orange, gold."""
    anchors = [
        _hsl_to_rgb(0.75 + rng.uniform(-0.03, 0.03), 0.8, 0.15),
        _hsl_to_rgb(0.85 + rng.uniform(-0.03, 0.03), 0.9, 0.30),
        _hsl_to_rgb(0.95 + rng.uniform(-0.03, 0.03), 1.0, 0.40),
        _hsl_to_rgb(0.05 + rng.uniform(-0.02, 0.02), 1.0, 0.50),
        _hsl_to_rgb(0.10 + rng.uniform(-0.02, 0.02), 0.9, 0.60),
        _hsl_to_rgb(0.13 + rng.uniform(-0.02, 0.02), 0.8, 0.75),
    ]
    return _gradient_palette(rng, anchors)


def palette_electric(rng: random.Random) -> list[tuple[int, int, int]]:
    """Electric blue/purple with bright white/cyan highlights."""
    base = rng.uniform(0.60, 0.75)
    anchors = [
        _hsl_to_rgb(base, 1.0, 0.08),
        _hsl_to_rgb(base + 0.05, 1.0, 0.25),
        _hsl_to_rgb(base + 0.10, 0.9, 0.50),
        _hsl_to_rgb(base - 0.10, 1.0, 0.55),
        _hsl_to_rgb(base + 0.15, 0.6, 0.80),
        (230, 240, 255),  # near-white blue
    ]
    return _gradient_palette(rng, anchors)


def palette_earth(rng: random.Random) -> list[tuple[int, int, int]]:
    """Earthy browns, greens, amber, clay."""
    anchors = [
        _hsl_to_rgb(0.08 + rng.uniform(-0.02, 0.02), 0.7, 0.10),
        _hsl_to_rgb(0.06 + rng.uniform(-0.02, 0.02), 0.8, 0.25),
        _hsl_to_rgb(0.10 + rng.uniform(-0.02, 0.02), 0.6, 0.40),
        _hsl_to_rgb(0.25 + rng.uniform(-0.05, 0.05), 0.5, 0.35),
        _hsl_to_rgb(0.12 + rng.uniform(-0.02, 0.02), 0.7, 0.55),
        _hsl_to_rgb(0.08 + rng.uniform(-0.02, 0.02), 0.4, 0.75),
    ]
    return _gradient_palette(rng, anchors)


def palette_candy(rng: random.Random) -> list[tuple[int, int, int]]:
    """Bright pinks, magentas, cyan, lavender."""
    anchors = [
        _hsl_to_rgb(0.90, 1.0, 0.40),
        _hsl_to_rgb(0.95, 0.9, 0.55),
        _hsl_to_rgb(0.50, 1.0, 0.50),
        _hsl_to_rgb(0.85, 0.8, 0.60),
        _hsl_to_rgb(0.75, 0.7, 0.70),
        _hsl_to_rgb(0.55, 0.6, 0.80),
    ]
    for i in range(len(anchors)):
        h, s, l = colorsys.rgb_to_hls(anchors[i][0] / 255, anchors[i][1] / 255, anchors[i][2] / 255)
        h += rng.uniform(-0.05, 0.05)
        anchors[i] = _hsl_to_rgb(h % 1.0, s, l)
    return _gradient_palette(rng, anchors)


def palette_random_gradient(rng: random.Random) -> list[tuple[int, int, int]]:
    """Fully random gradient with 3-7 anchor colors."""
    n = rng.randint(3, 7)
    anchors = []
    for _ in range(n):
        h = rng.random()
        s = rng.uniform(0.5, 1.0)
        l = rng.uniform(0.15, 0.85)
        anchors.append(_hsl_to_rgb(h, s, l))
    return _gradient_palette(rng, anchors)


# ---------------------------------------------------------------------------
# Stripy / banded / contrasting palettes
# ---------------------------------------------------------------------------

def palette_complementary_stripes(rng: random.Random) -> list[tuple[int, int, int]]:
    """Sharp alternating bands of a color and its complement.

    E.g. orange-blue, red-teal, yellow-purple stripes.
    """
    base_hue = rng.random()
    comp_hue = (base_hue + 0.5) % 1.0
    n_bands = rng.randint(5, 10)

    anchors = []
    for i in range(n_bands):
        h = base_hue if i % 2 == 0 else comp_hue
        # Vary lightness/saturation per band for richness
        s = rng.uniform(0.7, 1.0)
        l = rng.uniform(0.30, 0.65)
        anchors.append(_hsl_to_rgb(h + rng.uniform(-0.03, 0.03), s, l))

    trans = rng.uniform(0.03, 0.12)
    return _banded_palette(rng, anchors, transition_width=trans)


def palette_triadic_stripes(rng: random.Random) -> list[tuple[int, int, int]]:
    """Bands of three equally-spaced hues (triadic harmony).

    E.g. red-green-blue, orange-teal-violet stripes.
    """
    base_hue = rng.random()
    hues = [base_hue, (base_hue + 1 / 3) % 1.0, (base_hue + 2 / 3) % 1.0]
    n_bands = rng.randint(6, 12)

    anchors = []
    for i in range(n_bands):
        h = hues[i % 3] + rng.uniform(-0.04, 0.04)
        s = rng.uniform(0.75, 1.0)
        l = rng.uniform(0.30, 0.60)
        anchors.append(_hsl_to_rgb(h % 1.0, s, l))

    trans = rng.uniform(0.03, 0.10)
    return _banded_palette(rng, anchors, transition_width=trans)


def palette_warm_cool_clash(rng: random.Random) -> list[tuple[int, int, int]]:
    """Alternating warm and cool color bands.

    Specifically mixes yellows/oranges/reds with blues/teals/cyans.
    """
    warm_hues = [rng.uniform(0.0, 0.12) for _ in range(4)]   # reds, oranges, yellows
    cool_hues = [rng.uniform(0.50, 0.70) for _ in range(4)]   # blues, teals, cyans

    n_bands = rng.randint(5, 9)
    anchors = []
    for i in range(n_bands):
        if i % 2 == 0:
            h = rng.choice(warm_hues)
        else:
            h = rng.choice(cool_hues)
        s = rng.uniform(0.8, 1.0)
        l = rng.uniform(0.35, 0.60)
        anchors.append(_hsl_to_rgb(h, s, l))

    trans = rng.uniform(0.04, 0.15)
    return _banded_palette(rng, anchors, transition_width=trans)


def palette_split_complementary(rng: random.Random) -> list[tuple[int, int, int]]:
    """Base hue + two colors adjacent to its complement.

    Creates high contrast with less tension than direct complement.
    """
    base_hue = rng.random()
    split_a = (base_hue + 0.42) % 1.0  # ~150 degrees away
    split_b = (base_hue + 0.58) % 1.0  # ~210 degrees away

    hues = [base_hue, split_a, split_b]
    n_bands = rng.randint(6, 10)

    anchors = []
    for i in range(n_bands):
        h = hues[i % 3] + rng.uniform(-0.03, 0.03)
        s = rng.uniform(0.75, 1.0)
        l = rng.uniform(0.30, 0.60)
        anchors.append(_hsl_to_rgb(h % 1.0, s, l))

    # Mix of banded and gradient — use gradient but with many anchors close together
    if rng.random() < 0.5:
        return _banded_palette(rng, anchors, transition_width=rng.uniform(0.05, 0.15))
    else:
        return _gradient_palette(rng, anchors)


def palette_rainbow_stripes(rng: random.Random) -> list[tuple[int, int, int]]:
    """Full-spectrum rainbow bands with sharp transitions."""
    n_bands = rng.randint(6, 14)
    start_hue = rng.random()

    anchors = []
    for i in range(n_bands):
        h = (start_hue + i / n_bands) % 1.0
        s = rng.uniform(0.8, 1.0)
        l = rng.uniform(0.35, 0.55)
        anchors.append(_hsl_to_rgb(h, s, l))

    trans = rng.uniform(0.02, 0.08)
    return _banded_palette(rng, anchors, transition_width=trans)


def palette_jewel_tones(rng: random.Random) -> list[tuple[int, int, int]]:
    """Deep saturated jewel colors — emerald, ruby, sapphire, amethyst, topaz."""
    jewel_hues = {
        "emerald": (0.38, 0.9, 0.30),
        "ruby": (0.97, 0.9, 0.35),
        "sapphire": (0.62, 0.85, 0.30),
        "amethyst": (0.78, 0.7, 0.35),
        "topaz": (0.10, 0.9, 0.45),
        "garnet": (0.0, 0.85, 0.25),
        "citrine": (0.13, 0.95, 0.50),
        "jade": (0.42, 0.6, 0.35),
    }
    names = list(jewel_hues.keys())
    rng.shuffle(names)
    n_bands = rng.randint(4, 7)
    selected = names[:n_bands]

    anchors = []
    for name in selected:
        h, s, l = jewel_hues[name]
        h += rng.uniform(-0.02, 0.02)
        anchors.append(_hsl_to_rgb(h % 1.0, s, l))

    if rng.random() < 0.6:
        return _banded_palette(rng, anchors, transition_width=rng.uniform(0.05, 0.20))
    else:
        return _gradient_palette(rng, anchors)


def palette_random_stripes(rng: random.Random) -> list[tuple[int, int, int]]:
    """Fully random colors in sharp bands — maximum variety."""
    n_bands = rng.randint(4, 12)
    anchors = []
    for _ in range(n_bands):
        h = rng.random()
        s = rng.uniform(0.6, 1.0)
        l = rng.uniform(0.20, 0.70)
        anchors.append(_hsl_to_rgb(h, s, l))

    trans = rng.uniform(0.0, 0.15)
    return _banded_palette(rng, anchors, transition_width=trans)


PALETTE_GENERATORS = [
    palette_warm_fire,
    palette_cool_ocean,
    palette_aurora,
    palette_neon,
    palette_monochrome,
    palette_sunset,
    palette_electric,
    palette_earth,
    palette_candy,
    palette_random_gradient,
    # Stripy / contrasting (new)
    palette_complementary_stripes,
    palette_triadic_stripes,
    palette_warm_cool_clash,
    palette_split_complementary,
    palette_rainbow_stripes,
    palette_jewel_tones,
    palette_random_stripes,
]


def random_palette(rng: random.Random | None = None) -> list[tuple[int, int, int]]:
    """Generate a random palette using one of the palette strategies."""
    if rng is None:
        rng = random.Random()
    generator = rng.choice(PALETTE_GENERATORS)
    palette = generator(rng)
    # Optionally rotate the palette
    if rng.random() < 0.3:
        shift = rng.randint(0, 255)
        palette = palette[shift:] + palette[:shift]
    return palette


# ---------------------------------------------------------------------------
# Camera / framing adjustments for vignettes
# ---------------------------------------------------------------------------

def adjust_for_vignette(flame: dict, rng: random.Random | None = None) -> dict:
    """Adjust camera parameters to favor vignette output.

    - Centers the view
    - Adjusts zoom to keep the fractal contained
    - Sets black background
    """
    if rng is None:
        rng = random.Random()

    flame = deepcopy(flame)
    attrs = flame["attrs"]

    # Center the camera
    attrs["center"] = "0.0 0.0"

    # Adjust zoom: scale down to keep fractal contained
    # Original scale varies wildly (82 to 1381 in the presets we looked at).
    # cam_zoom is a multiplier on top of scale.
    # We want to zoom out slightly from whatever the preset had.
    if "cam_zoom" in attrs:
        original_zoom = float(attrs["cam_zoom"])
        # Reduce zoom by 20-40% to pull the fractal away from edges
        attrs["cam_zoom"] = str(original_zoom * rng.uniform(0.6, 0.8))

    # Ensure black background
    for bg_key in ["background_ul", "background_ur", "background_ll",
                   "background_lr", "background_cc", "background"]:
        if bg_key in attrs:
            attrs[bg_key] = "0.0 0.0 0.0"
    attrs["background_type"] = "GRADIENT_2X2_C"

    # Set output resolution to 512x512
    attrs["size"] = "512 512"

    # Ensure reasonable rendering params
    attrs["quality"] = "200.0"

    return flame


# ---------------------------------------------------------------------------
# Palette replacement
# ---------------------------------------------------------------------------

def replace_palette(flame: dict, new_palette: list[tuple[int, int, int]]) -> dict:
    """Replace the flame's palette with a new one."""
    flame = deepcopy(flame)
    flame["palette"] = new_palette
    return flame


# ---------------------------------------------------------------------------
# Parameter perturbation
# ---------------------------------------------------------------------------

def perturb_xform_params(flame: dict, rng: random.Random | None = None,
                         strength: float = 0.1) -> dict:
    """Slightly perturb xform affine coefficients for variation.

    This creates subtle variants of the same flame shape.
    """
    if rng is None:
        rng = random.Random()

    flame = deepcopy(flame)

    for xform in flame["xforms"]:
        if "coefs" in xform and isinstance(xform["coefs"], list):
            coefs = xform["coefs"]
            for i in range(len(coefs)):
                coefs[i] += rng.gauss(0, strength * abs(coefs[i]) + 0.01)

        # Slightly perturb variation amounts
        for key, val in list(xform.items()):
            if key in ("weight", "color", "symmetry", "coefs", "post", "chaos",
                       "opacity", "mirror_pre_post_translations",
                       "material", "material_speed",
                       "mod_gamma", "mod_gamma_speed",
                       "mod_contrast", "mod_contrast_speed",
                       "mod_saturation", "mod_saturation_speed",
                       "mod_hue", "mod_hue_speed"):
                continue
            # Skip string/non-numeric params
            try:
                fval = float(val)
            except (ValueError, TypeError):
                continue
            # Skip binary/integer flags
            if key.endswith("_fx_priority") or key.endswith("_zero"):
                continue
            # Perturb the value
            if abs(fval) > 0.001:
                xform[key] = str(fval + rng.gauss(0, strength * abs(fval)))

    return flame


# ---------------------------------------------------------------------------
# Full mutation pipeline
# ---------------------------------------------------------------------------

def mutate_flame(
    flame: dict,
    rng: random.Random | None = None,
    replace_colors: bool = True,
    adjust_camera: bool = True,
    perturb: bool = True,
    perturb_strength: float = 0.05,
    perturb_strength_range: tuple[float, float] | None = None,
) -> dict:
    """Apply a full mutation pipeline to a flame.

    Args:
        flame: parsed flame dict
        rng: random number generator
        replace_colors: whether to randomize the palette
        adjust_camera: whether to adjust for vignette framing
        perturb: whether to perturb xform parameters
        perturb_strength: how much to perturb (0.0-1.0), used when range is None
        perturb_strength_range: (min, max) — sample strength uniformly per flame
    """
    if rng is None:
        rng = random.Random()

    result = deepcopy(flame)

    if replace_colors:
        result = replace_palette(result, random_palette(rng))

    if adjust_camera:
        result = adjust_for_vignette(result, rng)

    if perturb:
        if perturb_strength_range is not None:
            strength = rng.uniform(*perturb_strength_range)
        else:
            strength = perturb_strength
        result = perturb_xform_params(result, rng, strength)

    return result
