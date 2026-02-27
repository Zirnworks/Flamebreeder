"""Crossover (interpolation) between pairs of flame fractals.

Takes pairs of .flame files and generates parameter-space interpolations
at various blend ratios, creating "bridge" images between visual families.

Supports multiple crossover strategies:
  geometry — only interpolate affine transforms + palette, keep variation
             structure from flame A. Safest mode.
  matched  — only pair flames with matching variation types, then do full
             interpolation. Moderate safety.
  full     — interpolate everything between any two flames. High failure rate.

Usage:
    python -m datagen.jwildfire.flame_crossover \
        --flames /path/to/source_flames \
        --output /path/to/crossover_flames \
        --mode geometry \
        --pairs 500 --steps 5 --seed 400
"""

import argparse
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from .flame_parser import parse_flame_file
from .flame_mutator import random_palette, replace_palette, adjust_for_vignette
from .flame_writer import write_flame_file


# Flame-level attributes that JWildfire expects as integers
_INTEGER_ATTRS = {
    "bg_transparency", "frame", "frame_count", "post_symmetry_order",
    "temporal_samples", "spatial_oversample",
}

# Flame-level attributes that should not be interpolated (strings/enums)
_SKIP_ATTRS = {
    "name", "smooth_gradient", "version", "filter_kernel",
    "cam_dof_shape", "shading_shading", "post_symmetry_type",
    "mixer_mode", "background_type",
}

# Xform keys that JWildfire expects as integers
_INTEGER_XFORM_KEYS = {
    "nBlur_exactCalc",
}


# Standard xform keys that are NOT variation names
_STANDARD_KEYS = {
    "weight", "color", "symmetry", "coefs", "post", "chaos",
    "opacity", "mirror_pre_post_translations",
    "material", "material_speed",
    "mod_gamma", "mod_gamma_speed",
    "mod_contrast", "mod_contrast_speed",
    "mod_saturation", "mod_saturation_speed",
    "mod_hue", "mod_hue_speed",
}


def _lerp(a: float, b: float, t: float) -> float:
    return a * (1 - t) + b * t


def _lerp_list(a: list, b: list, t: float) -> list:
    """Interpolate two lists element-wise, padding shorter with zeros."""
    max_len = max(len(a), len(b))
    a_padded = list(a) + [0.0] * (max_len - len(a))
    b_padded = list(b) + [0.0] * (max_len - len(b))
    return [_lerp(av, bv, t) for av, bv in zip(a_padded, b_padded)]


def _lerp_palette(
    pal_a: list[tuple[int, int, int]],
    pal_b: list[tuple[int, int, int]],
    t: float,
) -> list[tuple[int, int, int]]:
    """Interpolate two 256-color palettes."""
    result = []
    for (ra, ga, ba), (rb, gb, bb) in zip(pal_a, pal_b):
        result.append((
            int(_lerp(ra, rb, t)),
            int(_lerp(ga, gb, t)),
            int(_lerp(ba, bb, t)),
        ))
    return result


def get_variation_names(xform: dict) -> frozenset[str]:
    """Extract variation type names from an xform dict."""
    names = set()
    for key in xform:
        if key in _STANDARD_KEYS:
            continue
        # Variation params often have underscores (e.g. julian_power)
        # The variation name itself is the key with a non-zero float value
        try:
            val = float(xform[key])
            # Keys ending in _fx_priority or _zero are flags, not variations
            if key.endswith("_fx_priority") or key.endswith("_zero"):
                continue
            # Only count if nonzero — this is the variation weight
            if abs(val) > 1e-10:
                names.add(key)
        except (ValueError, TypeError):
            continue
    return frozenset(names)


def get_flame_signature(flame: dict) -> tuple:
    """Get a structural signature for matching compatible flames.

    Returns a tuple of (xform_count, per-xform variation name sets).
    Two flames with the same signature have identical structure.
    """
    xform_sigs = []
    for xf in flame["xforms"]:
        xform_sigs.append(get_variation_names(xf))
    return (len(flame["xforms"]), tuple(xform_sigs))


def get_flame_variation_set(flame: dict) -> frozenset[str]:
    """Get the union of all variation names across all xforms.

    Used for looser matching — same variations present but possibly
    in different xforms or different counts.
    """
    all_vars = set()
    for xf in flame["xforms"]:
        all_vars |= get_variation_names(xf)
    return frozenset(all_vars)


# ---------------------------------------------------------------------------
# Crossover modes
# ---------------------------------------------------------------------------

def crossover_geometry_only(flame_a: dict, flame_b: dict, t: float) -> dict:
    """Interpolate only affine transforms and palette.

    Keeps flame A's complete variation structure (types + params).
    Only blends the coefs, post, palette, and camera attributes.
    This is the safest mode — the IFS dynamics are preserved from A,
    only the spatial arrangement shifts toward B.
    """
    result = deepcopy(flame_a)

    # Interpolate flame-level numeric attributes (camera, size, etc.)
    for key in result["attrs"]:
        if key in _SKIP_ATTRS:
            continue
        try:
            va = float(flame_a["attrs"].get(key, 0))
            vb = float(flame_b["attrs"].get(key, 0))
            interp = _lerp(va, vb, t)
            if key in _INTEGER_ATTRS:
                result["attrs"][key] = str(int(round(interp)))
            else:
                result["attrs"][key] = str(interp)
        except (ValueError, TypeError):
            pass

    # Interpolate only affine transforms (coefs + post) per xform
    min_xforms = min(len(flame_a["xforms"]), len(flame_b["xforms"]))
    for i in range(min_xforms):
        xf_a = flame_a["xforms"][i]
        xf_b = flame_b["xforms"][i]

        if "coefs" in xf_a and "coefs" in xf_b:
            result["xforms"][i]["coefs"] = _lerp_list(xf_a["coefs"], xf_b["coefs"], t)
        if "post" in xf_a and "post" in xf_b:
            result["xforms"][i]["post"] = _lerp_list(xf_a["post"], xf_b["post"], t)

        # Also blend weight so relative xform importance shifts
        if "weight" in xf_a and "weight" in xf_b:
            result["xforms"][i]["weight"] = _lerp(xf_a["weight"], xf_b["weight"], t)

    # Interpolate finalxform geometry if both have one
    if flame_a.get("finalxform") and flame_b.get("finalxform"):
        fxf_a = flame_a["finalxform"]
        fxf_b = flame_b["finalxform"]
        if "coefs" in fxf_a and "coefs" in fxf_b:
            result["finalxform"]["coefs"] = _lerp_list(fxf_a["coefs"], fxf_b["coefs"], t)
        if "post" in fxf_a and "post" in fxf_b:
            result["finalxform"]["post"] = _lerp_list(fxf_a["post"], fxf_b["post"], t)

    # Interpolate palette
    if flame_a.get("palette") and flame_b.get("palette"):
        result["palette"] = _lerp_palette(flame_a["palette"], flame_b["palette"], t)

    return result


def crossover_matched(flame_a: dict, flame_b: dict, t: float) -> dict:
    """Full interpolation between structurally matched flames.

    Both flames should have the same variation types per xform.
    Interpolates everything: coefs, post, variation params, weights, palette.
    """
    result = deepcopy(flame_a)

    # Interpolate flame-level attributes
    for key in result["attrs"]:
        if key in _SKIP_ATTRS:
            continue
        try:
            va = float(flame_a["attrs"].get(key, 0))
            vb = float(flame_b["attrs"].get(key, 0))
            interp = _lerp(va, vb, t)
            if key in _INTEGER_ATTRS:
                result["attrs"][key] = str(int(round(interp)))
            else:
                result["attrs"][key] = str(interp)
        except (ValueError, TypeError):
            pass

    # Interpolate all xform parameters
    min_xforms = min(len(flame_a["xforms"]), len(flame_b["xforms"]))
    result["xforms"] = result["xforms"][:min_xforms]

    for i in range(min_xforms):
        xf_a = flame_a["xforms"][i]
        xf_b = flame_b["xforms"][i]
        blended = {}

        all_keys = set(xf_a.keys()) | set(xf_b.keys())
        for key in all_keys:
            val_a = xf_a.get(key)
            val_b = xf_b.get(key)

            if key in ("coefs", "post", "chaos"):
                if val_a is not None and val_b is not None:
                    blended[key] = _lerp_list(val_a, val_b, t)
                elif val_a is not None:
                    blended[key] = list(val_a)
                else:
                    blended[key] = list(val_b)
                continue

            if val_a is not None and val_b is not None:
                try:
                    fa = float(val_a)
                    fb = float(val_b)
                    interp = _lerp(fa, fb, t)
                    if key in ("weight", "color", "symmetry", "opacity"):
                        blended[key] = interp
                    elif key in _INTEGER_XFORM_KEYS:
                        blended[key] = str(int(round(interp)))
                    else:
                        blended[key] = str(interp)
                    continue
                except (ValueError, TypeError):
                    pass

            # One side only — keep from dominant side
            if val_a is not None and val_b is None:
                blended[key] = val_a
            elif val_b is not None:
                blended[key] = val_b

        result["xforms"][i] = blended

    # Interpolate finalxform
    if flame_a.get("finalxform") and flame_b.get("finalxform"):
        fxf_a = flame_a["finalxform"]
        fxf_b = flame_b["finalxform"]
        blended = {}
        for key in set(fxf_a.keys()) | set(fxf_b.keys()):
            va = fxf_a.get(key)
            vb = fxf_b.get(key)
            if key in ("coefs", "post"):
                if va is not None and vb is not None:
                    blended[key] = _lerp_list(va, vb, t)
                else:
                    blended[key] = list(va or vb)
            elif va is not None and vb is not None:
                try:
                    blended[key] = _lerp(float(va), float(vb), t)
                    if key not in ("weight", "color", "symmetry", "opacity"):
                        blended[key] = str(blended[key])
                except (ValueError, TypeError):
                    blended[key] = va
            else:
                blended[key] = va if va is not None else vb
        result["finalxform"] = blended
    elif t < 0.5 and flame_a.get("finalxform"):
        result["finalxform"] = deepcopy(flame_a["finalxform"])
    elif t >= 0.5 and flame_b.get("finalxform"):
        result["finalxform"] = deepcopy(flame_b["finalxform"])
    else:
        result["finalxform"] = None

    # Interpolate palette
    if flame_a.get("palette") and flame_b.get("palette"):
        result["palette"] = _lerp_palette(flame_a["palette"], flame_b["palette"], t)

    return result


# ---------------------------------------------------------------------------
# Pairing logic
# ---------------------------------------------------------------------------

def build_matched_pairs(
    sources: list[tuple[str, dict]],
    rng: random.Random,
    num_pairs: int,
    strict: bool = True,
) -> list[tuple[tuple[str, dict], tuple[str, dict]]]:
    """Build pairs of flames with matching variation structure.

    If strict=True, requires exact same signature (xform count + per-xform
    variation names). If strict=False, requires same overall variation set
    but allows different xform arrangements.
    """
    if strict:
        groups = defaultdict(list)
        for item in sources:
            sig = get_flame_signature(item[1])
            groups[sig].append(item)
    else:
        groups = defaultdict(list)
        for item in sources:
            var_set = get_flame_variation_set(item[1])
            n_xforms = len(item[1]["xforms"])
            groups[(n_xforms, var_set)].append(item)

    # Filter to groups with at least 2 members
    eligible = {k: v for k, v in groups.items() if len(v) >= 2}

    print(f"  Structural groups found: {len(groups)}")
    print(f"  Groups with 2+ members: {len(eligible)}")
    if eligible:
        sizes = sorted([len(v) for v in eligible.values()], reverse=True)
        print(f"  Largest groups: {sizes[:10]}")

    if not eligible:
        print("  WARNING: No compatible pairs found!")
        return []

    pairs = []
    eligible_list = list(eligible.values())
    for _ in range(num_pairs):
        group = rng.choice(eligible_list)
        a, b = rng.sample(group, 2)
        pairs.append((a, b))

    return pairs


def build_random_pairs(
    sources: list[tuple[str, dict]],
    rng: random.Random,
    num_pairs: int,
) -> list[tuple[tuple[str, dict], tuple[str, dict]]]:
    """Build random pairs (any two flames)."""
    pairs = []
    for _ in range(num_pairs):
        a, b = rng.sample(sources, 2)
        pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate parameter-space crossovers between flame pairs"
    )
    parser.add_argument(
        "--flames", "-f",
        type=Path,
        required=True,
        help="Directory containing source .flame files to pair up",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for crossover .flame files",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["geometry", "matched", "full"],
        default="geometry",
        help="Crossover mode (default: geometry)",
    )
    parser.add_argument(
        "--pairs", "-p",
        type=int,
        default=500,
        help="Number of random pairs to generate (default: 500)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Interpolation steps per pair (default: 5)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=400,
        help="Random seed (default: 400)",
    )
    parser.add_argument(
        "--new-palette",
        action="store_true",
        help="Replace palette with a random one instead of interpolating",
    )
    parser.add_argument(
        "--loose-match",
        action="store_true",
        help="For matched mode: allow same variation set with different xform arrangement",
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Load all source flames
    flame_files = sorted(args.flames.glob("*.flame"))
    print(f"Found {len(flame_files)} source flame files")

    sources = []
    for fp in flame_files:
        parsed = parse_flame_file(fp)
        if parsed:
            sources.append((fp.stem, parsed[0]))

    print(f"Parsed {len(sources)} valid flames")
    if len(sources) < 2:
        print("ERROR: Need at least 2 source flames for crossover.")
        return

    # Build pairs based on mode
    print(f"\nMode: {args.mode}")
    if args.mode == "matched":
        pairs = build_matched_pairs(
            sources, rng, args.pairs, strict=not args.loose_match
        )
        if not pairs:
            return
    else:
        pairs = build_random_pairs(sources, rng, args.pairs)

    # Select crossover function
    if args.mode == "geometry":
        cross_fn = crossover_geometry_only
    elif args.mode == "matched":
        cross_fn = crossover_matched
    else:
        cross_fn = crossover_matched  # full uses same fn, just random pairs

    # Generate interpolation ratios
    if args.steps == 5:
        ratios = [0.2, 0.35, 0.5, 0.65, 0.8]
    else:
        ratios = [i / (args.steps + 1) for i in range(1, args.steps + 1)]

    args.output.mkdir(parents=True, exist_ok=True)
    total = len(pairs) * len(ratios)
    print(f"Generating {len(pairs)} pairs x {len(ratios)} steps = {total} crossover flames")
    print(f"  Blend ratios: {[f'{r:.2f}' for r in ratios]}")

    idx = 0
    for pair_idx, ((stem_a, flame_a), (stem_b, flame_b)) in enumerate(pairs):
        for step, t in enumerate(ratios):
            crossed = cross_fn(flame_a, flame_b, t)

            if args.new_palette:
                crossed = replace_palette(crossed, random_palette(rng))

            crossed = adjust_for_vignette(crossed, rng)

            crossed["attrs"]["name"] = (
                f"{args.mode}_{stem_a[:12]}x{stem_b[:12]}_t{t:.2f}"
            )

            out_path = args.output / f"{args.mode}_{pair_idx:05d}_t{step}.flame"
            write_flame_file([crossed], out_path)
            idx += 1

        if (pair_idx + 1) % 100 == 0 or pair_idx == 0:
            print(f"  {pair_idx + 1}/{len(pairs)} pairs done ({idx} flames written)")

    print(f"\nDone. {idx} crossover flames written to {args.output}/")


if __name__ == "__main__":
    main()
