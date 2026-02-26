"""Batch-generate mutated JWildfire .flame files from a preset collection.

For each output flame, picks a random preset from the collection,
applies mutations (palette randomization, camera vignette adjustment,
optional parameter perturbation), and writes a render-ready .flame file.

Usage:
    python -m jwildfire.mutate_batch \
        --presets /path/to/JWildfire_Flames \
        --output data/mutated_flames \
        --count 15000 \
        --seed 42
"""

import argparse
import random
import sys
from pathlib import Path

from .flame_parser import collect_all_flames
from .flame_mutator import mutate_flame
from .flame_writer import write_flame_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate mutated JWildfire flame files for dataset generation"
    )
    parser.add_argument(
        "--presets", "-p",
        type=Path,
        required=True,
        help="Root directory of JWildfire .flame preset collection",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/mutated_flames"),
        help="Output directory for mutated .flame files (default: data/mutated_flames)",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=15000,
        help="Number of mutated flames to generate (default: 15000)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--variants-per-preset",
        type=int,
        default=0,
        help="Max variants per source preset (0 = no limit, distribute evenly)",
    )
    parser.add_argument(
        "--no-perturb",
        action="store_true",
        help="Disable xform parameter perturbation (palette + camera only)",
    )
    parser.add_argument(
        "--perturb-strength",
        type=float,
        default=0.05,
        help="Perturbation strength for xform params (default: 0.05)",
    )
    parser.add_argument(
        "--perturb-range",
        type=str,
        default=None,
        help="Perturbation strength range as min,max (e.g. '0.05,0.25'). "
             "Overrides --perturb-strength. Each flame samples a random "
             "strength from this range for more varied output.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for output filenames (default: 0). "
             "Use to append to an existing batch without overwriting.",
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Parse perturbation range
    perturb_range = None
    if args.perturb_range:
        parts = args.perturb_range.split(",")
        perturb_range = (float(parts[0]), float(parts[1]))

    # Collect all source flames
    print(f"Scanning presets in {args.presets}...")
    all_flames = collect_all_flames(args.presets)
    if not all_flames:
        print("ERROR: No parseable .flame files found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(all_flames)} source flames from {len(set(str(p) for p, _ in all_flames))} files")

    # Prepare output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Generate mutated flames
    start_idx = args.start_index
    print(f"Generating {args.count} mutated flames (indices {start_idx}-{start_idx + args.count - 1})...")
    if perturb_range:
        print(f"  Perturbation range: {perturb_range[0]:.2f} - {perturb_range[1]:.2f}")
    else:
        print(f"  Perturbation strength: {args.perturb_strength}")

    for i in range(args.count):
        idx = start_idx + i
        # Pick a random source flame
        _source_path, source_flame = rng.choice(all_flames)

        # Mutate
        mutated = mutate_flame(
            source_flame,
            rng=rng,
            replace_colors=True,
            adjust_camera=True,
            perturb=not args.no_perturb,
            perturb_strength=args.perturb_strength,
            perturb_strength_range=perturb_range,
        )

        # Update name for tracking
        original_name = mutated["attrs"].get("name", "unknown")
        mutated["attrs"]["name"] = f"mutant_{idx:05d}_from_{original_name[:30]}"

        # Write individual flame file
        out_path = args.output / f"{idx:05d}.flame"
        write_flame_file([mutated], out_path)

        if (i + 1) % 1000 == 0 or i == 0:
            print(f"  {i + 1}/{args.count} generated")

    print(f"\nDone. {args.count} flame files written to {args.output}/")
    print(f"Next step: render with JWildfire CLI:")
    print(f"  java -cp j-wildfire.jar org.jwildfire.cli.FlameRenderer \\")
    print(f"    -flameFile <file.flame> -width 512 -height 512 -quality 200")


if __name__ == "__main__":
    main()
