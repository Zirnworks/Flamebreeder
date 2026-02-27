"""Generate variations from a curated set of favorite flames.

Given a folder of PNGs (selected by the user), maps each back to its
.flame source file and generates N mutations per favorite with varied
perturbation strengths and palette swaps.

Usage:
    python -m datagen.jwildfire.generate_from_favorites \
        --favorites /path/to/selected_pngs \
        --flames /path/to/mutated_flames \
        --output /path/to/output_flames \
        --variants-per 50 \
        --seed 300
"""

import argparse
import random
from pathlib import Path

from .flame_parser import parse_flame_file
from .flame_mutator import mutate_flame
from .flame_writer import write_flame_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate variations from curated favorite flames"
    )
    parser.add_argument(
        "--favorites", "-f",
        type=Path,
        required=True,
        help="Directory of selected PNGs (filenames map to .flame files)",
    )
    parser.add_argument(
        "--flames", "-l",
        type=Path,
        required=True,
        help="Directory containing source .flame files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for mutated .flame files",
    )
    parser.add_argument(
        "--variants-per", "-n",
        type=int,
        default=50,
        help="Number of variants to generate per favorite (default: 50)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=300,
        help="Random seed (default: 300)",
    )
    parser.add_argument(
        "--perturb-range",
        type=str,
        default="0.02,0.10",
        help="Perturbation strength range as min,max (default: '0.02,0.10')",
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

    parts = args.perturb_range.split(",")
    perturb_range = (float(parts[0]), float(parts[1]))

    # Find all selected PNGs and map to .flame files
    png_files = sorted(args.favorites.glob("*.png"))
    print(f"Found {len(png_files)} selected favorites")

    sources = []
    missing = 0
    for png in png_files:
        stem = png.stem
        flame_path = args.flames / f"{stem}.flame"
        if flame_path.exists():
            flames = parse_flame_file(flame_path)
            if flames:
                sources.append((stem, flames[0]))
            else:
                missing += 1
                print(f"  Warning: could not parse {flame_path}")
        else:
            missing += 1
            print(f"  Warning: no .flame file for {stem}")

    print(f"Matched {len(sources)} source flames ({missing} missing)")
    if not sources:
        print("ERROR: No source flames found.")
        return

    # Generate variations
    args.output.mkdir(parents=True, exist_ok=True)
    total = len(sources) * args.variants_per
    print(f"Generating {args.variants_per} variants x {len(sources)} sources = {total} flames")
    print(f"  Perturbation range: {perturb_range[0]:.2f} - {perturb_range[1]:.2f}")

    idx = 0
    for src_idx, (stem, source_flame) in enumerate(sources):
        for v in range(args.variants_per):
            mutated = mutate_flame(
                source_flame,
                rng=rng,
                replace_colors=True,
                adjust_camera=True,
                perturb=True,
                perturb_strength_range=perturb_range,
            )

            original_name = mutated["attrs"].get("name", "unknown")
            mutated["attrs"]["name"] = f"fav_{stem}_v{v:03d}_{original_name[:20]}"

            out_path = args.output / f"fav_{stem}_v{v:03d}.flame"
            write_flame_file([mutated], out_path)
            idx += 1

        if (src_idx + 1) % 50 == 0 or src_idx == 0:
            print(f"  {src_idx + 1}/{len(sources)} sources done ({idx} flames written)")

    print(f"\nDone. {idx} flame files written to {args.output}/")


if __name__ == "__main__":
    main()
