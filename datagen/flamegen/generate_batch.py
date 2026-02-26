"""CLI entry point for batch fractal flame generation.

Generates N fractal flame images with random parameters using multiprocessing.
"""

import os
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import click
import numpy as np
from tqdm import tqdm

from .ifs import chaos_game
from .params import random_flame_params
from .renderer import render_flame


def generate_single_flame(args: tuple) -> str | None:
    """Generate a single fractal flame image.

    Args:
        args: Tuple of (index, output_dir, size, iterations, base_seed).

    Returns:
        Output path on success, None on failure.
    """
    idx, output_dir, size, iterations, base_seed = args

    try:
        rng = np.random.default_rng(base_seed + idx)
        params = random_flame_params(rng)

        # Allocate histogram buffers
        hist_count = np.zeros((size, size), dtype=np.float64)
        hist_color = np.zeros((size, size, 3), dtype=np.float64)

        # Run the chaos game
        chaos_game(
            num_iterations=iterations,
            size=size,
            num_xforms=params["num_xforms"],
            affine_coeffs=params["affine_coeffs"],
            variation_ids=params["variation_ids"],
            variation_weights=params["variation_weights"],
            num_variations=params["num_variations"],
            xform_weights=params["xform_weights"],
            xform_colors=params["xform_colors"],
            has_post=params["has_post"],
            post_coeffs=params["post_coeffs"],
            has_final=params["has_final"],
            final_affine=params["final_affine"],
            final_var_ids=params["final_var_ids"],
            final_var_weights=params["final_var_weights"],
            final_num_vars=params["final_num_vars"],
            cam_x_center=params["cam_x_center"],
            cam_y_center=params["cam_y_center"],
            cam_scale=params["cam_scale"],
            cam_rotation=params["cam_rotation"],
            hist_count=hist_count,
            hist_color=hist_color,
            palette=params["palette"],
            seed=int(rng.integers(0, 2**31)),
        )

        # Render to image
        img = render_flame(
            hist_count,
            hist_color,
            gamma=params["gamma"],
            brightness=params["brightness"],
            vibrancy=params["vibrancy"],
        )

        # Save
        out_path = os.path.join(output_dir, f"{idx:05d}.png")
        img.save(out_path)
        return out_path

    except Exception as e:
        return None


@click.command()
@click.option("--count", "-n", default=10, help="Number of images to generate.")
@click.option("--output", "-o", default="data/raw_python", help="Output directory.")
@click.option("--size", "-s", default=512, help="Image size (square).")
@click.option(
    "--iterations", "-i", default=5_000_000,
    help="Chaos game iterations per image.",
)
@click.option("--workers", "-w", default=None, type=int, help="Number of worker processes.")
@click.option("--seed", default=42, help="Base random seed.")
def main(count: int, output: str, size: int, iterations: int, workers: int | None, seed: int):
    """Generate a batch of random fractal flame images."""
    if workers is None:
        workers = min(cpu_count(), 8)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Generating {count} fractal flames at {size}x{size}")
    click.echo(f"  Iterations per image: {iterations:,}")
    click.echo(f"  Workers: {workers}")
    click.echo(f"  Output: {output_dir}")
    click.echo()

    # Warm up numba JIT by running a tiny flame first
    click.echo("Warming up JIT compiler...")
    t0 = time.time()
    generate_single_flame((0, str(output_dir), 64, 10_000, seed + 1_000_000))
    click.echo(f"  JIT warmup took {time.time() - t0:.1f}s")
    click.echo()

    args = [(i, str(output_dir), size, iterations, seed) for i in range(count)]

    t0 = time.time()
    success = 0
    failed = 0

    if workers == 1:
        for a in tqdm(args, desc="Generating"):
            result = generate_single_flame(a)
            if result:
                success += 1
            else:
                failed += 1
    else:
        with Pool(workers) as pool:
            for result in tqdm(
                pool.imap_unordered(generate_single_flame, args),
                total=count,
                desc="Generating",
            ):
                if result:
                    success += 1
                else:
                    failed += 1

    elapsed = time.time() - t0
    click.echo()
    click.echo(f"Done in {elapsed:.1f}s ({elapsed/count:.2f}s per image)")
    click.echo(f"  Success: {success}")
    click.echo(f"  Failed:  {failed}")


if __name__ == "__main__":
    main()
