"""Consolidate all training-eligible images into a flat directory with provenance.

Creates:
- A consolidated image directory with sequential 6-digit naming (hardlinks)
- A mirrored flame file directory for re-rendering at any resolution
- A global manifest.json mapping every image to its raw source and flame file

JWildfire-sourced images get their .flame files hardlinked.
Python-sourced images get their parameters regenerated from seed and saved as JSON.
"""

import json
import os
import sys
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

# Deterministic batch ordering for reproducible numbering
BATCH_ORDER = [
    "processed_jwildfire_v2",
    "processed_moderate_v2",
    "processed_favorites_v2",
    "processed_cross_geometry_v2",
    "processed_cross_matched_v2",
    "processed_python_v2",
]

# Training-eligible subdirectories (spotlit versions go into training)
TRAINING_SUBDIRS = ["train", "fields_spotlit", "edge_heavy_spotlit"]

# Map raw directory names to their flame parameter directories
RAW_TO_FLAME_DIR = {
    "raw_jwildfire": "mutated_flames",
    "raw_jwildfire_moderate": "moderate_flames_only",
    "raw_favorites": "mutated_flames_favorites",
    "raw_cross_geometry": "mutated_flames_cross_geometry",
    "raw_cross_matched": "mutated_flames_cross_matched",
}

# Python renders use a different provenance mechanism (seed-based regeneration)
PYTHON_RAW_DIR = "raw_python"
PYTHON_BASE_SEED = 42


def _identify_raw_dir(raw_source: str) -> str | None:
    """Extract the raw directory name from a full raw source path."""
    parts = Path(raw_source).parts
    for part in parts:
        if part.startswith("raw_"):
            return part
    return None


def _flame_path_for_raw(raw_source: str, data_dir: Path) -> Path | None:
    """Derive the flame file path from a raw PNG source path.

    Returns None for Python renders (handled separately).
    """
    raw_dir_name = _identify_raw_dir(raw_source)
    if raw_dir_name is None or raw_dir_name == PYTHON_RAW_DIR:
        return None

    flame_dir_name = RAW_TO_FLAME_DIR.get(raw_dir_name)
    if flame_dir_name is None:
        return None

    raw_basename = Path(raw_source).stem  # e.g., "fav_55046_v003"
    flame_path = data_dir / flame_dir_name / f"{raw_basename}.flame"
    return flame_path


def _python_raw_index(raw_source: str) -> int | None:
    """Extract the image index from a raw_python source path.

    raw_python/00003.png -> 3
    """
    raw_dir_name = _identify_raw_dir(raw_source)
    if raw_dir_name != PYTHON_RAW_DIR:
        return None
    try:
        return int(Path(raw_source).stem)
    except ValueError:
        return None


def _regenerate_python_params(index: int) -> dict:
    """Regenerate flame parameters for a Python render from its seed.

    Uses the same RNG construction as generate_batch.py:
        rng = np.random.default_rng(PYTHON_BASE_SEED + index)
        params = random_flame_params(rng)
    """
    from datagen.flamegen.params import random_flame_params

    rng = np.random.default_rng(PYTHON_BASE_SEED + index)
    params = random_flame_params(rng)

    # Convert numpy arrays to JSON-serializable lists
    serializable = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, (np.integer, np.bool_)):
            serializable[key] = int(value)
        elif isinstance(value, np.floating):
            serializable[key] = float(value)
        else:
            serializable[key] = value

    serializable["_provenance"] = {
        "generator": "datagen.flamegen.generate_batch",
        "base_seed": PYTHON_BASE_SEED,
        "index": index,
        "rng_seed": PYTHON_BASE_SEED + index,
    }
    return serializable


def _category_from_subdir(subdir: str) -> str:
    """Map subdirectory name to a human-readable category."""
    return {
        "train": "train",
        "fields_spotlit": "fields_spotlit",
        "edge_heavy_spotlit": "edge_heavy_spotlit",
    }.get(subdir, subdir)


def _batch_name_from_dir(dir_name: str) -> str:
    """Extract batch name from _v2 directory name."""
    # processed_jwildfire_v2 -> jwildfire
    return dir_name.replace("processed_", "").replace("_v2", "")


@click.command()
@click.option(
    "--data-dir", "-d", required=True,
    help="Base data directory containing processed_*_v2/ and raw_*/ directories.",
)
@click.option(
    "--output", "-o", default=None,
    help="Output directory for consolidated images. Default: <data-dir>/consolidated",
)
@click.option(
    "--flames-output", "-f", default=None,
    help="Output directory for mirrored flame files. Default: <data-dir>/consolidated_flames",
)
def main(data_dir: str, output: str | None, flames_output: str | None):
    """Consolidate training images with full provenance tracking."""
    data_dir = Path(data_dir).expanduser().resolve()
    out_dir = Path(output).expanduser().resolve() if output else data_dir / "consolidated"
    flames_dir = Path(flames_output).expanduser().resolve() if flames_output else data_dir / "consolidated_flames"

    if out_dir.exists() and any(out_dir.iterdir()):
        click.echo(f"ERROR: Output directory {out_dir} already exists and is not empty.")
        click.echo("Remove it first to avoid mixing old and new data.")
        sys.exit(1)

    if flames_dir.exists() and any(flames_dir.iterdir()):
        click.echo(f"ERROR: Flames directory {flames_dir} already exists and is not empty.")
        click.echo("Remove it first to avoid mixing old and new data.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    flames_dir.mkdir(parents=True, exist_ok=True)

    # Discover and validate _v2 directories
    click.echo("Discovering _v2 batch directories...")
    batch_dirs = []
    for batch_name in BATCH_ORDER:
        batch_path = data_dir / batch_name
        manifest_path = batch_path / "manifest.json"
        stats_path = batch_path / "dataset_stats.json"

        if not batch_path.exists():
            click.echo(f"  WARNING: {batch_name} not found, skipping")
            continue
        if not manifest_path.exists():
            click.echo(f"  WARNING: {batch_name} has no manifest.json, skipping")
            continue
        if not stats_path.exists():
            click.echo(f"  WARNING: {batch_name} has no dataset_stats.json, skipping")
            continue

        batch_dirs.append((batch_name, batch_path, manifest_path))
        click.echo(f"  {batch_name}: OK")

    if not batch_dirs:
        click.echo("ERROR: No valid _v2 directories found.")
        sys.exit(1)

    # Process each batch
    global_idx = 0
    global_manifest = {}
    stats = {
        "total_images": 0,
        "flame_files_linked": 0,
        "python_params_saved": 0,
        "flame_missing_warnings": 0,
        "per_batch": {},
    }
    python_indices_to_regenerate = []  # (global_idx, raw_index) pairs

    for batch_name, batch_path, manifest_path in batch_dirs:
        click.echo(f"\nProcessing {batch_name}...")
        with open(manifest_path) as f:
            batch_manifest = json.load(f)

        batch_label = _batch_name_from_dir(batch_name)
        batch_count = 0

        for subdir in TRAINING_SUBDIRS:
            subdir_path = batch_path / subdir
            if not subdir_path.exists():
                continue

            # Get all images in this subdir, sorted for determinism
            images = sorted(subdir_path.glob("*.png"))
            if not images:
                continue

            category = _category_from_subdir(subdir)
            click.echo(f"  {subdir}: {len(images)} images")

            for img_path in images:
                out_name = f"{global_idx:06d}.png"

                # Hardlink image into consolidated directory
                out_path = out_dir / out_name
                os.link(img_path, out_path)

                # Look up raw source from batch manifest
                manifest_key = f"{subdir}/{img_path.name}"
                raw_source = batch_manifest.get(manifest_key)

                if raw_source is None:
                    # Shouldn't happen, but handle gracefully
                    global_manifest[out_name] = {
                        "raw_source": None,
                        "flame_source": None,
                        "batch": batch_label,
                        "category": category,
                    }
                    global_idx += 1
                    batch_count += 1
                    continue

                # Determine flame source
                python_idx = _python_raw_index(raw_source)
                if python_idx is not None:
                    # Python render — defer param regeneration
                    python_indices_to_regenerate.append((global_idx, python_idx))
                    flame_source_str = str(flames_dir / f"{global_idx:06d}.json")
                    global_manifest[out_name] = {
                        "raw_source": raw_source,
                        "flame_source": flame_source_str,
                        "batch": batch_label,
                        "category": category,
                        "python_seed_index": python_idx,
                    }
                else:
                    # JWildfire render — hardlink flame file
                    flame_path = _flame_path_for_raw(raw_source, data_dir)
                    flame_source_str = None

                    if flame_path and flame_path.exists():
                        flame_out = flames_dir / f"{global_idx:06d}.flame"
                        os.link(flame_path, flame_out)
                        flame_source_str = str(flame_path)
                        stats["flame_files_linked"] += 1
                    elif flame_path:
                        click.echo(f"    WARNING: flame file not found: {flame_path}")
                        stats["flame_missing_warnings"] += 1

                    global_manifest[out_name] = {
                        "raw_source": raw_source,
                        "flame_source": flame_source_str,
                        "batch": batch_label,
                        "category": category,
                    }

                global_idx += 1
                batch_count += 1

        stats["per_batch"][batch_label] = batch_count
        click.echo(f"  Subtotal: {batch_count} images")

    stats["total_images"] = global_idx

    # Regenerate Python flame parameters
    if python_indices_to_regenerate:
        click.echo(f"\nRegenerating {len(python_indices_to_regenerate)} Python flame parameters...")
        for g_idx, raw_idx in tqdm(python_indices_to_regenerate, desc="Python params"):
            try:
                params = _regenerate_python_params(raw_idx)
                param_path = flames_dir / f"{g_idx:06d}.json"
                with open(param_path, "w") as f:
                    json.dump(params, f, indent=2)
                stats["python_params_saved"] += 1
            except Exception as e:
                click.echo(f"  WARNING: failed to regenerate params for index {raw_idx}: {e}")

    # Write global manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(global_manifest, f, indent=2)

    # Print summary
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Consolidation complete!")
    click.echo(f"  Images:             {stats['total_images']} in {out_dir}")
    click.echo(f"  Flame files linked: {stats['flame_files_linked']}")
    click.echo(f"  Python params saved:{stats['python_params_saved']}")
    if stats["flame_missing_warnings"]:
        click.echo(f"  ⚠ Missing flames:  {stats['flame_missing_warnings']}")
    click.echo(f"  Manifest:           {manifest_path}")
    click.echo(f"\nPer-batch breakdown:")
    for batch, count in stats["per_batch"].items():
        click.echo(f"  {batch:25s} {count:>7,}")


if __name__ == "__main__":
    main()
