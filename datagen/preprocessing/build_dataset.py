"""Assemble the final training dataset from raw images.

Applies quality filtering (with brightness correction), deduplication,
resizing, and produces a consistently named dataset ready for GAN training.

Field-like images are routed to a separate folder with spotlight-treated
copies for later review and potential reintegration.
"""

import json
import random
import shutil
from pathlib import Path

import click
from PIL import Image
from tqdm import tqdm

from .quality_filter import classify_image, apply_spotlight, deduplicate
from .resize_normalize import resize_image, random_orientation


@click.command()
@click.option(
    "--raw-dirs", "-r", multiple=True, required=True,
    help="Raw image directories (can specify multiple).",
)
@click.option("--output", "-o", default="data/processed", help="Output directory.")
@click.option("--size", "-s", default=512, help="Target image size.")
@click.option("--validation-split", default=1000, help="Number of images for validation set.")
@click.option("--skip-dedup", is_flag=True, help="Skip deduplication step.")
@click.option("--spotlight-fields/--no-spotlight-fields", default=True,
              help="Apply spotlight effect to field images (default: on).")
@click.option("--augment-orientation/--no-augment-orientation", default=True,
              help="Randomly rotate/flip each image (D4 symmetry group). Default: on.")
@click.option("--seed", default=42, help="RNG seed for orientation augmentation.")
def main(
    raw_dirs: tuple[str],
    output: str,
    size: int,
    validation_split: int,
    skip_dedup: bool,
    spotlight_fields: bool,
    augment_orientation: bool,
    seed: int,
):
    """Build the final training dataset from raw generated images."""
    output_dir = Path(output)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    fields_dir = output_dir / "fields"
    fields_spotlit_dir = output_dir / "fields_spotlit"
    edge_dir = output_dir / "edge_heavy"
    edge_spotlit_dir = output_dir / "edge_heavy_spotlit"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    fields_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)
    if spotlight_fields:
        fields_spotlit_dir.mkdir(parents=True, exist_ok=True)
        edge_spotlit_dir.mkdir(parents=True, exist_ok=True)

    # Collect all raw images
    raw_paths = []
    for d in raw_dirs:
        p = Path(d)
        if not p.exists():
            click.echo(f"Warning: {d} does not exist, skipping")
            continue
        raw_paths.extend(sorted(p.glob("*.png")) + sorted(p.glob("*.jpg")))

    click.echo(f"Found {len(raw_paths)} raw images")

    # Quality filtering with brightness correction and field detection
    click.echo("Filtering images...")
    passed = []          # (path, corrected_img) tuples
    field_images = []    # (path, corrected_img) tuples
    edge_images = []     # (path, corrected_img) tuples
    filter_stats = {
        "blank": 0, "low_contrast": 0, "too_dark": 0, "too_bright": 0,
        "edge_heavy": 0, "off_center": 0, "field": 0, "ok": 0,
        "brightness_corrected": 0,
    }

    for path in tqdm(raw_paths, desc="Quality filter"):
        try:
            img = Image.open(path).convert("RGB")
            category, processed = classify_image(img)
            filter_stats[category] = filter_stats.get(category, 0) + 1

            # Track how many got brightness-corrected
            if category in ("ok", "field", "edge_heavy") and processed is not img:
                filter_stats["brightness_corrected"] += 1

            if category == "ok":
                passed.append((path, processed))
            elif category == "field":
                field_images.append((path, processed))
            elif category == "edge_heavy":
                edge_images.append((path, processed))
        except Exception:
            filter_stats["error"] = filter_stats.get("error", 0) + 1

    click.echo(f"Quality filter results: {json.dumps(filter_stats, indent=2)}")
    click.echo(f"Passed: {len(passed)}/{len(raw_paths)} "
               f"(+{len(field_images)} fields, +{len(edge_images)} edge_heavy routed separately)")

    # Deduplication (on passed images only)
    if not skip_dedup and len(passed) > 0:
        click.echo("Deduplicating...")
        passed_paths = [p for p, _ in passed]
        kept_paths, removed = deduplicate(passed_paths)
        kept_set = set(str(p) for p in kept_paths)
        passed = [(p, img) for p, img in passed if str(p) in kept_set]
        click.echo(f"Removed {len(removed)} near-duplicates, kept {len(passed)}")

    # Resize and save training images
    rng = random.Random(seed)
    if augment_orientation:
        click.echo(f"Processing {len(passed)} images to {size}x{size} with random orientation (seed={seed})...")
    else:
        click.echo(f"Processing {len(passed)} images to {size}x{size}...")

    # Split: last N for validation
    if validation_split > 0 and len(passed) > validation_split:
        val_items = passed[-validation_split:]
        train_items = passed[:-validation_split]
    else:
        train_items = passed
        val_items = []

    for idx, (path, processed) in enumerate(tqdm(train_items, desc="Train set")):
        try:
            img = resize_image(processed, size)
            if augment_orientation:
                img = random_orientation(img, rng)
            img.save(train_dir / f"{idx:05d}.png", "PNG")
        except Exception:
            pass

    for idx, (path, processed) in enumerate(tqdm(val_items, desc="Val set")):
        try:
            img = resize_image(processed, size)
            if augment_orientation:
                img = random_orientation(img, rng)
            img.save(val_dir / f"{idx:05d}.png", "PNG")
        except Exception:
            pass

    # Save field images + spotlight versions
    if field_images:
        click.echo(f"Saving {len(field_images)} field images...")
        for idx, (path, processed) in enumerate(tqdm(field_images, desc="Fields")):
            try:
                img = resize_image(processed, size)
                if augment_orientation:
                    img = random_orientation(img, rng)
                img.save(fields_dir / f"{idx:05d}.png", "PNG")
                if spotlight_fields:
                    spotlit = apply_spotlight(img)
                    spotlit.save(fields_spotlit_dir / f"{idx:05d}.png", "PNG")
            except Exception:
                pass

    # Save edge_heavy images + spotlight versions
    if edge_images:
        click.echo(f"Saving {len(edge_images)} edge_heavy images...")
        for idx, (path, processed) in enumerate(tqdm(edge_images, desc="Edge heavy")):
            try:
                img = resize_image(processed, size)
                if augment_orientation:
                    img = random_orientation(img, rng)
                img.save(edge_dir / f"{idx:05d}.png", "PNG")
                if spotlight_fields:
                    spotlit = apply_spotlight(img)
                    spotlit.save(edge_spotlit_dir / f"{idx:05d}.png", "PNG")
            except Exception:
                pass

    # Save stats
    stats = {
        "total_raw": len(raw_paths),
        "filter_stats": filter_stats,
        "after_dedup": len(passed),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "field_count": len(field_images),
        "edge_heavy_count": len(edge_images),
        "image_size": size,
        "augment_orientation": augment_orientation,
        "augment_seed": seed if augment_orientation else None,
        "raw_dirs": list(raw_dirs),
    }
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    click.echo(f"\nDataset built:")
    click.echo(f"  Train:              {len(train_items)} images in {train_dir}")
    click.echo(f"  Val:                {len(val_items)} images in {val_dir}")
    click.echo(f"  Fields:             {len(field_images)} images in {fields_dir}")
    click.echo(f"  Edge heavy:         {len(edge_images)} images in {edge_dir}")
    if spotlight_fields:
        click.echo(f"  Fields spotlit:     {len(field_images)} images in {fields_spotlit_dir}")
        click.echo(f"  Edge heavy spotlit: {len(edge_images)} images in {edge_spotlit_dir}")
    click.echo(f"  Stats:              {output_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
