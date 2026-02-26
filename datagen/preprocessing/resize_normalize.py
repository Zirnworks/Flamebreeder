"""Resize and normalize images to a consistent format."""

from pathlib import Path

from PIL import Image


def resize_image(img: Image.Image, target_size: int = 512) -> Image.Image:
    """Resize image to target_size x target_size using Lanczos resampling."""
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.LANCZOS)
    return img.convert("RGB")


def process_file(
    input_path: Path, output_path: Path, target_size: int = 512
) -> bool:
    """Load, resize, and save a single image.

    Returns True on success.
    """
    try:
        img = Image.open(input_path)
        img = resize_image(img, target_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "PNG")
        return True
    except Exception:
        return False
