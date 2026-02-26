"""Quality filtering for generated fractal flame images.

Rejects images that are blank, low-contrast, or near-duplicates.
Supports brightness correction for dark-but-salvageable images,
field detection for images that fill the frame, and radial
spotlight effects to convert fields into vignettes.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import imagehash


# ---------------------------------------------------------------------------
# Brightness correction
# ---------------------------------------------------------------------------

def auto_levels(img: Image.Image, target_peak: float = 0.70, min_peak: float = 8.0) -> Image.Image | None:
    """Stretch brightness so the brightest content pixels reach target_peak.

    Uses the 99.5th percentile as the effective peak to avoid outlier pixels.

    Args:
        img: Input image.
        target_peak: Desired brightness for the peak content, as a fraction of 255.
            0.70 means ~178/255.
        min_peak: Minimum 99.5th-percentile luminance required for the image to be
            considered salvageable. Below this, the image is truly empty.

    Returns:
        Corrected image, or None if the image is too empty to salvage.
    """
    arr = np.array(img).astype(np.float32)
    luminance = arr.max(axis=2)
    peak = np.percentile(luminance, 99.5)

    if peak < min_peak:
        return None

    target = target_peak * 255.0
    if peak >= target:
        # Already bright enough, no correction needed
        return img

    scale = target / peak
    corrected = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(corrected)


# ---------------------------------------------------------------------------
# Spotlight effect for field-like images
# ---------------------------------------------------------------------------

def apply_spotlight(img: Image.Image, falloff: float = 2.0, radius: float = 0.65) -> Image.Image:
    """Apply radial gradient darkening to convert a field into a vignette.

    Center stays bright, edges fade to black following a power curve.

    Args:
        img: Input image.
        falloff: Power curve exponent. Higher = tighter spotlight.
        radius: Fraction of the half-diagonal where the mask reaches zero.
            1.0 means corners are fully black; < 1.0 darkens more aggressively.
    """
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    cy, cx = h / 2.0, w / 2.0

    Y, X = np.ogrid[:h, :w]
    max_dist = np.sqrt(cx ** 2 + cy ** 2) * radius
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max_dist

    mask = np.clip(1.0 - dist ** falloff, 0, 1)
    result = (arr * mask[:, :, np.newaxis]).astype(np.uint8)
    return Image.fromarray(result)


# ---------------------------------------------------------------------------
# Individual quality checks
# ---------------------------------------------------------------------------

def is_blank(img: Image.Image, threshold: float = 0.98) -> bool:
    """Reject images where >threshold fraction of pixels are a single color."""
    arr = np.array(img)
    total = arr.shape[0] * arr.shape[1]

    # Check if most pixels match the corner pixel (likely background)
    bg = arr[0, 0]
    matching = np.all(np.abs(arr.astype(int) - bg.astype(int)) < 10, axis=2)
    fraction = matching.sum() / total

    return fraction > threshold


def is_low_contrast(img: Image.Image, min_std: float = 15.0, bg_threshold: int = 15) -> bool:
    """Reject images with low variation in the content region.

    Measures standard deviation among non-background pixels so that
    small bright objects on large black backgrounds are not penalized.
    """
    arr = np.array(img).astype(np.float32)
    luminance = arr.max(axis=2)

    content_mask = luminance > bg_threshold
    if content_mask.sum() < 100:
        # Too few content pixels — let too_dark handle it
        return True

    content_pixels = arr[content_mask]
    return content_pixels.std() < min_std


def is_too_dark(img: Image.Image, max_dark_fraction: float = 0.95, dark_threshold: int = 10) -> bool:
    """Reject images that are nearly all black."""
    arr = np.array(img)
    dark = (arr.max(axis=2) < dark_threshold)
    return dark.mean() > max_dark_fraction


def is_too_bright(img: Image.Image, max_bright_fraction: float = 0.95, bright_threshold: int = 245) -> bool:
    """Reject images that are nearly all white."""
    arr = np.array(img)
    bright = (arr.min(axis=2) > bright_threshold)
    return bright.mean() > max_bright_fraction


def is_edge_heavy(
    img: Image.Image,
    border_fraction: float = 0.05,
    max_edge_brightness: float = 0.30,
    max_single_edge_brightness: float = 0.45,
    dark_threshold: int = 20,
) -> bool:
    """Reject images where too many border pixels are non-black.

    Checks both overall border brightness and per-side brightness.
    A fractal that bleeds through one side while leaving others clean
    is still not a good vignette.
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    band = max(int(min(h, w) * border_fraction), 2)

    # Collect border pixels per side
    sides = {
        "top": arr[:band, :, :].reshape(-1, 3),
        "bottom": arr[-band:, :, :].reshape(-1, 3),
        "left": arr[band:-band, :band, :].reshape(-1, 3),
        "right": arr[band:-band, -band:, :].reshape(-1, 3),
    }

    all_border = np.concatenate(list(sides.values()), axis=0)
    overall_bright = (all_border.max(axis=1) > dark_threshold).mean()
    if overall_bright > max_edge_brightness:
        return True

    # Also reject if any single side is too bright
    for side_pixels in sides.values():
        side_bright = (side_pixels.max(axis=1) > dark_threshold).mean()
        if side_bright > max_single_edge_brightness:
            return True

    return False


def is_off_center(
    img: Image.Image,
    max_offset_fraction: float = 0.25,
    dark_threshold: int = 15,
) -> bool:
    """Reject images where the luminance center-of-mass is far from image center.

    Computes a brightness-weighted centroid and checks if it's within
    max_offset_fraction of the image center.
    """
    arr = np.array(img).astype(np.float32)
    luminance = arr.max(axis=2)

    # Mask out near-black pixels to avoid noise dominating
    mask = luminance > dark_threshold
    if mask.sum() < 100:
        return False  # too few bright pixels to judge, let other filters handle

    h, w = luminance.shape
    ys, xs = np.mgrid[:h, :w]
    weights = luminance * mask

    total_w = weights.sum()
    if total_w < 1e-6:
        return False

    cx = (weights * xs).sum() / total_w
    cy = (weights * ys).sum() / total_w

    # Distance from center as fraction of image size
    offset_x = abs(cx - w / 2.0) / w
    offset_y = abs(cy - h / 2.0) / h

    return offset_x > max_offset_fraction or offset_y > max_offset_fraction


def has_low_background_ratio(
    img: Image.Image,
    min_background_fraction: float = 0.35,
    dark_threshold: int = 15,
) -> bool:
    """Reject images where not enough of the image is near-black background.

    For vignette-style images, a significant portion should be black background.
    If less than min_background_fraction of pixels are dark, the image is
    too field-like.
    """
    arr = np.array(img)
    dark = arr.max(axis=2) < dark_threshold
    return dark.mean() < min_background_fraction


def filter_image(img: Image.Image) -> tuple[bool, str]:
    """Run all quality filters on an image (legacy interface).

    Returns:
        (passed, reason): True if image passes all filters, else False with rejection reason.
    """
    category, _processed = classify_image(img)
    return (category == "ok", category)


def classify_image(img: Image.Image) -> tuple[str, Image.Image]:
    """Run quality filters with brightness correction and field detection.

    Pipeline:
        1. Reject blanks (>98% single color)
        2. Attempt auto-levels on dark/low-contrast images before rejecting
        3. Reject truly empty images (too_dark after correction attempt)
        4. Reject too_bright
        5. Classify field-like images (low background ratio) as "field"
           instead of rejecting — these can be spotlight-salvaged
        6. Reject edge_heavy, off_center

    Returns:
        (category, processed_img) where category is one of:
        - "ok": passes all filters; processed_img may be brightness-corrected
        - "field": fills the frame; processed_img is brightness-corrected if needed
        - rejection reason string: "blank", "too_dark", "too_bright",
          "edge_heavy", "off_center", "low_contrast"
    """
    if is_blank(img):
        return "blank", img

    # Attempt brightness correction before contrast/darkness checks.
    # This rescues images with real content that are just rendered too dim.
    corrected = auto_levels(img)
    if corrected is None:
        # Peak luminance below min_peak — truly empty
        return "too_dark", img

    working = corrected

    # Re-check after correction
    if is_low_contrast(working):
        return "low_contrast", img
    if is_too_dark(working):
        return "too_dark", img
    if is_too_bright(working):
        return "too_bright", img

    # Field detection — route instead of reject
    if has_low_background_ratio(working):
        return "field", working

    if is_edge_heavy(working):
        return "edge_heavy", working
    if is_off_center(working):
        return "off_center", img

    return "ok", working


def deduplicate(
    image_paths: list[Path], hash_size: int = 16, threshold: int = 10
) -> tuple[list[Path], list[Path]]:
    """Remove near-duplicate images using perceptual hashing.

    Args:
        image_paths: List of image file paths.
        hash_size: Size of the perceptual hash (higher = more discriminating).
        threshold: Maximum Hamming distance to consider as duplicate.

    Returns:
        (kept, removed): Lists of paths that were kept and removed.
    """
    hashes: dict[str, tuple[imagehash.ImageHash, Path]] = {}
    kept = []
    removed = []

    for path in image_paths:
        try:
            img = Image.open(path)
            h = imagehash.phash(img, hash_size=hash_size)

            is_dup = False
            for existing_key, (existing_hash, existing_path) in hashes.items():
                if h - existing_hash < threshold:
                    is_dup = True
                    removed.append(path)
                    break

            if not is_dup:
                hashes[str(h)] = (h, path)
                kept.append(path)

        except Exception:
            removed.append(path)

    return kept, removed
