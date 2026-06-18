"""
Day 09: Affine Transform — safe rotate + image pyramid
Date: 2026-06-18
Goal: Rotate images without cropping (safe_rotate), build Gaussian pyramid with
      pyrDown, and stitch 12 rotation angles into a single collage figure.
Runtime: < 10 s for a ~2K image
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_IMG = PROJECT_ROOT / "data" / "raw" / "test_large.jpg"  # 1200×1200
OUTPUT_ROTATION = RESULT_DIR / "day09_rotation_collage.jpg"
OUTPUT_PYRAMID = RESULT_DIR / "day09_pyramid_collage.jpg"


def _rotated_canvas_size(w: int, h: int, angle_deg: float, scale: float = 1.0) -> tuple[int, int]:
    """Compute the minimal canvas (width, height) to hold a rotated image.

    Formula: new_w = h·|sinθ| + w·|cosθ|, scaled and ceiled.
    Used by safe_rotate() and print_rotation_stats().
    """
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = abs(np.cos(rad)), abs(np.sin(rad))
    # subtract epsilon so ceil(1200.0000001) → 1200 instead of 1201
    new_w = int(np.ceil(scale * (h * sin_a + w * cos_a) - 1e-10))
    new_h = int(np.ceil(scale * (h * cos_a + w * sin_a) - 1e-10))
    return new_w, new_h


# ═══════════════════════════════════════════════════════════════════════════════
# 1. safe_rotate() — rotate by angle without clipping content
# ═══════════════════════════════════════════════════════════════════════════════
#
# Standard warpAffine rotates inside the original canvas — corners get cut off.
# Safe rotate: compute a larger canvas, then translate the rotation center
# so all content fits.
#
#   new_w = h·|sinθ| + w·|cosθ|
#   new_h = h·|cosθ| + w·|sinθ|
#   Adjust M[:, 2] += (new_w - w) / 2, (new_h - h) / 2


def safe_rotate(img: np.ndarray, angle_deg: float, scale: float = 1.0) -> tuple[np.ndarray, float]:
    """Rotate img counter-clockwise by angle_deg, auto-expand canvas.

    The standard warpAffine rotates inside the original canvas — corners get clipped.
    This version computes a larger canvas so the entire rotated image fits.

    Args:
        img:       input image (BGR or grayscale)
        angle_deg: rotation angle in degrees (counter-clockwise)
        scale:     optional uniform scale factor

    Returns:
        (rotated_image, expansion_ratio = new_area / old_area)
    """
    h, w = img.shape[:2]

    # 1. Build rotation matrix around image center
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale)

    # 2. Compute new canvas size so all four corners fit
    new_w, new_h = _rotated_canvas_size(w, h, angle_deg, scale)

    # 3. Shift the rotation center to the new canvas center
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # 4. Apply the affine transform on the expanded canvas
    rotated = cv2.warpAffine(img, M, (new_w, new_h))

    expansion_ratio = (new_w * new_h) / (w * h)
    return rotated, expansion_ratio


# ═══════════════════════════════════════════════════════════════════════════════
# 2. build_pyramid() — Gaussian pyramid (repeated pyrDown)
# ═══════════════════════════════════════════════════════════════════════════════
#
# pyrDown: Gaussian blur → discard even rows/cols → size halves.
# Repeat until width or height < 64.  Return stacked list from coarse to fine.


def build_pyramid(img: np.ndarray, min_size: int = 64) -> list[np.ndarray]:
    """Build a Gaussian pyramid, stopping when width or height < min_size.

    Args:
        img:      input image
        min_size: stop threshold for the smaller dimension

    Returns:
        List of pyramid levels [original, L1, L2, ..., smallest]
    """
    pyramid = [img]
    while min(img.shape[0], img.shape[1]) // 2 >= min_size:
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


# ═══════════════════════════════════════════════════════════════════════════════
# 3. stitch_horizontal() — concatenate images into one wide strip
# ═══════════════════════════════════════════════════════════════════════════════
#
# All images must have the same height.  Shorter images are padded with black
# at the bottom to match the tallest one.  Labels are drawn in the top-left
# corner of each cell.


def stitch_horizontal(
    images: list[np.ndarray],
    labels: list[str],
    cell_width: int,
) -> np.ndarray:
    """Horizontally concatenate images with labels, padding shorter ones.

    Args:
        images:    list of BGR images (may differ in height and width)
        labels:    per-image label strings (e.g. angle or level name)
        cell_width: force each cell to this width (smaller images get right-padded)

    Returns:
        Single BGR image (tallest height × len(images) * cell_width)
    """
    # Normalise to 3-channel BGR so np.hstack works on grayscale inputs
    bgr_images = []
    for img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        bgr_images.append(img)

    tallest = max(img.shape[0] for img in bgr_images)

    cells = []
    for img, label in zip(bgr_images, labels):
        h, w = img.shape[:2]

        # Pad height: shorter images get black bars at the bottom
        if h < tallest:
            img = cv2.copyMakeBorder(
                img, 0, tallest - h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        # Pad width to ensure every cell is exactly cell_width
        if w < cell_width:
            img = cv2.copyMakeBorder(
                img, 0, 0, 0, cell_width - w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        # Draw label at top-left
        cv2.putText(img, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cells.append(img)

    return np.hstack(cells)


# ═══════════════════════════════════════════════════════════════════════════════


def build_rotation_collage(img: np.ndarray, step: int = 30) -> np.ndarray:
    """Rotate img every `step` degrees (0..330), stitch into a row.

    Args:
        img:  input image
        step: angle increment (default 30° → 12 frames)

    Returns:
        Horizontally stitched collage with angle labels
    """

    cells = []
    labels = []

    for angle in range(0, 360, step):
        rotated, _ = safe_rotate(img, angle)
        cells.append(rotated)
        labels.append(f"{angle}°")

    cell_width = max(c.shape[1] for c in cells)
    return stitch_horizontal(cells, labels, cell_width=cell_width)


# ═══════════════════════════════════════════════════════════════════════════════


def build_pyramid_collage(img: np.ndarray) -> np.ndarray:
    """Build Gaussian pyramid, upscale each level to original size, stitch.

    Each pyrDown level is half the resolution of the previous.  We resize every
    level back to the original dimensions so the progressive blur is visible —
    the classic textbook pyramid illustration.

    Args:
        img: input image

    Returns:
        Horizontally stitched pyramid strip with level labels
    """
    pyramid = build_pyramid(img)
    h0, w0 = pyramid[0].shape[:2]

    labels = []
    scaled = []
    for i, level in enumerate(pyramid):
        labels.append(f"L{i} ({level.shape[1]}×{level.shape[0]})")
        scaled.append(cv2.resize(level, (w0, h0), interpolation=cv2.INTER_CUBIC))

    return stitch_horizontal(scaled, labels, cell_width=w0)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. print_rotation_stats() — terminal summary of canvas expansion per angle
# ═══════════════════════════════════════════════════════════════════════════════


def print_rotation_stats(orig_w: int, orig_h: int, angles: list[int]) -> None:
    """Print a table of canvas expansion ratio per angle (pre-computed, no rotate)."""
    print(f"\n{'Angle':>6}  {'Canvas':>11} {'Expansion':>10}")
    print("-" * 31)
    for a in angles:
        new_w, new_h = _rotated_canvas_size(orig_w, orig_h, a)
        ratio = (new_w * new_h) / (orig_w * orig_h)
        print(f"{a:>5}°  {new_w:>4}×{new_h:<4}  {ratio:>9.2f}×")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. main() — pipeline orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Rotation collage + pyramid collage → save two report images."""
    # 1. Load image
    img = cv2.imread(str(DEFAULT_IMG))
    if img is None:
        print(f"Error: cannot read {DEFAULT_IMG}")
        sys.exit(1)
    h, w = img.shape[:2]
    print(f"Input: {DEFAULT_IMG.name}  {w}×{h}\n")

    # 2. Rotation collage
    print("Building rotation collage (0°–330°, step 30°)...")
    rotation_strip = build_rotation_collage(img, step=30)
    cv2.imwrite(str(OUTPUT_ROTATION), rotation_strip)
    print(
        f"  Saved → {OUTPUT_ROTATION.name}  ({rotation_strip.shape[0]}×{rotation_strip.shape[1]})"
    )

    # 3. Rotation stats
    print_rotation_stats(w, h, list(range(0, 360, 30)))

    # 4. Pyramid collage
    print("\nBuilding Gaussian pyramid...")
    pyramid_strip = build_pyramid_collage(img)
    cv2.imwrite(str(OUTPUT_PYRAMID), pyramid_strip)
    print(f"  Saved → {OUTPUT_PYRAMID.name}  ({pyramid_strip.shape[0]}×{pyramid_strip.shape[1]})")


if __name__ == "__main__":
    main()
