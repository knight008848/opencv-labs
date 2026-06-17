"""
Day 06: HSV Color Filter — inRange + bitwise_and
Date: 2026-06-17
Goal: Build a color-based object filter with 5 presets (red/green/blue/yellow/white).
      For each preset: generate mask, extract colored regions, count objects,
      and produce a multi-panel report figure.
Runtime: < 5 s
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULT_DIR = Path(__file__).resolve().parent / "results"
IMG_PATH = RESULT_DIR / "Day06-00.png"  # workspace desk with colored objects
OUTPUT_PATH = RESULT_DIR / "day06_color_filter_report.jpg"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. hsv_presets() — define HSV range pairs for 5 target colors
# ═══════════════════════════════════════════════════════════════════════════════
# H ∈ [0, 179]  (OpenCV half-scale: standard 0-360 ÷ 2)
# S ∈ [0, 255]  (0 = gray, 255 = pure color)
# V ∈ [0, 255]  (0 = black, 255 = full brightness)
#
# Red hue straddles 0° → two segments [0-10] ∪ [160-179]
# Each color maps to a list of (lower, upper) pairs for flexible multi-range masks.


def hsv_presets() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Return 5 color presets, each mapping to 1+ HSV range pairs."""
    return {
        "red": [
            (np.array([0, 150, 120], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
            (np.array([160, 150, 120], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
        ],
        "green": [
            (np.array([40, 50, 50], dtype=np.uint8), np.array([80, 255, 255], dtype=np.uint8)),
        ],
        "blue": [
            (np.array([100, 50, 50], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8)),
        ],
        "yellow": [
            (np.array([20, 100, 100], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8)),
        ],
        "white": [
            (np.array([0, 0, 230], dtype=np.uint8), np.array([179, 20, 255], dtype=np.uint8)),
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. apply_color_filter() — HSV + inRange → combined binary mask
# ═══════════════════════════════════════════════════════════════════════════════
# Iterates all (lower, upper) pairs for a color and merges them via bitwise_or.
# Caller is responsible for BGR→HSV conversion (done once, reused 5×).


def apply_color_filter(
    hsv: np.ndarray, bounds: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Apply inRange for each bound pair on a pre-computed HSV image, combine with OR."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in bounds:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# 3. extract_objects() — keep only mask-white pixels from the original image
# ═══════════════════════════════════════════════════════════════════════════════
# bitwise_and(src, src, mask=mask): where mask == 0 → pixel becomes (0,0,0).


def extract_objects(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return img with only mask regions preserved; everything else black."""
    return cv2.bitwise_and(img, img, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. count_objects() — count connected regions (blobs) in a binary mask
# ═══════════════════════════════════════════════════════════════════════════════
# RETR_EXTERNAL: only outermost contours (ignore holes inside objects).
# CHAIN_APPROX_SIMPLE: compress straight-line segments → fewer points.
# min_area=500 filters out sensor noise and texture fragments.


def count_objects(mask: np.ndarray, min_area: int = 500) -> int:
    """Count connected regions in mask larger than min_area pixels."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    return sum(1 for a in areas if a >= min_area)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. build_filter_report() — multi-panel GridSpec figure: original / mask / extracted
# ═══════════════════════════════════════════════════════════════════════════════
# 5 rows (one per color) × 3 columns.
# BGR → RGB conversion needed because matplotlib expects RGB channel order.


def build_filter_report(
    img: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    counts: dict[str, int],
) -> plt.Figure:
    """Generate a 5-row × 3-col report showing original, mask, and extraction per color.

    Args:
        img:     original BGR image
        results: {color_name: {'mask': ..., 'extracted': ...}, ...}
        counts:  {color_name: object_count, ...}
    """
    from matplotlib.gridspec import GridSpec

    color_names = list(results.keys())
    n_colors = len(color_names)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(18, 3.5 * n_colors))
    gs = GridSpec(n_colors, 3, figure=fig)

    for i, name in enumerate(color_names):
        mask = results[name]["mask"]
        extracted = results[name]["extracted"]
        extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)
        count = counts[name]

        # Column 1: original
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(img_rgb)
        ax1.set_title(f"{name} — original")
        ax1.axis("off")

        # Column 2: mask
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(mask, cmap="gray")
        ax2.set_title(f"{name} — mask ({count} objects)")
        ax2.axis("off")

        # Column 3: extracted
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.imshow(extracted_rgb)
        ax3.set_title(f"{name} — extracted")
        ax3.axis("off")

    fig.suptitle("Day 06 — HSV Color Filter Report", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 6. print_summary() — per-color detection stats to terminal
# ═══════════════════════════════════════════════════════════════════════════════


def print_summary(
    counts: dict[str, int], results: dict[str, dict[str, np.ndarray]]
) -> None:
    """Print a terminal summary of detection results."""
    any_mask = next(iter(results.values()))["mask"]
    total_pixels = any_mask.size
    print("\n  Color    Objects    Mask Coverage")
    print("  ───────  ───────    ─────────────")
    for name in counts:
        mask = results[name]["mask"]
        nonzero = int(cv2.countNonZero(mask))
        coverage = nonzero / total_pixels * 100
        print(f"  {name:<7}  {counts[name]:>5}      {coverage:>6.1f}%  ({nonzero} px)")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. main() — pipeline orchestrator
# ═══════════════════════════════════════════════════════════════════════════════
# Load → presets → [filter → extract → count] × 5 → report figure → terminal summary


def main() -> None:
    """Run the full HSV color-filter pipeline and save a report figure."""
    # ── 1. Load image ───────────────────────────────────────────────────
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        print(f"Error: failed to read image at {IMG_PATH}")
        sys.exit(1)
    print(f"shape: {img.shape}")

    # ── 2. Get presets ──────────────────────────────────────────────────
    presets = hsv_presets()

    # ── 3. Process each color ───────────────────────────────────────────
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert once, reuse 5×
    results: dict[str, dict[str, np.ndarray]] = {}
    counts: dict[str, int] = {}
    for name, bounds in presets.items():
        mask = apply_color_filter(hsv, bounds)
        extracted = extract_objects(img, mask)
        count = count_objects(mask)
        results[name] = {"mask": mask, "extracted": extracted}
        counts[name] = count

    # ── 4. Report figure ────────────────────────────────────────────────
    fig = build_filter_report(img, results, counts)
    fig.savefig(str(OUTPUT_PATH), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Report saved to {OUTPUT_PATH}")

    # ── 5. Terminal summary ─────────────────────────────────────────────
    print_summary(counts, results)


if __name__ == "__main__":
    main()
