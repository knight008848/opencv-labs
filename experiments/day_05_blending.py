"""
Day 05: Image Blending — addWeighted & absdiff
Date: 2026-06-16
Goal: Blend two images at varying alpha levels, compute difference maps,
      and generate a multi-panel report figure.
Runtime: < 3 s
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULT_DIR = Path(__file__).resolve().parent / "results"
IMG_A_PATH = RESULT_DIR / "sample_f0107.jpg"
IMG_B_PATH = RESULT_DIR / "sample_f0576.jpg"
OUTPUT_PATH = RESULT_DIR / "day05_blending_report.jpg"

# ── Alpha sweep (headless — no Trackbar) ─────────────────────────────────────
# Instead of an interactive slider, we iterate through these alpha values
# and save every step as a subplot panel.
ALPHA_STEPS = np.linspace(0.0, 1.0, num=11)  # 0.0, 0.1, 0.2, ..., 1.0


# ── blend_images() ─────────────────────────────────────────────────────────────
# Alpha-blend two images with cv2.addWeighted.
#   src1 * α  +  src2 * β  +  γ,   α + β = 1,  γ = 0
#   alpha=0 → all img_a    alpha=1 → all img_b


def blend_images(img_a: np.ndarray, img_b: np.ndarray, alpha: float) -> np.ndarray:
    """Alpha-blend two images. alpha=0 → all img_a, alpha=1 → all img_b."""
    return cv2.addWeighted(img_a, alpha, img_b, 1 - alpha, 0)


# ── difference_map() ──────────────────────────────────────────────────────────
# Compute absdiff between two images, threshold noise, return change ratio.
#   diff_raw    — per-pixel absolute difference (BGR)
#   diff_thresh — binary mask (gray), 255 where change > threshold
#   change_ratio — fraction of pixels above threshold


def difference_map(
    img_a: np.ndarray,
    img_b: np.ndarray,
    threshold: int = 30,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute absdiff between two images, threshold, and return the change ratio."""
    diff_raw = cv2.absdiff(img_a, img_b)

    diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
    
    _, diff_thresh = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    total_pixels = diff_thresh.size
    changed_pixels = np.count_nonzero(diff_thresh)
    change_ratio = float(changed_pixels / total_pixels)

    return diff_raw, diff_thresh, change_ratio


# ── build_blending_report() ───────────────────────────────────────────────────
# Multi-panel report figure using matplotlib.gridspec.GridSpec (2 rows × 3 cols).
#   Row 1 — 3 key alpha steps (0.0, 0.5, 1.0)
#   Row 2 — img_a | diff JET heatmap | img_b


def build_blending_report(
    img_a: np.ndarray,
    img_b: np.ndarray,
    diff_thresh: np.ndarray,
    change_ratio: float,
) -> plt.Figure:
    """Generate a multi-panel report figure for the blending experiment."""
    from matplotlib.gridspec import GridSpec

    KEY_ALPHAS = [0.0, 0.5, 1.0]

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 3, figure=fig)

    # ── Row 1: alpha blending progression (3 key steps) ──────────────
    for i, alpha in enumerate(KEY_ALPHAS):
        blended = cv2.addWeighted(img_a, 1.0 - alpha, img_b, alpha, 0.0)
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        ax.set_title(f"α = {alpha:.2f}", fontsize=11)
        ax.axis("off")

    # ── Row 2: img_a | diff heatmap | img_b ──────────────────────────
    diff_raw = cv2.absdiff(img_a, img_b)
    diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
    diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

    ax_a = fig.add_subplot(gs[1, 0])
    ax_a.imshow(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
    ax_a.set_title("Image A", fontsize=11)
    ax_a.axis("off")

    ax_diff = fig.add_subplot(gs[1, 1])
    ax_diff.imshow(cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB))
    ax_diff.set_title(f"Difference (JET)  |  change = {change_ratio:.3f}", fontsize=11)
    ax_diff.axis("off")

    ax_b = fig.add_subplot(gs[1, 2])
    ax_b.imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
    ax_b.set_title("Image B", fontsize=11)
    ax_b.axis("off")

    fig.tight_layout()
    return fig


# ── main() ────────────────────────────────────────────────────────────────────
# Pipeline: load → resize (if needed) → alpha sweep → difference map → report


def main() -> None:
    # ── 1. Load images ─────────────────────────────────────────────────
    img_a = cv2.imread(str(IMG_A_PATH))
    if img_a is None:
        print(f"Error: failed to read image at {IMG_A_PATH}")
        sys.exit(1)
    print(f"shape: {img_a.shape}")

    img_b = cv2.imread(str(IMG_B_PATH))
    if img_b is None:
        print(f"Error: failed to read image at {IMG_B_PATH}")
        sys.exit(1)
    print(f"shape: {img_b.shape}")

    # ── 2. Resize if needed ────────────────────────────────────────────
    # Unify to the smaller dimension to avoid upscaling artefacts.
    if img_a.shape != img_b.shape:
        h = min(img_a.shape[0], img_b.shape[0])
        w = min(img_a.shape[1], img_b.shape[1])
        print(f"resizing both images to {h}×{w}")
        img_a = cv2.resize(img_a, (w, h))
        img_b = cv2.resize(img_b, (w, h))

    # ── 3. Alpha sweep ─────────────────────────────────────────────────
    for alpha in ALPHA_STEPS:
        blended = blend_images(img_a, img_b, alpha)
        print(f"alpha={alpha:.2f}, blended.shape={blended.shape}")

    # ── 4. Difference map ──────────────────────────────────────────────
    diff_raw, diff_thresh, change_ratio = difference_map(img_a, img_b)
    print("-" * 54)
    print(f"  Diff  |  changed pixels: {np.count_nonzero(diff_thresh)} / {diff_thresh.size}")
    print(f"        |  change ratio:   {change_ratio:.4f}  ({change_ratio * 100:.2f}%)")
    print("-" * 54)

    # ── 5. Build & save report ─────────────────────────────────────────
    fig = build_blending_report(img_a, img_b, diff_thresh, change_ratio)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Report saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
