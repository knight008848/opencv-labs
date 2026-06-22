# Day 11 / 2026-06-22
# Goal: Filter comparison lab — 6 filters × timing × Canny edge retention
# Runtime: ~1.5h
#
# Headless: matplotlib for all display, no cv2.imshow.

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_IMAGE = PROJECT_ROOT / "data" / "raw" / "your_document.jpg"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

FILTER_SPECS = [
    ("Original", None, None),
    ("Gaussian 5×5", "gaussian", {"ksize": (5, 5), "sigmaX": 0}),
    ("Gaussian 15×15", "gaussian", {"ksize": (15, 15), "sigmaX": 0}),
    ("Median 5", "median", {"ksize": 5}),
    ("Bilateral", "bilateral", {"d": 9, "sigmaColor": 75, "sigmaSpace": 75}),
    ("Mean 5×5", "mean", {"ksize": (5, 5)}),
]

CANNY_LOW = 50
CANNY_HIGH = 150


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------


def apply_filter(img: np.ndarray, filter_type: str, params: dict | None) -> np.ndarray:
    """Apply a single filter to a BGR image.

    Args:
        img: BGR source image.
        filter_type: "gaussian" | "median" | "bilateral" | "mean"
        params: kwargs for the corresponding cv2 function.

    Returns:
        Filtered BGR image.
    """
    # HINT: dispatch dict → cv2.GaussianBlur / cv2.medianBlur /
    #       cv2.bilateralFilter / cv2.blur
    # HINT: bilateralFilter returns the image directly (no dst arg needed)
    if filter_type == "gaussian":
        return cv2.GaussianBlur(img, **params)
    elif filter_type == "median":
        return cv2.medianBlur(img, **params)
    elif filter_type == "bilateral":
        return cv2.bilateralFilter(img, **params)
    elif filter_type == "mean":
        return cv2.blur(img, **params)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


# ---------------------------------------------------------------------------
# Edge retention metric (Canny edge count)
# ---------------------------------------------------------------------------


def count_canny_edges(img: np.ndarray, low: int = CANNY_LOW, high: int = CANNY_HIGH) -> int:
    """Count edge pixels using Canny edge detection on the grayscale image.

    More edges retained = better edge preservation after filtering.
    Used to build the edge-retention bar chart.

    Args:
        img: BGR image.
        low, high: Canny thresholds.

    Returns:
        Integer count of edge pixels (cv2.countNonZero on the Canny result).
    """
    # HINT: cvtColor BGR→GRAY, then cv2.Canny, then cv2.countNonZero
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, low, high)
    return cv2.countNonZero(edges)


# ---------------------------------------------------------------------------
# Filter pipeline (timed)
# ---------------------------------------------------------------------------


def run_filter_benchmark(
    img: np.ndarray, specs: list[tuple]
) -> tuple[list[str], list[np.ndarray], list[float]]:
    """Apply every filter in specs, return images + elapsed times.

    Args:
        img: BGR source image.
        specs: list of (label, filter_type, params) tuples.

    Returns:
        images: list of BGR results (same order as specs).
        times_ms: list of elapsed milliseconds per filter.
    """
    labels = []
    images = []
    times_ms = []
    original_img = img.copy()
    for label, filter_type, params in specs:
        if filter_type is None:
            images.append(original_img)
            times_ms.append(0.0)
            labels.append(label)
        else:
            start_time = time.perf_counter()
            img = apply_filter(original_img, filter_type, params)
            end_time = time.perf_counter()
            labels.append(label)
            images.append(img)
            times_ms.append((end_time - start_time) * 1000)
    return labels, images, times_ms


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def build_filter_report(
    images: list[np.ndarray],
    labels: list[str],
    times_ms: list[float],
) -> plt.Figure:
    """Build a 3-row figure:

        Row 1-2:  2×3 grid of filtered results (BGR→RGB for display).
                   Each subplot has a title with the filter name.
        Row 3:     edge-retention horizontal bar chart (Canny count per filter).

    Args:
        images:  6 BGR images (original + 5 filtered).
        labels:  matching labels.
        times_ms: matching elapsed times.

    Returns:
        matplotlib Figure.
    """
    # HINT: fig = plt.figure with GridSpec(3, 3).
    #       Top 2 rows (indices 0-5): imshow each image.
    #       Bottom row (span all 3 cols): horizontal bar chart of
    #       count_canny_edges() per filter, sorted descending.
    # HINT: bar colours — use a muted palette so the chart doesn't scream.
    # HINT: annotate each bar with the edge count and the timing in ms.
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.15)

    # --- Top 2 rows: 2×3 image grid ---
    for i, (img_bgr, label) in enumerate(zip(images, labels)):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    # --- Bottom row: edge-retention bar chart ---
    ax_bar = fig.add_subplot(gs[2, :])
    edge_counts = [count_canny_edges(im) for im in images]
    bar_colours = ["#6c8cbf"] * len(labels)  # muted blue

    # Sort descending by edge count
    indexed = sorted(
        enumerate(zip(labels, edge_counts, times_ms)), key=lambda x: x[1][1], reverse=True
    )
    sorted_labels = [labels[i] for i, _ in indexed]
    sorted_counts = [edge_counts[i] for i, _ in indexed]
    sorted_times = [times_ms[i] for i, _ in indexed]

    bars = ax_bar.barh(
        sorted_labels, sorted_counts, color=bar_colours, edgecolor="#3a5a8c", height=0.6
    )
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Canny Edge Pixels", fontsize=11)
    ax_bar.set_title(
        "Edge Retention After Filtering  (Canny low=50  high=150)", fontsize=12, fontweight="bold"
    )

    # Annotate each bar
    for bar, count, t_ms in zip(bars, sorted_counts, sorted_times):
        ax_bar.text(
            bar.get_width() + max(sorted_counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,d} px  |  {t_ms:.1f} ms",
            va="center",
            fontsize=9,
            color="#333333",
        )

    fig.suptitle("Filter Comparison Lab — Day 11", fontsize=14, fontweight="bold", y=0.98)
    return fig


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Full pipeline: load → benchmark → report → save."""

    # 1. Load image
    img_path = INPUT_IMAGE
    if not img_path.exists():
        print(f"[WARN] {img_path} not found. Generating a synthetic test image.")
        img = _make_test_filter_image()
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), img)
        print(f"[INFO] Saved synthetic image → {img_path}")
    else:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[ERROR] Could not load {img_path}")
            sys.exit(1)

    print(f"[INFO] Loaded image: {img.shape[1]}×{img.shape[0]}")

    # 2. Run benchmark
    print("[INFO] Running filter benchmark ...")
    labels, images, times_ms = run_filter_benchmark(img, FILTER_SPECS)

    # 3. Print timing stats
    print("\n[INFO] Filter timing (ms):")
    ranked = sorted(zip(labels, times_ms), key=lambda x: x[1])
    for rank, (name, t) in enumerate(ranked, 1):
        marker = "← fastest" if rank == 1 else ("← slowest" if rank == len(ranked) else "")
        print(f"  {rank}. {name:20s} {t:7.2f} ms  {marker}")

    # 4. Save report figure
    fig = build_filter_report(images, labels, times_ms)
    out_dir = RESULTS_DIR
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "day_11_filter_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Report saved → {out_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Synthetic test image
# ---------------------------------------------------------------------------


def _make_test_filter_image(size: tuple[int, int] = (600, 800)) -> np.ndarray:
    """Generate an image with sharp edges, smooth gradients, and salt-and-
    pepper noise — purpose-built for filter comparison.

    Contains:
      - Coloured rectangles (sharp edges → test bilateral edge retention)
      - A smooth radial gradient (test Gaussian blur smoothness)
      - 2% salt-and-pepper noise (test median filter noise removal)
      - Thin text (test whether filters destroy fine detail)
    """
    h, w = size
    img = np.ones((h, w, 3), dtype=np.uint8) * 240  # light grey background

    # --- coloured rectangles (sharp edges) ---
    colours_bgr = [
        (0, 0, 255),  # red
        (0, 255, 0),  # green
        (255, 0, 0),  # blue
        (0, 255, 255),  # yellow
    ]
    rect_w, rect_h = 120, 100
    for i, color in enumerate(colours_bgr):
        x = 50 + i * 170
        y = 50
        cv2.rectangle(img, (x, y), (x + rect_w, y + rect_h), color, -1)

    # --- radial gradient (smooth tonal transitions) ---
    cx, cy = w // 2, h // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    gradient = (dist / dist.max() * 200).astype(np.uint8)
    # apply gradient only inside a circle, blended over existing pixels
    mask = (dist < 200).astype(np.uint8)
    for c in range(3):
        img[:, :, c] = np.where(
            mask, (img[:, :, c] * 0.3 + gradient * 0.7).astype(np.uint8), img[:, :, c]
        )

    # --- thin text (fine detail) ---
    cv2.putText(
        img, "Edge Preservation Test", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2
    )
    cv2.putText(
        img, "Salt & Pepper Noise →", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1
    )

    # --- salt-and-pepper noise (2%) ---
    np.random.seed(42)
    # salt (white)
    salt_mask = np.random.rand(h, w) < 0.01
    img[salt_mask] = (255, 255, 255)
    # pepper (black)
    pepper_mask = np.random.rand(h, w) < 0.01
    img[pepper_mask] = (0, 0, 0)

    return img


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
