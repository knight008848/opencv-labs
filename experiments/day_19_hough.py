"""
Day 19 / 2026-07-17 / Module 9: Hough Transform (Concepts A + B)
Goal: Detect lines (HoughLinesP) and circles (HoughCircles) in a synthetic
      geometric image. Compare parameter sensitivity.
Deliverable: annotated images + parameter sweep grids + terminal report.
Runtime: ~1 h

Headless note: Trackbar-based interactive tuning is replaced by threshold
sweeps (for loop + batch savefig). See CLAUDE.md for headless policy.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


# ==============================  Utility  ====================================


def load_image(path: str | Path) -> np.ndarray:
    """Load an image in BGR color. Raise if unreadable."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def preprocess(gray: np.ndarray) -> np.ndarray:
    """
    Blur + Canny edge detection.

    HINT: GaussianBlur reduces noise so Hough doesn't vote on stray
    edge pixels. Canny thresholds (50, 150) are a good starting point.
    """

    return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)


# ===========================  Line Detection  ================================


def detect_lines(
    edges: np.ndarray,
    threshold: int = 100,
    min_line_length: int = 30,
    max_line_gap: int = 10,
) -> np.ndarray | None:
    """
    Detect line segments via Probabilistic Hough Transform.

    Parameters
    ----------
    edges : uint8 binary edge map (0 or 255)
    threshold : minimum votes needed to accept a line candidate
    min_line_length : discard segments shorter than this (pixels)
    max_line_gap : connect segments whose gap is <= this (pixels)

    Returns
    -------
    lines : shape (N, 1, 4) or None  — each row is [x1, y1, x2, y2]

    HINT: Use cv2.HoughLinesP with rho=1, theta=np.pi/180.
    """
    return cv2.HoughLinesP(edges, 1, np.pi/180, threshold, min_line_length, max_line_gap)


def draw_lines(
    image: np.ndarray,
    lines: np.ndarray | None,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw line segments onto a copy of *image*.

    Each line in *lines* has format [x1, y1, x2, y2].  If *lines* is
    None, return the original image unmodified.

    HINT: cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    """
    result = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), color, thickness)

    return result


# ===========================  Circle Detection  ==============================


def detect_circles(
    gray: np.ndarray,
    dp: float = 1.0,
    min_dist: int = 50,
    param1: int = 100,
    param2: int = 30,
    min_radius: int = 10,
    max_radius: int = 200,
) -> np.ndarray | None:
    """
    Detect circles via Hough Gradient method.

    Parameters
    ----------
    gray : uint8 grayscale image
    dp : accumulator resolution (1 = same as image)
    min_dist : minimum distance between circle centres
    param1 : internal Canny high threshold
    param2 : accumulator threshold — lower = more (false) circles
    min_radius, max_radius : size filter

    Returns
    -------
    circles : shape (1, N, 3) or None  — each row is [cx, cy, radius]

    HINT: Use cv2.HoughCircles with cv2.HOUGH_GRADIENT.
    Note: input is GRAYSCALE (not edges), HoughCircles builds its own
    Canny internally.
    """
    return cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist, param1, param2, min_radius, max_radius)


def draw_circles(
    image: np.ndarray,
    circles: np.ndarray | None,
    color: tuple = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw circle outlines + centre dots onto a copy of *image*.

    Each circle is *[cx, cy, radius]*.  The centre is drawn as a small
    filled dot (radius=2) and the circumference as a circle outline.

    HINT: cv2.circle for outline (thickness=2), cv2.circle for centre dot (-1)
    HINT: circles[0] gives the array of (cx, cy, r) rows
    """
    result = image.copy()
    if circles is not None:
        for circle in circles[0]:
            cx, cy, r = circle
            cv2.circle(result, (cx, cy), r, color, thickness)
            cv2.circle(result, (cx, cy), 2, color, -1)

    return result


# =======================  Parameter Sweeps  ==================================


def sweep_line_threshold(
    edges: np.ndarray,
    thresholds: list[int],
    output_dir: Path,
) -> None:
    """
    Run HoughLinesP at multiple *threshold* values and save a 1×N grid.

    Each panel shows detected lines at a given threshold.  The panel
    title prints the threshold value and the resulting line count.

    HINT: plt.subplots(1, n, figsize=(4*n, 4)). For each threshold,
    call detect_lines -> draw_lines on the edge map (converted to BGR
    then RGB), annotate with count, save as day_19_line_sweep.png.
    """
    n = len(thresholds)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle("Day 19 — Hough Transform Detection", fontsize=14)

    for ax, threshold in zip(axes, thresholds):
        lines = detect_lines(edges, threshold)
        counts = len(lines) if lines is not None else 0
    
        # 在边缘图上画绿色直线
        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)   # 灰度 → BGR
        annotated = draw_lines(edge_bgr, lines, color=(0, 255, 0), thickness=1)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)  # BGR → RGB

        ax.imshow(annotated_rgb)
        ax.set_title(f"Threshold: {threshold} — {counts}")
        ax.axis("off")
    

    plt.tight_layout()
    fig.savefig(str(output_dir / "day_19_line_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved line sweep: {output_dir / 'day_19_line_sweep.png'}")


def sweep_circle_param2(
    gray: np.ndarray,
    param2_values: list[int],
    output_dir: Path,
) -> None:
    """
    Run HoughCircles at multiple *param2* values and save a 1×N grid.

    *param2* is the accumulator threshold — the single most important
    knob for controlling circle detection sensitivity.

    HINT: Same grid pattern as sweep_line_threshold. Use a white or
    semi-transparent background so circles are clearly visible.
    Annotate with param2 value and circle count.
    """
    n = len(param2_values)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle("Day 19 — Hough Transform Detection", fontsize=14)

    for ax, param2 in zip(axes, param2_values):
        circles = detect_circles(gray, param2=param2)
        counts = len(circles) if circles is not None else 0

        # 在灰度图上画蓝色圆
        bg = np.full_like(gray, 255, dtype=np.uint8)
        gray_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        annotated = draw_circles(gray_bgr, circles, color=(0, 255, 0), thickness=2)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)  # BGR → RGB

        ax.imshow(annotated_rgb)
        ax.set_title(f"Param2: {param2} — {counts}")
        ax.axis("off")


    plt.tight_layout()
    fig.savefig(str(output_dir / "day_19_circle_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved circle sweep: {output_dir / 'day_19_circle_sweep.png'}")


# =======================  Debug Grid  ========================================


def build_debug_grid(
    gray: np.ndarray,
    edges: np.ndarray,
    line_result: np.ndarray,
    circle_result: np.ndarray,
    combined: np.ndarray,
    line_count: int,
    circle_count: int,
    output_dir: Path,
) -> None:
    """
    Build a 2×2 summary grid: grayscale input, edge map, lines, circles.

    Panel layout:
        [0,0] Grayscale input
        [0,1] Canny edge map
        [1,0] Lines overlay (green)
        [1,1] Circles overlay (blue)

    HINT: plt.subplots(2, 2). Convert BGR -> RGB for matplotlib.
    Use imshow(..., cmap="gray") for grayscale and edge panels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Day 19 — Hough Transform Detection", fontsize=14)

    # Panel 1: Grayscale input
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title(f"1. Grayscale Input ({gray.shape[1]}×{gray.shape[0]})")
    axes[0, 0].axis("off")

    # Panel 2: Canny edge map
    axes[0, 1].imshow(edges, cmap="gray")
    edge_ratio = cv2.countNonZero(edges) / edges.size
    axes[0, 1].set_title(f"2. Canny Edges\n({cv2.countNonZero(edges)}/{edges.size} px, {edge_ratio:.1%})")
    axes[0, 1].axis("off")

    # Panel 3: Lines overlay (BGR -> RGB)
    line_rgb = cv2.cvtColor(line_result, cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(line_rgb)
    axes[1, 0].set_title(f"3. Hough Lines ({line_count} segments)")
    axes[1, 0].axis("off")

    # Panel 4: Circles overlay (BGR -> RGB)
    circle_rgb = cv2.cvtColor(circle_result, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(circle_rgb)
    axes[1, 1].set_title(f"4. Hough Circles ({circle_count} circles)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    path = output_dir / "day_19_debug.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved debug grid: {path}")


# ==============================  Main  =======================================


def main():
    """
    Pipeline:
      1. Load type_test.png (resize if > 1200px)
      2. Preprocess: gray -> blur -> Canny
      3. Detect lines + circles with sensible defaults
      4. Draw overlay images (lines only, circles only, combined)
      5. Run parameter sweeps for threshold and param2
      6. Build debug grid
      7. Print terminal summary
    """
    # --- Setup output directory ---
    output_dir = PROJECT_DIR / "data" / "processed" / "day_19"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load test image ---
    image_path = PROJECT_DIR / "data" / "raw" / "type_test.png"
    print(f"[1/6] Loading {image_path.name}...")
    image_bgr = load_image(image_path)

       # --- Preprocess ---
    print("[2/6] Converting to grayscale and detecting edges...")
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = preprocess(image_gray)
    edge_ratio = cv2.countNonZero(edges) / edges.size

    # --- Line detection ---
    print("[3/6] Running Hough line detection...")
    line_result = detect_lines(edges)
    line_count = len(line_result) if line_result is not None else 0
    line_image = draw_lines(image_bgr.copy(), line_result, color=(0, 255, 0), thickness=1)

    # --- Circle detection ---
    print("[4/6] Running Hough circle detection...")
    circle_result = detect_circles(image_gray)
    circle_count = len(circle_result) if circle_result is not None else 0
    circle_image = draw_circles(image_bgr.copy(), circle_result, color=(0, 255, 0), thickness=2)

    # --- Combined overlay ---
    combined = image_bgr.copy()
    combined = draw_lines(combined, line_result, color=(0, 255, 0), thickness=1)
    combined = draw_circles(combined, circle_result, color=(0, 255, 0), thickness=2)
    cv2.imwrite(output_dir / "day_19_combined.jpg", combined)

    # --- Parameter sweeps ---
    print("[5/6] Running parameter sweeps...")
    sweep_line_threshold(image_gray, edges, line_result, line_image, output_dir)
    sweep_circle_param2(image_gray, edges, circle_result, circle_image, output_dir)

    # --- Debug grid ---
    print("[6/6] Building debug visualization...")
    build_debug_grid(
        image_gray,
        edges,
        line_image,
        circle_image,
        combined,
        line_count,
        circle_count,
        output_dir,
    )

    # --- Terminal summary ---
    print(f"\n{'='*50}")
    print(f"  Day 19 Hough Detection — Summary")
    print(f"{'='*50}")
    # TODO: print image size, edge ratio, line count, circle count
    print(f"{'='*50}")
    print(f"\nView results in {output_dir}/")


if __name__ == "__main__":
    main()
