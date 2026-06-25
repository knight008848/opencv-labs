# Day 13 / 2026-06-25
# Goal: Pipeline combo — denoise → edge → contour → perspective correction
# Runtime: ~1.5h
#
# Pipeline:
#   1. Median filter  (salt-pepper removal)
#   2. Gaussian blur  (further smoothing for clean edges)
#   3. Canny          (extract document boundary)
#   4. findContours + approxPolyDP  (locate the largest quadrilateral)
#   5. warpPerspective  (flat document correction)
#   6. Side-by-side report  (gray + edge, before/after)
#
# Headless: matplotlib savefig for all output, no cv2.imshow.

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Support HEIC input via pillow-heif (optional dependency)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import any_image_reader  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_IMAGE = PROJECT_ROOT / "data" / "raw" / "your_document.jpg"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# --- Pipeline parameters (tune these on your test image) ---
MEDIAN_KSIZE = 5          # kernel size for median filter (odd)
GAUSSIAN_KSIZE = (5, 5)   # kernel size for Gaussian blur (odd, odd)
CANNY_LOW = 50            # Canny low threshold
CANNY_HIGH = 150          # Canny high threshold

# --- Output document size (A4-like proportions) ---
OUTPUT_W = 2100
OUTPUT_H = 2970

# --- Salt-pepper noise for synthetic test image ---
ADD_NOISE = False         # set True to inject noise (for synthetic test images only)
NOISE_PROB = 0.02         # 2% salt-pepper density


# ---------------------------------------------------------------------------
# Pipeline step 1: Add noise (only for the synthetic test image)
# ---------------------------------------------------------------------------


def add_salt_pepper_noise(img: np.ndarray, prob: float = NOISE_PROB) -> np.ndarray:
    """Add salt-and-pepper noise to an image.

    Each pixel has a probability *prob* of being replaced:
      - half the noisy pixels → white (salt)
      - half the noisy pixels → black (pepper)

    This simulates the kind of noise a cheap document scanner or
    low-light camera might produce.  The Day 13 pipeline must clean
    this noise *before* running Canny, otherwise the edge map will
    be dominated by false edges from noise.

    Args:
        img:  BGR image, uint8.
        prob: Total noise probability per pixel (default 0.02 = 2%).

    Returns:
        Noisy BGR image (same shape, dtype).
    """
    noisy = img.copy()
    h, w = noisy.shape[:2]
    mask = np.random.random((h, w))
    salt_mask = mask < prob / 2
    pepper_mask = (mask >= prob / 2) & (mask < prob)
    noisy[salt_mask] = 255
    noisy[pepper_mask] = 0
    return noisy


# ---------------------------------------------------------------------------
# Pipeline step 2-3: Noise removal (median + Gaussian)
# ---------------------------------------------------------------------------


def denoise_median(img: np.ndarray, ksize: int = MEDIAN_KSIZE) -> np.ndarray:
    """Remove salt-and-pepper noise with a median filter.

    Median filter replaces each pixel with the median value in its
    neighbourhood — excellent at removing isolated black/white dots
    while preserving edges better than mean blur.

    Why median for salt-pepper:  a single white pixel in a dark area
    won't pull the median away from the surrounding dark values, so
    the dot disappears without smearing the edge.

    Args:
        img:  BGR image, uint8.
        ksize: Kernel size (odd, default 5).

    Returns:
        Filtered BGR image.
    """
    return cv2.medianBlur(img, ksize)


def smooth_gaussian(gray: np.ndarray, ksize: tuple[int, int] = GAUSSIAN_KSIZE) -> np.ndarray:
    """Apply Gaussian blur to further smooth the image.

    After median removal, Gaussian blur attenuates remaining high-
    frequency noise so Canny doesn't hallucinate edges from it.

    Why *both* median and Gaussian?  Median kills impulse noise but
    doesn't smooth gradual texture; Gaussian smooths gradual texture
    but lets impulse noise leak through.  Together they cover both.

    Args:
        gray:  Grayscale image, uint8.
        ksize: Kernel size (default (5, 5)).

    Returns:
        Blurred grayscale image.
    """
    return cv2.GaussianBlur(gray, ksize, 0)


# ---------------------------------------------------------------------------
# Pipeline step 4: Canny edge detection
# ---------------------------------------------------------------------------


def extract_edges(
    blurred: np.ndarray,
    low: int = CANNY_LOW,
    high: int = CANNY_HIGH,
) -> np.ndarray:
    """Run Canny edge detection on a pre-blurred grayscale image.

    The caller is responsible for blurring — this function wraps
    only the Canny call and prints the edge-pixel density.

    Args:
        blurred:  Pre-blurred grayscale image.
        low:      Canny low threshold.
        high:     Canny high threshold.

    Returns:
        Binary edge map (uint8, 0 or 255).
    """
    edges = cv2.Canny(blurred, low, high)
    n_edges = cv2.countNonZero(edges)
    total_px = edges.shape[0] * edges.shape[1]
    print(f"[extract_edges] Canny({low},{high}): {n_edges:,} edge px ({n_edges/total_px*100:.1f}%)")
    return edges


# ---------------------------------------------------------------------------
# Pipeline step 5: Find the largest quadrilateral contour
# ---------------------------------------------------------------------------
# This is the key step — and the hardest one.  Strategy:
#   1. Find all contours from the edge map (RETR_EXTERNAL)
#   2. Approximate each contour with approxPolyDP
#   3. Filter for 4-vertex approximations
#   4. Pick the one with the largest area → that's the document boundary
#
# Common gotchas:
#   - approxPolyDP epsilon controls how aggressively to simplify.
#     Start with 0.02 * arcLength and adjust up/down.
#   - The document edge might not form a perfect closed contour if
#     Canny thresholds are too high or the background is cluttered.
#     In a real scan you'd add morphological close before step 4.
#   - contourArea returns area in pixels — the document should be
#     by far the largest quadrilateral in the frame.


def find_largest_quadrilateral(edges: np.ndarray) -> np.ndarray | None:
    """Find the largest 4-vertex contour from an edge map.

    Steps:
      1. cv2.findContours(edges, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
      2. For each contour:  approx = cv2.approxPolyDP(contour, epsilon, closed=True)
      3. Keep approx where len(approx) == 4
      4. Return the one with the largest cv2.contourArea

    Args:
        edges: Binary edge map (uint8, 0 or 255).

    Returns:
        float32 (4, 2) array of corner coordinates (TL→TR→BR→BL order),
        or None if no quadrilateral found.

    Raises:
        RuntimeError:  If no valid quadrilateral is found — the caller
                       should handle this and suggest re-tuning Canny.
    """
    # 1. Find all external contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[find_largest_quadrilateral] No contours found.")
        return None

    # 2-4. Approximate → filter 4-vertex → pick largest area
    candidates = []
    for c in contours:
        peri = cv2.arcLength(c, closed=True)
        epsilon = 0.02 * peri
        approx = cv2.approxPolyDP(c, epsilon, closed=True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            candidates.append((area, approx.reshape(4, 2).astype(np.float32)))

    if not candidates:
        print("[find_largest_quadrilateral] No quadrilateral contours found.")
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    print(f"[find_largest_quadrilateral] Found quad with area={best[0]:.0f} px")
    return best[1]


# ---------------------------------------------------------------------------
# Pipeline step 6: Perspective warp
# ---------------------------------------------------------------------------


def warp_document(
    img: np.ndarray,
    src_pts: np.ndarray,
    dst_size: tuple[int, int] = (OUTPUT_W, OUTPUT_H),
) -> np.ndarray:
    """Perspective-warp the detected document quadrilateral to a flat rectangle.

    This reuses the same logic as Day 10:  getPerspectiveTransform
    followed by warpPerspective.

    Args:
        img:     BGR source image.
        src_pts: float32 (4, 2) source corners in TL→TR→BR→BL order.
        dst_size: (width, height) of output rectangle.

    Returns:
        Warped BGR image (flat document).
    """
    dst_pts = np.float32([
        [0, 0],
        [dst_size[0], 0],
        [dst_size[0], dst_size[1]],
        [0, dst_size[1]],
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, dst_size, borderMode=cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def print_pipeline_summary(
    steps: list[tuple[str, float]],
    corners: np.ndarray | None,
) -> None:
    """Print a clean summary of pipeline execution.

    Args:
        steps:   List of (step_name, elapsed_seconds).
        corners: Found quadrilateral corners (or None).
    """
    print("\n[PIPELINE SUMMARY]")
    total = 0.0
    for name, t in steps:
        total += t
        print(f"  {name:35s} {t*1000:8.2f} ms")
    print(f"  {'TOTAL':35s} {total*1000:8.2f} ms")

    if corners is not None:
        print(f"\n[INFO] Document corners (TL→TR→BR→BL):")
        for i, pt in enumerate(corners):
            print(f"  pt{i}: ({pt[0]:.1f}, {pt[1]:.1f})")


def build_combo_report(
    original_gray: np.ndarray,
    noisy: np.ndarray,
    after_median: np.ndarray,
    edges: np.ndarray,
    warped_gray: np.ndarray,
    warped_edges: np.ndarray,
    corners: np.ndarray | None,
) -> plt.Figure:
    """Build a report figure showing each pipeline stage.

    Layout (2×3):
      Row 0: Noisy input | After denoise | Canny edges
      Row 1: Warped gray | Warped edges  | (optional text summary)

    If no quadrilateral was found, row 1 shows the failure state
    with a warning annotation instead.

    Args:
        original_gray: Source grayscale image.
        noisy:         After adding salt-pepper noise.
        after_median:  After median + Gaussian denoising.
        edges:         Canny edge map pre-warp.
        warped_gray:   Perspective-corrected gray image.
        warped_edges:  Canny edges of the corrected document.
        corners:       Detected quadrilateral (or None).

    Returns:
        matplotlib Figure ready for savefig.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Day 13 — Document Correction Pipeline", fontsize=14, fontweight="bold")

    titles = [
        "Original (Gray)",
        "After Median + Gaussian",
        "Edges + Quadrilateral",
        "Warped (Gray)",
        "Warped Edges",
        "Status",
    ]

    for i, ax in enumerate(axes.flat):
        ax.set_title(titles[i], fontsize=10)
        ax.axis("off")

    # Row 0
    axes[0, 0].imshow(original_gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].imshow(cv2.cvtColor(after_median, cv2.COLOR_BGR2RGB))
    # Build edge+contour overlay: edges as gray background, then draw quad on top
    if edges is not None:
        edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edge_rgb = cv2.cvtColor(edge_rgb, cv2.COLOR_BGR2RGB)
        if corners is not None:
            cv2.polylines(edge_rgb, [corners.astype(np.int32)], True, (255, 0, 0), 3)
            for pt in corners:
                cv2.circle(edge_rgb, pt.astype(np.int32), 8, (0, 255, 0), -1)
        axes[0, 2].imshow(edge_rgb)

    # Row 1
    if corners is not None:
        axes[1, 0].imshow(warped_gray, cmap="gray", vmin=0, vmax=255)
        axes[1, 1].imshow(warped_edges, cmap="gray", vmin=0, vmax=255)
        axes[1, 2].text(
            0.5, 0.5,
            f"Quadrilateral found!\n{warped_gray.shape[1]}×{warped_gray.shape[0]}",
            ha="center", va="center", fontsize=12, transform=axes[1, 2].transAxes,
        )
    else:
        axes[1, 0].text(0.5, 0.5, "No quadrilateral\ndetected", ha="center", va="center",
                        fontsize=12, color="red", transform=axes[1, 0].transAxes)
        axes[1, 1].axis("off")
        axes[1, 2].text(
            0.5, 0.5,
            "Try:\n- Lower Canny thresholds\n- Larger median kernel\n- Morphological close",
            ha="center", va="center", fontsize=10, transform=axes[1, 2].transAxes,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the 6-step document correction pipeline end-to-end.

    Pipeline:
      1. Load (or generate) a tilted document image with salt-pepper noise
      2. Median filter → kill impulse noise
      3. Gaussian blur  → further smooth for clean Canny
      4. Canny          → extract binary edge map
      5. findContours + approxPolyDP  → find the largest quadrilateral
      6. warpPerspective  → flat document correction
      7. Build + save side-by-side report figure
    """
    import time

    steps_log: list[tuple[str, float]] = []

    # ---- Step 0: Load / generate image ----
    t0 = time.perf_counter()
    img_path = INPUT_IMAGE
    if not img_path.exists():
        print(f"[WARN] {img_path} not found. Generating a synthetic test image.")
        img = _make_test_document()
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), img)
        print(f"[INFO] Saved synthetic image → {img_path}")
    else:
        img = any_image_reader(img_path)
        if img is None:
            print(f"[ERROR] Could not load {img_path}")
            sys.exit(1)
    print(f"[INFO] Loaded image: {img.shape[1]}×{img.shape[0]}")
    original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- Inject salt-pepper noise (optional, for synthetic images) ----
    if ADD_NOISE:
        print(f"[INFO] Adding {NOISE_PROB*100:.0f}% salt-pepper noise ...")
        noisy = add_salt_pepper_noise(img)
        steps_log.append(("Add noise", time.perf_counter() - t0))
    else:
        noisy = img.copy()
        print("[INFO] Noise injection skipped (using real image).")

    # ---- Step 1: Median filter ----
    t1 = time.perf_counter()
    after_median = denoise_median(noisy, MEDIAN_KSIZE)
    steps_log.append(("Median filter", time.perf_counter() - t1))
    print(f"  Median filter: done")

    # ---- Step 2: Gaussian blur (on grayscale) ----
    t2 = time.perf_counter()
    denoised_gray = cv2.cvtColor(after_median, cv2.COLOR_BGR2GRAY)
    blurred_gray = smooth_gaussian(denoised_gray, GAUSSIAN_KSIZE)
    steps_log.append(("Gaussian blur", time.perf_counter() - t2))
    print(f"  Gaussian blur: done")

    # ---- Step 3: Canny ----
    t3 = time.perf_counter()
    edges = extract_edges(blurred_gray, CANNY_LOW, CANNY_HIGH)
    steps_log.append(("Canny edges", time.perf_counter() - t3))

    # ---- Step 4: Find largest quadrilateral ----
    t4 = time.perf_counter()
    corners = find_largest_quadrilateral(edges)
    steps_log.append(("Find quad", time.perf_counter() - t4))

    # ---- Step 5: Perspective warp ----
    warped_gray = np.zeros((1, 1), dtype=np.uint8)
    warped_edges = np.zeros((1, 1), dtype=np.uint8)
    if corners is not None:
        t5 = time.perf_counter()
        warped_bgr = warp_document(img, corners, (OUTPUT_W, OUTPUT_H))
        warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        # Re-run Canny on the warped image for comparison
        warped_blurred = smooth_gaussian(warped_gray, GAUSSIAN_KSIZE)
        warped_edges = extract_edges(warped_blurred, CANNY_LOW, CANNY_HIGH)
        steps_log.append(("Perspective warp", time.perf_counter() - t5))
    else:
        print("[ERROR] No quadrilateral detected — skipping warp. Tune Canny thresholds or add morphological close.")
        steps_log.append(("Perspective warp (skipped)", 0.0))

    # ---- Step 6: Build + save report ----
    fig = build_combo_report(
        original_gray, noisy, after_median, edges,
        warped_gray, warped_edges, corners,
    )
    out_dir = RESULTS_DIR
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "day_13_pipeline_report.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Report saved → {out_path}")

    # Print terminal summary
    print_pipeline_summary(steps_log, corners)


# ---------------------------------------------------------------------------
# Synthetic test image with controllable noise
# ---------------------------------------------------------------------------


def _make_test_document(height: int = 1200, width: int = 1600) -> np.ndarray:
    """Generate a synthetic tilted document with text content.

    Same layout as Day 10/12 — tilted document on a wood-like desk
    background.  This version does NOT add salt-pepper noise (that's
    done separately by add_salt_pepper_noise so you can toggle it).

    Returns:
        BGR image with a perspective-distorted document in the frame.
    """
    h, w = height, width
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = (60, 100, 180)

    # Document corners (tilted ~20°)
    src_corners = np.float32([[250, 170], [1350, 280], [1200, 1050], [380, 930]])

    doc_w, doc_h = 1400, 900
    flat = np.ones((doc_h, doc_w, 3), dtype=np.uint8) * 245

    cv2.putText(flat, "OPENCV LABS — Day 13", (60, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (0, 0, 0), 3)
    cv2.line(flat, (60, 90), (doc_w - 60, 90), (0, 0, 0), 2)

    overlay = flat.copy()
    cv2.putText(overlay, "CONFIDENTIAL",
                (doc_w // 2 - 280, doc_h // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (200, 200, 210), 6)
    flat = cv2.addWeighted(flat, 0.72, overlay, 0.28, 0)

    paragraphs = [
        "Day 13 combines everything from the last two weeks:",
        "median filtering (remove salt-pepper noise), Gaussian",
        "blur (smooth residual texture), Canny (clean edges),",
        "contour detection (find the document boundary), and",
        "perspective warp (flat-correct the tilted view).",
        "",
        "This is the same pipeline used in every mobile",
        "document scanner (CamScanner, Adobe Scan, etc.):",
        "    1. Denoise",
        "    2. Edge detection",
        "    3. Largest quadrangle",
        "    4. Perspective correction",
        "    5. Output as flat grayscale image",
    ]
    y = 140
    for line in paragraphs:
        if line:
            cv2.putText(flat, line, (60, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (50, 50, 50), 1)
            y += 28
        else:
            y += 14

    flat_corners = np.float32([[0, 0], [doc_w, 0], [doc_w, doc_h], [0, doc_h]])
    M_inv = cv2.getPerspectiveTransform(flat_corners, src_corners)
    warped_tex = cv2.warpPerspective(flat, M_inv, (w, h))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [src_corners.astype(np.int32)], 255)
    bg[mask == 255] = warped_tex[mask == 255]

    cv2.polylines(bg, [src_corners.astype(np.int32)], True, (0, 0, 0), 3)
    cv2.circle(bg, (250, 170), 20, (0, 0, 255), -1)
    cv2.putText(bg, "TL", (280, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return bg


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
