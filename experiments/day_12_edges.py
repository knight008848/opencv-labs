# Day 12 / 2026-06-23
# Goal: Edge detection lab — Sobel X/Y/combined, Laplacian, Canny threshold sweep, auto Canny
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

GAUSSIAN_BLUR_KSIZE = (5, 5)


# ---------------------------------------------------------------------------
# Edge detectors
# ---------------------------------------------------------------------------


def sobel_edges(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sobel X, Y, and combined magnitude gradients."""
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    return cv2.convertScaleAbs(sobel_x), cv2.convertScaleAbs(sobel_y), cv2.convertScaleAbs(sobel_combined)


def laplacian_edges(gray: np.ndarray) -> np.ndarray:
    """Compute Laplacian (2nd derivative) — intentionally no GaussianBlur first."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)


def canny_edges(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    """Canny edge detection with prior Gaussian blur."""
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KSIZE, 0)
    return cv2.Canny(blurred, low, high)


def auto_canny(gray: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Auto Canny using the median method: t1 = median*0.66, t2 = median*1.33."""
    median = np.median(gray)
    t1 = int(max(0, min(255, median * 0.66)))
    t2 = int(max(0, min(255, median * 1.33)))
    return canny_edges(gray, t1, t2), t1, t2


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_edge_benchmark(gray: np.ndarray) -> dict:
    """Run all 10 edge detectors, return results with timing."""
    labels = []
    images = []
    times_ms = []

    # ---- Row 1 detectors ----
    t0 = time.perf_counter()
    sx, sy, scomb = sobel_edges(gray)
    t_sobel = (time.perf_counter() - t0) * 1000

    labels.append("Sobel X")
    images.append(sx)
    times_ms.append(t_sobel)

    labels.append("Sobel Y")
    images.append(sy)
    times_ms.append(t_sobel)

    labels.append("Sobel Combined")
    images.append(scomb)
    times_ms.append(t_sobel)

    t0 = time.perf_counter()
    lap = laplacian_edges(gray)
    times_ms.append((time.perf_counter() - t0) * 1000)
    labels.append("Laplacian")
    images.append(lap)

    t0 = time.perf_counter()
    c50_150 = canny_edges(gray, 50, 150)
    t_c50_150 = (time.perf_counter() - t0) * 1000
    labels.append("Canny(50,150)")
    images.append(c50_150)
    times_ms.append(t_c50_150)

    # ---- Row 2 detectors ----
    t0 = time.perf_counter()
    c30_90 = canny_edges(gray, 30, 90)
    times_ms.append((time.perf_counter() - t0) * 1000)
    labels.append("Canny(30,90)")
    images.append(c30_90)

    # Canny(50,150) reused — same image, same timing
    labels.append("Canny(50,150)")
    images.append(c50_150)
    times_ms.append(t_c50_150)

    t0 = time.perf_counter()
    c100_200 = canny_edges(gray, 100, 200)
    times_ms.append((time.perf_counter() - t0) * 1000)
    labels.append("Canny(100,200)")
    images.append(c100_200)

    t0 = time.perf_counter()
    c150_250 = canny_edges(gray, 150, 250)
    times_ms.append((time.perf_counter() - t0) * 1000)
    labels.append("Canny(150,250)")
    images.append(c150_250)

    t0 = time.perf_counter()
    ac_edges, auto_t1, auto_t2 = auto_canny(gray)
    times_ms.append((time.perf_counter() - t0) * 1000)
    labels.append(f"Canny(auto: {auto_t1:.0f},{auto_t2:.0f})")
    images.append(ac_edges)

    return {
        "labels": labels,
        "images": images,
        "times_ms": times_ms,
        "auto_t1": auto_t1,
        "auto_t2": auto_t2,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def build_edge_report(
    labels: list[str],
    images: list[np.ndarray],
    times_ms: list[float],
) -> plt.Figure:
    """Build a 2x5 comparison figure with 10 subplots."""
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.1)

    for i, (label, edge_img, t_ms) in enumerate(zip(labels, images, times_ms)):
        row, col = divmod(i, 5)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(edge_img, cmap="gray")
        ax.set_title(f"{label}\n{t_ms:.2f} ms", fontsize=9)
        ax.axis("off")

    fig.suptitle("Edge Detection Lab — Day 12", fontsize=14, fontweight="bold", y=0.98)
    return fig


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Full pipeline: load -> detect -> report -> save."""
    # 1. Load image
    img_path = INPUT_IMAGE
    if not img_path.exists():
        print(f"[WARN] {img_path} not found. Generating a synthetic test image.")
        img = _make_test_document()
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), img)
        print(f"[INFO] Saved synthetic image -> {img_path}")
    else:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[ERROR] Could not load {img_path}")
            sys.exit(1)

    print(f"[INFO] Loaded image: {img.shape[1]}x{img.shape[0]}")

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"[INFO] Median gray value: {np.median(gray)}")

    # 3. Run all edge detectors
    results = run_edge_benchmark(gray)

    # 4. Print timing ranking (deduplicate reused detectors)
    seen = set()
    unique = []
    for name, t in zip(results["labels"], results["times_ms"]):
        if name not in seen:
            seen.add(name)
            unique.append((name, t))
    ranked = sorted(unique, key=lambda x: x[1])
    print("\n[INFO] Edge detector timing (ms):")
    for rank, (name, t) in enumerate(ranked, 1):
        marker = " <- fastest" if rank == 1 else (" <- slowest" if rank == len(ranked) else "")
        print(f"  {rank:2d}. {name:35s} {t:7.2f} ms{marker}")

    # 5. Print Canny edge pixel density
    print("\n[INFO] Canny edge pixel density:")
    seen_canny = set()
    for name, edges in zip(results["labels"], results["images"]):
        if "Canny" in name and name not in seen_canny:
            seen_canny.add(name)
            print(f"  {name:35s}: {cv2.countNonZero(edges):,} px")

    # 6. Save report figure
    fig = build_edge_report(results["labels"], results["images"], results["times_ms"])
    out_dir = RESULTS_DIR
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "day_12_edge_detection.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Report saved -> {out_path}")


# ---------------------------------------------------------------------------
# Synthetic test image
# ---------------------------------------------------------------------------


def _make_test_document(height: int = 1200, width: int = 1600) -> np.ndarray:
    """Generate a synthetic tilted document on a desk-like background."""
    h, w = height, width
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = (60, 100, 180)

    src_corners = np.float32([
        [250, 170], [1350, 280], [1200, 1050], [380, 930]
    ])

    doc_w, doc_h = 1400, 900
    flat = np.ones((doc_h, doc_w, 3), dtype=np.uint8) * 245

    cv2.putText(flat, "OPENCV LABS — Day 12", (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
    cv2.line(flat, (60, 90), (doc_w - 60, 90), (0, 0, 0), 2)

    overlay = flat.copy()
    cv2.putText(overlay, "CONFIDENTIAL", (doc_w // 2 - 280, doc_h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (200, 200, 210), 6)
    flat = cv2.addWeighted(flat, 0.72, overlay, 0.28, 0)

    paragraphs = [
        "Edge detection is the foundation of computer vision —",
        "before you can recognize an object, you must first find its",
        "boundaries.  Sobel (1st derivative) captures gradient",
        "direction and magnitude; Laplacian (2nd derivative) is fast",
        "but amplifies noise; Canny adds Gaussian blur, non-maximum",
        "suppression, and hysteresis thresholding for clean, single-",
        "pixel-wide edges that are the gold standard in industry.",
        "",
        "    Sobel X:          Canny thresholds:",
        "     -1  0 +1          low=30,90  (dense)",
        "     -2  0 +2          low=50,150 (balanced)",
        "     -1  0 +1          low=100,200 (moderate)",
        "                       low=150,250 (sparse)",
        "    Auto Canny: t = median * [0.66, 1.33]",
    ]
    y = 140
    for line in paragraphs:
        if line:
            cv2.putText(flat, line, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
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
