#!/usr/bin/env python3
"""Day 13 — Parameter tuning script for real photos.

Tries parameter combinations and saves the best warped result per image.
Headless-safe: all output via imwrite.

Usage:
    python experiments/day_13_tune.py data/raw/IMG_0701.png
    python experiments/day_13_tune.py data/raw/ --batch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "tuning"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MAX_DIM = 1500  # downscale so longest edge ≤ this

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"})


# ---------------------------------------------------------------------------
def load_and_downscale(path: Path) -> tuple[np.ndarray, float]:
    """Load image and downscale so longest edge ≤ MAX_DIM."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot load {path}")
    h, w = img.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img, scale


def _order_pts(pts: np.ndarray) -> np.ndarray:
    """Order 4 points TL→TR→BR→BL."""
    sorted_y = pts[np.argsort(pts[:, 1])]
    top2, bot2 = sorted_y[:2], sorted_y[2:]
    tl, tr = top2[np.argsort(top2[:, 0])]
    bl, br = bot2[np.argsort(bot2[:, 0])]
    return np.float32([tl, tr, br, bl])


def _quad_not_degenerate(pts: np.ndarray, min_dist: float = 30.0) -> bool:
    """Check that all 4 corners are sufficiently separated."""
    from itertools import combinations

    for i, j in combinations(range(4), 2):
        if np.linalg.norm(pts[i] - pts[j]) < min_dist:
            return False
    return True


def find_best_quad(
    img: np.ndarray,
    median_ksize: int,
    gauss_ksize: tuple[int, int],
    canny_low: int,
    canny_high: int,
    morph_close: bool = True,
    morph_ksize: int = 5,
) -> tuple[np.ndarray | None, str]:
    """Run edge pipeline; return (ordered_corners, method).

    Uses a two-stage strategy:
      1. approxPolyDP with adaptive epsilon — best for intact docs
      2. minAreaRect on largest contour — best for partial/damaged docs
    """
    # --- shared pre-processing ---
    med = cv2.medianBlur(img, median_ksize)
    gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, gauss_ksize, 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    if morph_close:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "no_contours"

    # --- Stage 1: approxPolyDP ---
    quads = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        for eps_factor in [0.02, 0.05, 0.08, 0.10, 0.15]:
            approx = cv2.approxPolyDP(c, eps_factor * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                if _quad_not_degenerate(pts):
                    quads.append((cv2.contourArea(pts), pts))
                break
    if quads:
        best = max(quads, key=lambda x: x[0])
        return _order_pts(best[1]), "approxPolyDP"

    # --- Stage 2: minAreaRect (for partial/damaged documents) ---
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(np.float32)
    if _quad_not_degenerate(box):
        return _order_pts(box), "minAreaRect"

    return None, "degenerate"


def warp_document(
    img: np.ndarray, src_pts: np.ndarray, dst_w: int = 2100, dst_h: int = 2970
) -> np.ndarray:
    """Perspective-warp to flat rectangle."""
    dst_pts = np.float32([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (dst_w, dst_h), borderMode=cv2.BORDER_CONSTANT)


def build_report(
    original_gray: np.ndarray,
    edges: np.ndarray,
    corners: np.ndarray | None,
    warped: np.ndarray | None,
    params: str,
) -> plt.Figure:
    """4-panel report: original | edges | original+corners | warped."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 18))
    axes[0, 0].imshow(original_gray, cmap="gray")
    axes[0, 0].set_title("Original (grayscale)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(edges, cmap="gray")
    axes[0, 1].set_title(f"Edges + morph close\n{params}")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(original_gray, cmap="gray")
    if corners is not None:
        corners_i = corners.astype(np.int32)
        cv2.polylines(original_gray, [corners_i], True, 255, 3)
        for i, (x, y) in enumerate(corners_i):
            axes[1, 0].text(x + 5, y - 5, f"P{i}", color="red", fontsize=8)
    axes[1, 0].set_title("Detected quadrilateral")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(warped if warped is not None else np.zeros((1, 1)), cmap="gray")
    axes[1, 1].set_title("Warped (corrected)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
PARAM_GRID = [
    # (median_ksize, gauss_ksize, canny_low, canny_high)
    # Light blur + low thresholds (good for high-contrast docs)
    (3, (5, 5), 20, 60),
    (3, (9, 9), 20, 60),
    (3, (9, 9), 30, 90),
    (5, (5, 5), 20, 60),
    (5, (9, 9), 20, 60),
    (5, (9, 9), 30, 90),
    # Stronger blur for textured backgrounds
    (5, (13, 13), 20, 60),
    (5, (13, 13), 30, 90),
    (7, (9, 9), 30, 90),
    (7, (13, 13), 30, 90),
    # Aggressive: very low thresholds for low-contrast docs
    (3, (5, 5), 10, 40),
    (3, (9, 9), 10, 40),
    (5, (9, 9), 10, 40),
    (5, (13, 13), 10, 40),
    (5, (17, 17), 10, 40),
    (5, (17, 17), 15, 50),
    (5, (21, 21), 15, 50),
    (7, (17, 17), 15, 50),
]


def tune_single(img_path: Path) -> dict:
    """Try parameter combinations on one image, save best result."""
    stem = img_path.stem
    print(f"\n{'=' * 70}")
    print(f"Tuning: {img_path.name}")
    print(f"{'=' * 70}")

    img_full = cv2.imread(str(img_path))
    img, scale = load_and_downscale(img_path)
    h, w = img.shape[:2]
    img_area = h * w
    print(f"  Resolution: {img_full.shape[1]}×{img_full.shape[0]} → scaled {w}×{h}")

    best_area_pct = 0
    best_result = None  # (quad, warped_full_res, edges, params_str)

    for mk, gk, cl, ch in PARAM_GRID:
        t0 = time.perf_counter()
        quad, method = find_best_quad(img, mk, gk, cl, ch, morph_close=True, morph_ksize=5)
        elapsed = (time.perf_counter() - t0) * 1000

        if quad is None:
            continue
        area = cv2.contourArea(quad)
        area_pct = area / img_area * 100

        status = ""
        if area_pct < 10:
            status = "  ✗ too small"
        elif area_pct < 20:
            status = "  △ marginal"
        else:
            status = "  ✓"

        method_tag = f" [{method}]" if method == "minAreaRect" else ""
        print(
            f"  mk={mk} gk={str(gk):12s} canny=({cl:3d},{ch:3d}) → area={area_pct:5.1f}% {elapsed:5.0f}ms{status}{method_tag}"
        )

        if area_pct > best_area_pct:
            best_area_pct = area_pct
            # Warp at full resolution
            quad_full = quad / scale
            warped = warp_document(img_full, quad_full)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, edges = cv2.threshold(
                cv2.Canny(cv2.GaussianBlur(gray, gk, 0), cl, ch), 0, 255, cv2.THRESH_BINARY
            )
            params_str = f"median={mk} gauss={gk} canny=({cl},{ch}) morph_close=5 method={method}"
            best_result = (quad, warped, edges, params_str, area_pct)

    if best_result is None:
        print("  ✗ No quadrilateral found in any parameter combination!")
        return {"path": img_path, "success": False, "area_pct": 0}

    quad, warped, edges, params_str, area_pct = best_result
    print(f"\n  ★ Best: {params_str}  area={area_pct:.1f}%")

    # Save report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Draw quad on gray for visualization
    gray_viz = gray.copy()
    quad_scaled = (quad * 1).astype(np.int32)  # quad is already in scaled coords
    cv2.polylines(gray_viz, [quad_scaled], True, 255, 3)

    fig = build_report(gray, edges, quad, warped, params_str)
    report_path = RESULTS_DIR / f"tune_{stem}_report.png"
    fig.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Report → {report_path}")

    # Save warped
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    warp_path = PROCESSED_DIR / f"{stem}_corrected.png"
    cv2.imwrite(str(warp_path), warped)
    print(f"  Warped → {warp_path}")

    return {"path": img_path, "success": True, "area_pct": area_pct, "params": params_str}


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Day 13 — parameter tuning for real photos")
    parser.add_argument("input", nargs="?", type=Path, default=None)
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()

    if args.input is None:
        root = PROJECT_ROOT / "data" / "raw"
        paths = sorted(
            [
                p
                for p in root.iterdir()
                if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.stem.startswith("IMG_")
            ]
        )
    elif args.input.is_dir() or args.batch:
        root = args.input if args.input.is_dir() else PROJECT_ROOT / "data" / "raw"
        paths = sorted([p for p in root.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    else:
        paths = [args.input]

    if not paths:
        print("[ERROR] No images found.")
        sys.exit(1)

    print(f"Tuning {len(paths)} image(s)\nParameter grid: {len(PARAM_GRID)} combinations")

    results = []
    for i, p in enumerate(paths, 1):
        print(f"\n[{i}/{len(paths)}]")
        r = tune_single(p)
        results.append(r)

    # Summary
    ok = sum(1 for r in results if r["success"])
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {ok}/{len(results)} succeeded")
    for r in results:
        status = "✓" if r["success"] else "✗"
        area_s = f" {r.get('area_pct', 0):.1f}%  {r.get('params', '')}" if r["success"] else ""
        print(f"  {status} {r['path'].name:30s}{area_s}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
