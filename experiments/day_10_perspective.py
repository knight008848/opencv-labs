# Day 10 / 2026-06-22
# Goal: Document perspective corrector — click 4 corners, warp to A4 ratio
# Runtime: ~1.5h
#
# Headless note: uses matplotlib for all display (no cv2.imshow).
# Interactive point selection via plt.ginput.

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

A4_RATIO = 210 / 297  # width / height
OUTPUT_W = 2100  # output pixel width (A4 width in mm × 10)
OUTPUT_H = int(OUTPUT_W / A4_RATIO)  # → 2970

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # opencv-labs/
INPUT_IMAGE = PROJECT_ROOT / "data" / "raw" / "your_document.jpg"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


# ---------------------------------------------------------------------------
# Coordinate selection
# ---------------------------------------------------------------------------


def pick_four_points(image: np.ndarray) -> np.ndarray:
    """Let the user pick 4 document corners via matplotlib.

    Displays the image and uses plt.ginput(n=4, timeout=0).
    Left-click to select, middle-click to undo last point.

    Order enforced: top-left → top-right → bottom-right → bottom-left.

    Returns:
        float32 (4, 2) array of (col, row) coordinates.
    """
    # HINT: Convert BGR→RGB for matplotlib, then call plt.ginput(4, timeout=0).
    # HINT: plt.ginput returns [(x,y), ...] — validate len == 4, else raise.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.title("Pick 4 corners of the document")
    plt.grid()
    points = plt.ginput(4, timeout=0, mouse_stop=None, mouse_pop=2)
    if len(points) != 4:
        raise ValueError("Please pick exactly 4 corners.")

    plt.close()
    return np.float32(points)


# ---------------------------------------------------------------------------
# Perspective warp
# ---------------------------------------------------------------------------


def compute_warp_matrix(src_pts: np.ndarray, dst_size: tuple[int, int]) -> np.ndarray:
    """Compute 3×3 perspective transform matrix.

    Args:
        src_pts: float32 (4, 2) — four source corners (TL→TR→BR→BL).
        dst_size: (width, height) of the output rectangle.

    Returns:
        3×3 float64 perspective matrix.
    """
    # HINT: dst_pts = four corners of output rect: (0,0), (w,0), (w,h), (0,h)
    # HINT: cv2.getPerspectiveTransform (NOT getAffineTransform)
    dst_pts = np.float32([[0, 0], [dst_size[0], 0], [dst_size[0], dst_size[1]], [0, dst_size[1]]])
    return cv2.getPerspectiveTransform(src_pts, dst_pts)


def apply_perspective_warp(
    image: np.ndarray, M: np.ndarray, dst_size: tuple[int, int]
) -> np.ndarray:
    """Apply the perspective transform to produce a flat document view.

    Args:
        image: BGR source image.
        M: 3×3 perspective matrix.
        dst_size: (width, height) of output.

    Returns:
        Warped BGR image.
    """
    # HINT: cv2.warpPerspective(image, M, dst_size,
    #        borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return cv2.warpPerspective(image, M, dst_size, borderMode=cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def build_comparison_figure(
    original: np.ndarray,
    src_pts: np.ndarray,
    warped: np.ndarray,
) -> plt.Figure:
    """Side-by-side figure: annotated original (with quad overlay) + warped result.

    Args:
        original: BGR source image.
        src_pts: float32 (4, 2) source corners.
        warped: BGR warped output.

    Returns:
        matplotlib Figure ready for savefig.
    """
    # HINT: 1×2 subplots. Left = original + red quad overlay (cv2.polylines
    # before converting to RGB). Right = warped result.
    # HINT: use fig.suptitle to show the output dimensions.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Left: original with red quad overlay
    overlay = original.copy()
    cv2.polylines(overlay, [src_pts.astype(np.int32)], True, (0, 0, 255), 3)
    ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original (clicked corners)")
    ax1.axis("off")

    # Right: warped flat document
    ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    ax2.set_title("Warped (flat document)")
    ax2.axis("off")

    fig.suptitle(f"Perspective Correction  —  {warped.shape[1]}×{warped.shape[0]}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Full pipeline: load → pick points → warp → save comparison."""

    # 1. Load image
    img_path = INPUT_IMAGE
    if not img_path.exists():
        # Fallback: use a built-in test pattern if no input image is available
        print(f"[WARN] {img_path} not found. Generating a synthetic test image.")
        img = _make_test_document()
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), img)
        print(f"[INFO] Saved synthetic image → {img_path}")
    else:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[ERROR] Could not load {img_path}")
            sys.exit(1)

    print(f"[INFO] Loaded image: {img.shape[1]}×{img.shape[0]}")

    # 2. Pick the four corners
    print("[INFO] Click the 4 corners in order: TL → TR → BR → BL")
    print("[INFO]   Left-click  = place point")
    print("[INFO]   Middle-click = undo last point")
    src_pts = pick_four_points(img)

    # 3. Compute warp & apply
    dst_size = (OUTPUT_W, OUTPUT_H)
    M = compute_warp_matrix(src_pts, dst_size)
    warped = apply_perspective_warp(img, M, dst_size)

    # 4. Save outputs
    out_dir = RESULTS_DIR
    out_dir.mkdir(exist_ok=True)

    # Comparison figure
    fig = build_comparison_figure(img, src_pts, warped)
    comp_path = out_dir / "day_10_warped_comparison.png"
    fig.savefig(comp_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Comparison saved → {comp_path}")

    # Standalone flat document (BGR→RGB for matplotlib imsave)
    flat_path = out_dir / "day_10_flat_document.png"
    plt.imsave(str(flat_path), cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    print(f"[INFO] Flat document saved → {flat_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Synthetic test image (for development without a real document photo)
# ---------------------------------------------------------------------------


def _make_test_document(height: int = 1200, width: int = 1600) -> np.ndarray:
    """Generate a synthetic tilted document on a desk-like background.

    So you can develop the pipeline before you have a real photo.
    """
    h, w = height, width
    # Desk background (wood-like gradient)
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = (60, 100, 180)  # warm brownish BGR

    # Document corners (tilted ≈ 20°, trapezoidal from perspective)
    src_corners = np.float32(
        [
            [250, 170],  # top-left
            [1350, 280],  # top-right
            [1200, 1050],  # bottom-right
            [380, 930],  # bottom-left
        ]
    )

    # ── Build a flat document texture with text & watermark ──
    doc_w, doc_h = 1400, 900
    flat = np.ones((doc_h, doc_w, 3), dtype=np.uint8) * 245  # off-white paper

    # --- header ---
    cv2.putText(
        flat,
        "OPENCV LABS — Day 10",
        (60, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 0, 0),
        3,
    )
    cv2.line(flat, (60, 90), (doc_w - 60, 90), (0, 0, 0), 2)

    # --- watermark (diagonal across the centre) ---
    overlay = flat.copy()
    cv2.putText(
        overlay,
        "CONFIDENTIAL",
        (doc_w // 2 - 280, doc_h // 2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        (200, 200, 210),
        6,
    )
    flat = cv2.addWeighted(flat, 0.72, overlay, 0.28, 0)

    # --- body text ---
    paragraphs = [
        "This document simulates a photograph of a piece of paper",
        "taken from an angle.  Perspective correction (Day 10) uses",
        "a 3x3 homography matrix to map four clicked corners onto a",
        "flat A4-proportioned rectangle — the same math that powers",
        "document scanners, AR marker tracking, and bird's-eye view",
        "transforms in autonomous driving.",
        "",
        "    TL               TR",
        "     o─────────────o",
        "     |             |",
        "     |    CLICK    |",
        "     |    FOUR     |",
        "     |   CORNERS   |",
        "     o─────────────o",
        "    BL               BR",
    ]
    y = 140
    for line in paragraphs:
        if line:
            cv2.putText(
                flat,
                line,
                (60, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (50, 50, 50),
                1,
            )
            y += 28
        else:
            y += 14  # blank-line spacing

    # --- reverse-warp the flat document onto the desk ---
    flat_corners = np.float32([[0, 0], [doc_w, 0], [doc_w, doc_h], [0, doc_h]])
    M_inv = cv2.getPerspectiveTransform(flat_corners, src_corners)
    warped_tex = cv2.warpPerspective(flat, M_inv, (w, h))

    # composite: only inside the document polygon
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [src_corners.astype(np.int32)], 255)
    bg[mask == 255] = warped_tex[mask == 255]

    # Draw document border
    cv2.polylines(bg, [src_corners.astype(np.int32)], True, (0, 0, 0), 3)

    # Add a corner marker so the user knows which order to click
    cv2.circle(bg, (250, 170), 20, (0, 0, 255), -1)
    cv2.putText(bg, "TL", (280, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return bg


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
