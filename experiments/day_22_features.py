"""
Day 22 / 2026-07-23 / Module 10: Feature Detection (ORB)
File: day_22_features.py
Goal: ORB keypoint analysis on a synthetic textured image —
      detect, visualize, density heatmap, and compare nfeatures settings.
No external file dependencies (uses synthetic checkerboard).
Deliverable: 3 visualization panels + terminal summary
Runtime: ~30 sec
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── Config ────────────────────────────

NCOMPARE_VALUES = [100, 500, 2000]  # nfeatures values to compare
HGRID, WGRID = 8, 8  # density heatmap grid divisions
CANVAS_W, CANVAS_H = 600, 400


# ──────────────────────── 1. Synthetic Image ────────────────────


def make_synthetic_texture(w: int = CANVAS_W, h: int = CANVAS_H) -> np.ndarray:
    """
    Create a grayscale image with checkerboard + shapes for rich ORB features.
    """
    block_size = 40
    rows, cols = np.indices((h, w))
    checkerboard = ((rows // block_size) % 2) ^ ((cols // block_size) % 2)
    checkerboard = checkerboard.astype(np.uint8) * 255

    # Add geometric shapes for extra corner features
    cv2.rectangle(checkerboard, (50, 50), (150, 150), 128, -1)  # gray rect
    cv2.circle(checkerboard, (400, 200), 60, 64, -1)  # dark circle
    pts = np.array([[500, 350], [420, 250], [580, 250]], dtype=np.int32)
    cv2.fillPoly(checkerboard, [pts], 192)  # light triangle

    return checkerboard.astype(np.uint8)


# ──────────────────── 2. ORB Detection ──────────────────────────


def detect_keypoints(
    gray: np.ndarray,
    nfeatures: int = 500,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """
    Run ORB detection on grayscale image.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    return orb.detectAndCompute(gray, None)


# ──────────────────── 3. Visualization ──────────────────────────


def draw_rich_keypoints(
    gray: np.ndarray,
    keypoints: list[cv2.KeyPoint],
) -> np.ndarray:
    """
    Draw keypoints with circles showing orientation and scale.
    """
    return cv2.drawKeypoints(
        gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


def compute_density_map(
    gray: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    h_grid: int = HGRID,
    w_grid: int = WGRID,
) -> np.ndarray:
    """
    Divide image into h_grid × w_grid cells, count keypoints per cell.
    Returns 2D array of shape (h_grid, w_grid) with counts.
    """
    h_step = gray.shape[0] // h_grid
    w_step = gray.shape[1] // w_grid
    density = np.zeros((h_grid, w_grid), dtype=np.int32)
    for kp in keypoints:
        cell = (int(kp.pt[1] // h_step), int(kp.pt[0] // w_step))
        density[cell] += 1
    return density


def draw_density_heatmap(
    gray: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """
    Overlay density heatmap (red=high, blue=low) on the original image.
    """
    density = (255 * density / np.max(density)).astype(np.uint8)
    density = cv2.applyColorMap(density, cv2.COLORMAP_JET)
    # Upscale density grid to match original image dimensions
    density = cv2.resize(density, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(gray, 0.5, density, 0.5, 0)


# ──────────────────── 4. Comparison Panel ───────────────────────


def build_comparison_panel(
    gray: np.ndarray,
    results: list[dict],
    output_path: Path,
) -> None:
    """
    Create a 2×N grid: row 0 = rich keypoints, row 1 = density heatmap.
    """
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, res in enumerate(results):
        nf = res["nfeatures"]
        kps = res["keypoints"]

        # Row 0: rich keypoints
        viz = draw_rich_keypoints(gray, kps)
        axes[0, i].imshow(viz, cmap="gray")
        axes[0, i].set_title(f"nfeatures={nf}  (found {len(kps)})")
        axes[0, i].axis("off")

        # Row 1: density heatmap
        heat = draw_density_heatmap(gray, res["density"])
        axes[1, i].imshow(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f"Density {HGRID}x{WGRID}")
        axes[1, i].axis("off")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(nfeatures: int, keypoints: list[cv2.KeyPoint]) -> None:
    """
    Print total count, average response, and average size for keypoints.
    """
    responses = [kp.response for kp in keypoints]
    sizes = [kp.size for kp in keypoints]
    print(
        f"  nfeatures={nfeatures:5d} → found {len(keypoints):5d} pts, "
        f"avg response={np.mean(responses):.2f}, avg size={np.mean(sizes):.1f}"
    )


def answer_question(nfeatures_results: list[dict]) -> str:
    """
    Based on the three nfeatures runs, answer: "Are more features always better?"
    """
    # Compute avg response for each nfeatures setting
    lines = []
    for res in nfeatures_results:
        nf = res["nfeatures"]
        kps = res["keypoints"]
        if len(kps) == 0:
            continue
        avg_resp = np.mean([kp.response for kp in kps])
        lines.append(f"    nfeatures={nf:5d} → avg response={avg_resp:.2f}")

    # Build answer based on trend
    msg = (
        "More features is NOT always better. "
        "When nfeatures is small, ORB selects only the strongest corners "
        "(highest response). As nfeatures grows, the extra keypoints "
        "have progressively lower response values, meaning they are less "
        "distinctive and more likely to cause false matches in the "
        "matching stage. The trade-off: enough features for robust matching "
        "vs. too many weak features that add noise."
    )
    return "\n".join(lines) + "\n\n" + msg


# ──────────────────────────── Main ──────────────────────────────


def main() -> None:
    """
    Pipeline:
      1. Generate synthetic textured image
      2. Detect ORB keypoints with nfeatures=100, 500, 2000
      3. Print summary for each setting
      4. Build 2×3 comparison panel (rich keypoints + density heatmaps)
      5. Answer "more features = better?" question
    """
    print("\n" + "=" * 55)
    print("  Day 22 — ORB Feature Detection & Density Analysis")
    print("=" * 55)

    try:
        # Step 1: Generate synthetic texture
        print("\n[1/5] Generating synthetic textured image...")
        gray = make_synthetic_texture()
        if gray is None:
            raise RuntimeError("Failed to generate synthetic image")
        print(f"       Image: {gray.shape[1]}×{gray.shape[0]}")

        # Step 2-3: Detect keypoints at different nfeatures settings
        print("\n[2/5] Detecting ORB keypoints...")
        results = []
        for nf in NCOMPARE_VALUES:
            kps, desc = detect_keypoints(gray, nfeatures=nf)
            density = compute_density_map(gray, kps)
            results.append(
                {
                    "nfeatures": nf,
                    "keypoints": kps,
                    "descriptors": desc,
                    "density": density,
                }
            )
            print_summary(nf, kps)

        # Step 4: Build comparison panel
        print("\n[3/5] Building comparison panel...")
        compare_path = OUTPUT_DIR / "day_22_orb_comparison.png"
        build_comparison_panel(gray, results, compare_path)
        print(f"       Saved: {compare_path}")

        # Step 5: Answer the question
        print("\n[4/5] Analysis:")
        answer = answer_question(results)
        print(f"       {answer}")

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        return

    print("\n" + "=" * 55)
    print("  Done — check data/processed/day_22_orb_comparison.png")
    print("=" * 55)


if __name__ == "__main__":
    main()
