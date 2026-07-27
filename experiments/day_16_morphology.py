"""
Day 16 / 2026-07-02 / Module 7: Morphology (Concept B + C)
Goal: 6-in-1 morph operation comparison across 3 structuring element shapes.
      Deliverable: comparison grid + terminal shape recommendations.
Runtime: ~45 min
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Step 1 — Create a noisy binary test image
# ---------------------------------------------------------------------------


def create_test_image(size: tuple[int, int] = (1200, 1600)) -> np.ndarray:
    """
    Generate a synthetic binary image for morphology testing.

    The image should contain:
      - A few large shapes: filled rectangles, circles
      - A thin line (simulating a stroke/scratch)
      - Small white noise dots (salt) in background
      - Small black holes inside some shapes
      - A broken line (2-3 px gap) to test closing

    HINT: Start with np.zeros (black background). Draw white filled shapes
    with cv2.rectangle / cv2.circle / cv2.line. Add noise with
    np.random.rand and set random pixels to 255 (salt) or 0 (pepper).
    Leave a gap in one line by drawing two separate segments.

    Returns:
        uint8 binary image (0 or 255).
    """
    height, width = size
    bg = np.zeros((height, width), dtype=np.uint8)

    # --- Large filled shapes ---
    cv2.rectangle(bg, (0, 0), (width, 200), 255, -1)  # top bar
    cv2.rectangle(bg, (100, 400), (400, 700), 255, -1)  # left rectangle
    cv2.rectangle(bg, (width - 400, 500), (width - 100, 900), 255, -1)  # right rectangle

    cv2.circle(bg, (width // 2, height // 2), int(height // 4), 255, -1)  # center circle

    # --- Thin vertical line with two gaps — for closing/erosion test ---
    # Placed at x=450, well left of the circle (left edge at x=500).
    # Two small gaps let closing try to reconnect the broken line.
    vx = 450
    cv2.line(bg, (vx, 0), (vx, 224), 255, 2)  # top segment
    cv2.line(bg, (vx, 232), (vx, 794), 255, 2)  # middle segment
    cv2.line(bg, (vx, 802), (vx, height - 1), 255, 2)  # bottom segment

    # --- Broken horizontal line — for closing test ---
    # Placed above the circle (y < 300) so the circle fill doesn't
    # override the gap.  Gap is 10 px wide (no vertical line passes
    # through it anymore — the vertical line is now at x=450).
    cy = height // 2 - 320  # above the circle (y≈280)
    cv2.line(bg, (200, cy), (width // 2 - 6, cy), 255, 2)
    cv2.line(bg, (width // 2 + 6, cy), (width - 200, cy), 255, 2)

    # --- Salt noise: white dots on black background only ---
    fg_mask = bg > 0
    salt = (np.random.rand(height, width) < 0.005) & ~fg_mask
    bg[salt] = 255

    # --- Pepper noise: black dots inside white shapes only ---
    pepper = (np.random.rand(height, width) < 0.015) & fg_mask
    bg[pepper] = 0

    return bg


def binarize_from_color(image_path: Path) -> np.ndarray:
    """
    Alternative: load a real color image and convert to binary.

    Use cv2.imread -> cv2.cvtColor BGR2GRAY ->
    cv2.threshold with Otsu or adaptive.

    This is useful if you want to test on real data.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# Step 2 — Build structuring element
# ---------------------------------------------------------------------------


def get_kernel(shape_name: str, ksize: int) -> np.ndarray:
    """
    Return a structuring element of given shape and size.

    cv2.getStructuringElement supports:
      - cv2.MORPH_RECT    — rectangular (all ones)
      - cv2.MORPH_ELLIPSE — elliptical (round-ish)
      - cv2.MORPH_CROSS   — cross (center row + column)

    HINT: ksize must be odd and positive. Use ksize as both width and height.

    Returns:
        uint8 kernel of shape (ksize, ksize).
    """
    # Check ksize is odd and positive
    if ksize % 2 == 0 or ksize <= 0:
        raise ValueError("ksize must be odd and positive")

    if shape_name == "rect":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    elif shape_name == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    elif shape_name == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))
    else:
        raise ValueError(f"Unknown shape name: {shape_name}")


# ---------------------------------------------------------------------------
# Step 3 — Apply a single morphological operation
# ---------------------------------------------------------------------------


def apply_morphology(binary: np.ndarray, kernel: np.ndarray, operation: str) -> np.ndarray:
    """
    Apply one of 6 morphological operations to a binary image.

    Supported operations:
      - "original"   — return binary unchanged
      - "erode"      — cv2.erode
      - "dilate"     — cv2.dilate
      - "opening"    — cv2.morphologyEx + MORPH_OPEN (erode -> dilate)
      - "closing"    — cv2.morphologyEx + MORPH_CLOSE (dilate -> erode)
      - "gradient"   — cv2.morphologyEx + MORPH_GRADIENT (dilate - erode)

    HINT: For opening/closing/gradient, use cv2.morphologyEx with the
    appropriate cv2.MORPH_* flag. For gradient, think about what
    "dilate minus erode" produces visually.

    Returns:
        uint8 binary image.
    """
    if operation == "original":
        return binary
    elif operation == "erode":
        return cv2.erode(binary, kernel, iterations=1)
    elif operation == "dilate":
        return cv2.dilate(binary, kernel, iterations=1)
    elif operation == "opening":
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    elif operation == "closing":
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    elif operation == "gradient":
        return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    else:
        raise ValueError(f"Unknown operation: {operation}")


OPERATIONS = ["original", "erode", "dilate", "opening", "closing", "gradient"]


# ---------------------------------------------------------------------------
# Step 4 — Build comparison grid
# ---------------------------------------------------------------------------


def build_morphology_grid(
    binary: np.ndarray,
    kernel_sizes: list[int],
    shapes: list[str],
    output_dir: Path,
) -> None:
    """
    Create a grid comparing all operations × kernel sizes × shapes.

    Since this is a headless environment (no Trackbar), generate a
    static grid figure using matplotlib:
      - One big figure per kernel size
      - Subplot grid: 3 rows (shapes) x 6 columns (operations)
      - Each panel shows the morphology result for that combination

    Args:
        binary: Input binary image.
        kernel_sizes: List of kernel sizes to test (odd, > 0).
        shapes: List of structuring element shape names.
        output_dir: Directory to save the PNG grids.

    Saves:
        morph_grid_k{N}.png for each kernel size.
    """
    for ksize in kernel_sizes:
        fig, axes = plt.subplots(3, 6, figsize=(16, 7))
        fig.suptitle(f"Kernel size: {ksize}x{ksize}", fontsize=14, y=0.98)

        for row_idx, shape in enumerate(shapes):
            kernel = get_kernel(shape, ksize)
            for col_idx, op in enumerate(OPERATIONS):
                result = apply_morphology(binary, kernel, op)
                ax = axes[row_idx, col_idx]
                ax.imshow(result, cmap="gray", vmin=0, vmax=255)
                ax.axis("off")

                # Column header on the first row
                if row_idx == 0:
                    ax.set_title(op, fontsize=11)

            # Row label on the leftmost column
            axes[row_idx, 0].set_ylabel(
                shape.upper(), rotation=0, fontsize=11, labelpad=15, va="center"
            )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = output_dir / f"morph_grid_k{ksize}.png"
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        print(f"  Saved {save_path.name}")


# ---------------------------------------------------------------------------
# Step 5 — Analyze structuring element shapes
# ---------------------------------------------------------------------------


def analyze_element_shapes() -> str:
    """
    Return a text analysis of when each structuring element shape is best.

    For each shape (RECT, ELLIPSE, CROSS), explain:
      - What it looks like visually (draw a small grid)
      - What kind of object it preserves best
      - When it causes unwanted artifacts

    Returns:
        Multi-line string for terminal output.
    """

    lines = []
    lines.append("=" * 60)
    lines.append("Structuring Element Shape Analysis")
    lines.append("=" * 60)
    lines.append("")

    for name, morph_const in [
        ("RECT", cv2.MORPH_RECT),
        ("ELLIPSE", cv2.MORPH_ELLIPSE),
        ("CROSS", cv2.MORPH_CROSS),
    ]:
        k = cv2.getStructuringElement(morph_const, (7, 7))
        # Render the kernel as ASCII art
        ascii_rows = []
        for row in k:
            ascii_rows.append("  " + " ".join("█" if v else "·" for v in row.flatten()))
        kernel_str = "\n".join(ascii_rows)

        lines.append(f"── {name} (7×7) ──")
        lines.append(kernel_str)
        lines.append("")

    lines.append("─" * 60)
    lines.append("Shape-by-shape guidance")
    lines.append("─" * 60)
    lines.append("")
    lines.append(
        "RECT  — all 4 directions weighted equally.\n"
        "  Best for: QR codes, document text, PCB inspection —\n"
        "            anything with sharp 90° corners.\n"
        "  Artifact: rounds NOTHING — it preserves corners so\n"
        "            aggressively that a circle becomes an octagon\n"
        "            after a few iterations."
    )
    lines.append("")
    lines.append(
        "ELLIPSE — approximates a circle (isotropic).\n"
        "  Best for: cells, coins, pellets, organic shapes.\n"
        "  Artifact: smooths sharp corners — if you erode a\n"
        "            rectangle with ELLIPSE 7×7, its corners\n"
        "            get visibly rounded."
    )
    lines.append("")
    lines.append(
        "CROSS  — only the center row and column are active.\n"
        "  Best for: preserving / detecting thin lines that\n"
        "            are exactly horizontal or vertical.\n"
        "  Artifact: diagonal features (45°, 135°) are\n"
        "            obliterated — a diagonal scratch that\n"
        "            survives RECT erosion may disappear under\n"
        "            CROSS erosion."
    )
    lines.append("")
    lines.append(
        "Compare morph_grid_k{3,5,7}.png — the gradient\n"
        "column is the most sensitive: RECT produces blocky\n"
        "edges, ELLIPSE gives smooth contours, and CROSS\n"
        "picks up horizontal & vertical edges but misses\n"
        "diagonal ones."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    output_dir = Path("../data/processed/day_16")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: create test image
    print("[1/4] Creating test binary image...")
    binary = create_test_image()
    cv2.imwrite(str(output_dir / "original_binary.png"), binary)
    print(f"  Shape: {binary.shape}, dtype: {binary.dtype}")
    print(f"  White pixels: {cv2.countNonZero(binary)} / {binary.size}")

    # --- Quick sanity check ---
    print("[2/4] Quick check: erode + dilate with 3x3 RECT...")
    k = get_kernel("rect", 3)
    before = int(cv2.countNonZero(binary))
    eroded = apply_morphology(binary, k, "erode")
    dilated = apply_morphology(binary, k, "dilate")
    after_erode = int(cv2.countNonZero(eroded))
    after_dilate = int(cv2.countNonZero(dilated))
    print(f"  Before:       {before:>7} white pixels")
    print(f"  After  erode: {after_erode:>7} (Δ{after_erode - before:+d})")
    print(f"  After  dilate:{after_dilate:>7} (Δ{after_dilate - before:+d})")

    # --- Full grid ---
    print("[3/4] Generating morphology comparison grid...")
    kernel_sizes = [3, 5, 7]
    shapes = ["rect", "ellipse", "cross"]
    build_morphology_grid(binary, kernel_sizes, shapes, output_dir)

    # --- Shape analysis ---
    print("[4/4] Structuring element shape analysis:")
    print(analyze_element_shapes())

    print(f"\nDone. View results in {output_dir}/")
    print("Tip: open the PNG grid and compare how RECT / ELLIPSE / CROSS")
    print("  affect the same operation differently.")


if __name__ == "__main__":
    main()
