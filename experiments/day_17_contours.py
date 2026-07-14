"""
Day 17 / 2026-07-14 / Module 8: Contour Extraction (Concept A)
Goal: Object detection labeler — detect + label objects in a desk-scene photo.
      Deliverable: labeled annotation image with ID + area per object.
Runtime: ~45 min
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Step 1 — Load an input image
# ---------------------------------------------------------------------------


def load_image(path: str | Path) -> np.ndarray:
    """
    Load a desk-scene photo in BGR color.

    Use cv2.imread. If the image is None (bad path / unreadable),
    raise FileNotFoundError with a helpful message.

    Returns:
        uint8 BGR image, shape (H, W, 3).
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


# ---------------------------------------------------------------------------
# Step 2 — Preprocess: gray -> blur -> edges
# ---------------------------------------------------------------------------


def preprocess(gray: np.ndarray) -> np.ndarray:
    """
    Convert grayscale to edge map for findContours.

    Pipeline:
        1. GaussianBlur (choose ksize — odd, >= 3)
        2. Canny (choose low/high thresholds)

    HINT: If you're getting too many/few edges, adjust the Canny thresholds.
    Start with 50 and 150 (the canonical defaults), then tune.

    Returns:
        uint8 edge image (binary: 0 or 255).
    """

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray_blur, 50, 150)


# ---------------------------------------------------------------------------
# Step 3 — Find and filter contours
# ---------------------------------------------------------------------------


def find_objects(edge: np.ndarray, min_area: int = 500) -> list[np.ndarray]:
    """
    Find contours in a binary edge image, filtering out small noise.

    Steps:
        1. cv2.findContours with RETR_EXTERNAL + CHAIN_APPROX_SIMPLE
        2. Filter contours by cv2.contourArea >= min_area
        3. Return the filtered list (sorted by area descending = big first)

    HINT: The edge from Canny is 0/255. findContours treats white (255) as
    foreground. If contours look "hollow" (just the edges, not the fill),
    consider dilating the edge map first to close gaps, or use RETR_FLOODFILL
    — but RETR_EXTERNAL on edges is usually fine for well-separated objects.

    Returns:
        List of contour arrays, each shape (N, 1, 2).
    """
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    return sorted(filtered, key=cv2.contourArea, reverse=True)


# ---------------------------------------------------------------------------
# Step 4 — Draw contours with distinct colors
# ---------------------------------------------------------------------------


def get_color_palette(n: int) -> list[tuple[int, int, int]]:
    """
    Return n distinct BGR colors for drawing contours.

    HINT: Use a fixed palette or generate evenly-spaced hues in HSV
    (H from 0 to 179, S=255, V=255), then convert back to BGR via
    cv2.cvtColor. This guarantees every contour gets a visually distinct color.

    Returns:
        List of n BGR tuples like [(B, G, R), ...].
    """

    hsv = np.zeros((1, n, 3), dtype=np.uint8)
    hsv[0, :, 0] = np.linspace(0, 179, n, dtype=np.uint8)  # evenly spaced hues
    hsv[0, :, 1] = 255  # full saturation
    hsv[0, :, 2] = 255  # full value
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return [tuple(int(v) for v in c) for c in bgr[0]]


def draw_labeled_objects(
    image: np.ndarray,
    contours: list[np.ndarray],
    output_dir: Path,
) -> None:
    """
    Draw all contours with distinct colors and annotate ID + area.

    For each contour:
        1. Pick a color from the palette (cycle if more contours than colors)
        2. cv2.drawContours to draw the contour outline (thickness ≈ 2-3)
        3. Compute the centroid via cv2.moments
        4. Put text "ID:{i} Area:{area}" near the centroid

    HINT for text placement:
        - Use cv2.putText with cv2.FONT_HERSHEY_SIMPLEX
        - Choose font scale and color that contrast with the image
        - If centroid is near the edge, offset the text so it stays on-screen

    Saves:
        output_dir / "day_17_labeled.jpg"
        output_dir / "day_17_labeled.png"  (lossless backup)
    """

    # Early return if no contours (get_color_palette(0) would crash)
    if not contours:
        cv2.imwrite(str(output_dir / "day_17_labeled.jpg"), image)
        cv2.imwrite(str(output_dir / "day_17_labeled.png"), image)
        return

    colors = get_color_palette(len(contours))
    for i, c in enumerate(contours):
        color = colors[i]

        # Draw only the i-th contour (contourIdx=i), not all at once
        cv2.drawContours(image, contours, i, color, thickness=2)
        moments = cv2.moments(c)
        if moments["m00"] > 0:
            x = int(moments["m10"] / moments["m00"])
            y = int(moments["m01"] / moments["m00"])
            cv2.putText(
                image,
                f"ID:{i} Area:{int(cv2.contourArea(c))}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

    cv2.imwrite(str(output_dir / "day_17_labeled.jpg"), image)
    cv2.imwrite(str(output_dir / "day_17_labeled.png"), image)


# ---------------------------------------------------------------------------
# Step 5 — Optional: create a debug grid showing pipeline intermediates
# ---------------------------------------------------------------------------


def build_debug_grid(
    gray: np.ndarray,
    edge: np.ndarray,
    labeled: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Create a 2x2 debug grid showing pipeline intermediates.

    Panels:
        1. Original (grayscale)
        2. Edges (Canny output)
        3. Labeled result (contours + annotations)
        4. (optional) Area histogram of detected objects

    Saves:
        output_dir / "day_17_debug.png"
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Contour Detection Pipeline", fontsize=14)

    # Panel 1: Grayscale original
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("1. Grayscale Input")
    axes[0, 0].axis("off")

    # Panel 2: Edge map
    axes[0, 1].imshow(edge, cmap="gray")
    axes[0, 1].set_title("2. Canny Edges")
    axes[0, 1].axis("off")

    # Panel 3: Labeled result (BGR -> RGB for matplotlib)
    labeled_rgb = cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(labeled_rgb)
    axes[1, 0].set_title("3. Labeled Objects")
    axes[1, 0].axis("off")

    # Panel 4: Info placeholder (contour areas not available here)
    axes[1, 1].text(
        0.5,
        0.5,
        "Contour area histogram\ncan be added after\nfind + draw steps.",
        ha="center",
        va="center",
        fontsize=12,
        transform=axes[1, 1].transAxes,
    )
    axes[1, 1].set_title("4. Stats")
    axes[1, 1].axis("off")

    plt.tight_layout()
    save_path = output_dir / "day_17_debug.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved debug grid: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    output_dir = PROJECT_DIR / "data" / "processed" / "day_17"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load a real desk-scene photo ---
    image_path = PROJECT_DIR / "data" / "raw" / "IMG_0701.png"
    print(f"[1/5] Loading {image_path.name}...")
    image_bgr = load_image(image_path)
    h, w = image_bgr.shape[:2]
    print(f"  Original size: {w}x{h}")

    # Resize if too large (keep aspect ratio, max dim 1200px)
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image_bgr = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
        print(f"  Resized to: {new_size[0]}x{new_size[1]}")

    # Save a copy of input
    cv2.imwrite(str(output_dir / "input.jpg"), image_bgr)

    # --- Pipeline ---
    print("[2/5] Converting to grayscale...")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    print(f"  Gray shape: {gray.shape}")

    print("[3/5] Running Canny edge detection...")
    edge = preprocess(gray)
    cv2.imwrite(str(output_dir / "edges.png"), edge)
    nonzero = cv2.countNonZero(edge)
    print(f"  Edge pixels: {nonzero}/{edge.size} ({100 * nonzero / edge.size:.1f}%)")

    print("[4/5] Finding and filtering contours...")
    contours = find_objects(edge, min_area=500)

    if not contours:
        print("  WARNING: No contours found. Adjust Canny thresholds or")
        print("  dilate the edge map to close gaps between edge pixels.")
    else:
        areas = [cv2.contourArea(c) for c in contours]
        print(f"  Found {len(contours)} objects (filtered area >= 500)")
        print(f"  Areas: min={int(min(areas))}, max={int(max(areas))}, total={int(sum(areas))}")

    print("[5/5] Drawing labels and saving results...")
    labeled = image_bgr.copy()
    draw_labeled_objects(labeled, contours, output_dir)
    build_debug_grid(gray, edge, labeled, output_dir)

    print(f"\nDone. View results in {output_dir}/")


if __name__ == "__main__":
    main()
