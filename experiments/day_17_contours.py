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
    # TODO: cv2.imread + error handling
    ...


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
    # TODO: GaussianBlur -> Canny
    ...


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
    # TODO: findContours + area filter + sort
    ...


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
    # TODO: generate n distinct colors
    ...


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
    # TODO: drawContours + moments centroid + putText
    ...


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

    HINT: Use plt.subplots(2, 2, figsize=(10, 8)). Save with tight_layout.

    Saves:
        output_dir / "day_17_debug.png"
    """
    # TODO: build matplotlib grid with imshow
    ...


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    output_dir = Path("../data/processed/day_17")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Config ---
    # TODO: Change this to a real photo path when you have one.
    # For now, we use a synthetic test image.
    # image_path = Path("../data/raw/your_desk_photo.jpg")
    # image_bgr = load_image(image_path)

    # --- Fallback: create a synthetic test scene ---
    print("[1/5] Creating synthetic test scene...")
    # TODO: generate a simple binary image with several distinct shapes
    # (rectangles, circles, triangles) that simulates "objects on a desk"
    image_bgr = np.ones((600, 800, 3), dtype=np.uint8) * 200  # light gray bg
    # Draw some colored shapes as stand-in objects
    cv2.rectangle(image_bgr, (50, 50), (200, 200), (50, 100, 200), -1)   # blue rect
    cv2.circle(image_bgr, (400, 150), 80, (50, 200, 100), -1)            # green circle
    cv2.rectangle(image_bgr, (600, 50), (750, 200), (100, 50, 200), -1)  # red rect
    cv2.rectangle(image_bgr, (100, 350), (300, 550), (200, 100, 50), -1) # cyan rect
    cv2.circle(image_bgr, (550, 450), 100, (50, 100, 200), -1)           # blue circle
    cv2.imwrite(str(output_dir / "synthetic_input.jpg"), image_bgr)
    print(f"  Saved synthetic input: {output_dir / 'synthetic_input.jpg'}")

    # --- Pipeline ---
    print("[2/5] Converting to grayscale...")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    print(f"  Gray shape: {gray.shape}")

    print("[3/5] Running Canny edge detection...")
    # TODO: call preprocess(gray)
    edge = np.zeros_like(gray)  # placeholder
    cv2.imwrite(str(output_dir / "edges.png"), edge)

    print("[4/5] Finding and filtering contours...")
    # TODO: call find_objects(edge, min_area=500)
    contours: list[np.ndarray] = []  # placeholder

    if not contours:
        print("  WARNING: No contours found. Adjust Canny thresholds or")
        print("  dilate the edge map to close gaps between edge pixels.")

    print(f"  Found {len(contours)} objects (filtered area >= 500)")

    print("[5/5] Drawing labels and saving results...")
    # TODO: call draw_labeled_objects(image_bgr.copy(), contours, output_dir)
    # TODO: call build_debug_grid(gray, edge, labeled, output_dir)

    print(f"\nDone. View results in {output_dir}/")
    print("Tip: Replace the synthetic image with a real desk photo")
    print("  and tune Canny thresholds for best detection.")


if __name__ == "__main__":
    main()
