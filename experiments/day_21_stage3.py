"""
Day 21 / 2026-07-23 / Module 10: Pipeline Practice
File: day_21_stage3.py
Goal: 6-step standalone pipeline on a synthetic image —
      gray → blur → Canny → morph_close → contours → geometry analysis.
      No external file dependencies.
Deliverable: terminal print + 2×3 debug grid saved to data/processed/
Runtime: ~5 min
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── Config ────────────────────────────

# Define image size and shape positions for the synthetic canvas
CANVAS_W, CANVAS_H = 600, 400

BLUR_KSIZE = (5, 5)
CANNY_T1, CANNY_T2 = 50, 150
MORPH_KERNEL = 3
MIN_AREA = 50


# ──────────────────────── 1. Synthesize Image ───────────────────


def make_synthetic_image(w: int = CANVAS_W, h: int = CANVAS_H) -> np.ndarray:
    """
    Create a white background BGR image with 3-4 basic geometric shapes
    drawn in black (filled).

    HINT:
      canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
      Use cv2.rectangle, cv2.circle, cv2.fillPoly to draw shapes.
    """
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Rectangle (50, 50) → (150, 150)
    cv2.rectangle(canvas, (50, 50), (150, 150), 0, -1)

    # Circle at center (300, 100), radius 60
    cv2.circle(canvas, (300, 100), 60, 0, -1)

    # Triangle via fillPoly
    pts = np.array([[500, 180], [420, 30], [580, 30]], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], 0)

    return canvas


# ────────────────────── Pipeline Steps ──────────────────────────


def step_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to single-channel grayscale."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def step_blur(gray: np.ndarray, ksize: tuple[int, int] = BLUR_KSIZE) -> np.ndarray:
    """Apply Gaussian blur to reduce noise before edge detection."""
    return cv2.GaussianBlur(gray, ksize, 0)


def step_canny(blurred: np.ndarray, t1: int = CANNY_T1, t2: int = CANNY_T2) -> np.ndarray:
    """Canny edge detection on blurred grayscale image."""
    return cv2.Canny(blurred, t1, t2)


def step_morph_close(edges: np.ndarray, kernel_size: int = MORPH_KERNEL) -> np.ndarray:
    """Morphological closing (dilate → erode) to connect broken edges."""
    if edges is None:
        return edges

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)


def step_find_contours(binary: np.ndarray, min_area: float = MIN_AREA) -> list[np.ndarray]:
    """Find external contours, filter by min area, sort descending."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    return sorted(filtered, key=cv2.contourArea, reverse=True)


# ──────────────────────── 6. Geometry Analysis ──────────────────


def analyze_contour(cnt: np.ndarray) -> dict:
    """
    Compute area, circularity, and classify shape for one contour.

    Returns dict with keys: area, circularity, shape.
    circularity = 4 * pi * area / (perimeter ** 2)
    shape via approxPolyDP vertex count.
    """
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0.0

    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    vertices = len(approx)

    if vertices == 3:
        shape = "Triangle"
    elif vertices == 4:
        shape = "Rectangle"
    elif vertices >= 8:
        shape = "Circle"
    else:
        shape = "Irregular"

    return {
        "area": area,
        "circularity": circularity,
        "shape": shape,
    }


# ──────────────────────── Draw Annotations ──────────────────────


def draw_contours_and_labels(
    canvas_bgr: np.ndarray,
    contours: list[np.ndarray],
    analysis: list[dict],
) -> np.ndarray:
    """
    Draw contour outlines + centroid labels (area, shape) on the original canvas.

    Returns annotated BGR image.
    """
    image = canvas_bgr.copy()

    # Simple inline color palette (BGR)
    palette = [
        (0, 0, 255),  # red
        (0, 255, 0),  # green
        (255, 0, 0),  # blue
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
    ]

    for i, cnt in enumerate(contours):
        color = palette[i % len(palette)]

        # Draw contour outline
        cv2.drawContours(image, contours, i, color, thickness=2)

        # Axis-aligned bounding box (blue)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Centroid via moments
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y

        # Label
        area = analysis[i]["area"]
        shape = analysis[i]["shape"]
        cv2.putText(
            image,
            f"ID:{i} A:{int(area)} {shape}",
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    return image


# ──────────────────────── 2×3 Debug Grid ────────────────────────


def save_debug_grid(
    images: list[np.ndarray],
    titles: list[str],
    output_path: Path,
) -> None:
    """
    Display 6 intermediate results in a 2×3 grid and save to disk.

    HINT:
      fig, axes = plt.subplots(2, 3, figsize=(12, 8))
      For single-channel images use cmap="gray".
      fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
      plt.close(fig)
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for ax, img, title in zip(axes.flat, images, titles):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Debug grid saved: {output_path}")


# ──────────────────────────── Main ──────────────────────────────


def main() -> None:
    """
    Run the 6-step pipeline with try/except protection.

    Steps:
      1. Synthesize test image
      2. Grayscale
      3. Gaussian blur
      4. Canny
      5. Morph close
      6. Find contours → analyze each
      (7. Draw annotations for the grid)

    Print each object's area, circularity, shape.
    Save 2×3 grid to data/processed/day_21_pipeline.png.
    """
    print("\n" + "=" * 50)
    print("  Day 21 — Stage 3: Standalone Pipeline")
    print("=" * 50)

    try:
        # ---- Step 1: Synthesize ----
        print("\n[1/6] Synthesizing test image...")
        canvas = make_synthetic_image()
        print(f"       Canvas: {canvas.shape[1]}×{canvas.shape[0]}")

        # ---- Steps 2-5: grayscale → blur → Canny → morph_close ----
        gray = step_grayscale(canvas)
        blurred = step_blur(gray)
        edges = step_canny(blurred)
        closed = step_morph_close(edges)

        # ---- Step 6: contours → analyze ----
        contours = step_find_contours(closed)

        analysis = [analyze_contour(c) for c in contours]

        print(f"\n  Detected {len(contours)} object(s):")
        for i, a in enumerate(analysis):
            print(
                f"    [{i}] area={a['area']:8.1f}  "
                f"circularity={a['circularity']:.3f}  "
                f"shape={a['shape']}"
            )

        # ---- Draw bare contour overlay for the grid ----
        contour_img = canvas.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)

        # ---- Draw annotations on a separate copy ----
        annotated = draw_contours_and_labels(canvas, contours, analysis)

        # ---- Build 2×3 grid ----
        images = [gray, blurred, edges, closed, contour_img, annotated]
        titles = [
            "2. Grayscale",
            "3. Gaussian Blur",
            "4. Canny Edges",
            "5. Morph Close",
            "6. Contours",
            "7. Annotated",
        ]

        # ---- Save grid ----
        grid_path = OUTPUT_DIR / "day_21_pipeline.png"
        save_debug_grid(images, titles, grid_path)

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        return

    print("\n" + "=" * 50)
    print("  Done — check data/processed/day_21_pipeline.png")
    print("=" * 50)


if __name__ == "__main__":
    main()
