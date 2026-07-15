"""
Day 18 / 2026-07-15 / Module 8: Contour Geometric Analysis (Concepts B + C)
Goal: Extend Day 17 pipeline — compute geometry properties + shape classification per contour.
      Deliverable: labeled annotation + CSV attribute table.
Runtime: ~1 h
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from day_17_contours import load_image, preprocess, find_objects, get_color_palette

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# TODO — Reuse Day 17 functions (import or copy)
#
# You already have these working functions from day_17_contours.py:
#   load_image()
#   preprocess(gray) -> edge
#   find_objects(edge, min_area) -> contours (sorted by area descending)
#   get_color_palette(n) -> list of distinct BGR tuples
#
# HINT: Instead of copy-pasting, you can put shared utilities in src/ and
# import them. Or just copy the four functions above — Day 18 adds NEW
# functions below. The choice is yours.
# ---------------------------------------------------------------------------


# ==============================  NEW  ======================================
# Concept B — Contour Geometric Properties
# ==============================


def compute_properties(cnt: np.ndarray) -> dict:
    """
    Compute a full set of geometric properties for a single contour.

    Use the following cv2 / numpy functions (each is one line):
        - cv2.contourArea(cnt)
        - cv2.arcLength(cnt, closed=True)
        - cv2.moments(cnt)   → cx, cy from m10/m00, m01/m00
        - cv2.boundingRect(cnt)  → (x, y, w, h)
        - cv2.minAreaRect(cnt)   → ((cx, cy), (w, h), angle) — rotated

    HINT: circularity = 4 * pi * area / (perimeter ** 2)
          aspect_ratio = w / h  (use boundingRect's w, h)

    Returns dict with keys:
        area, perimeter, cx, cy, bbox_x, bbox_y, bbox_w, bbox_h,
        rotated_cx, rotated_cy, rotated_w, rotated_h, rotated_angle,
        circularity, aspect_ratio
    """
    # TODO: Compute each property and store in a dict
    # - If m00 == 0, set cx/cy = 0 (prevent ZeroDivisionError)
    raise NotImplementedError("compute_properties")


def build_property_table(
    contours: list[np.ndarray],
    shapes: list[str],
) -> list[dict]:
    """
    Build a list of property dicts (one per contour).

    Iterates over contours, calls compute_properties + shape label,
    returns a list that can be written as CSV or printed as a table.
    """
    # TODO: zip(contours, shapes), call compute_properties for each,
    #       prepend "id" field, collect into a list.
    raise NotImplementedError("build_property_table")


# ==============================  NEW  ======================================
# Concept C — Contour Approximation & Shape Classification
# ==============================


def classify_shape(cnt: np.ndarray) -> str:
    """
    Classify a contour as 'Triangle', 'Rectangle', 'Circle', or 'Irregular'.

    Pipeline:
        1. perimeter = cv2.arcLength(cnt, True)
        2. epsilon = 0.02 * perimeter  (starting value — tune if misclassifies)
        3. approx = cv2.approxPolyDP(cnt, epsilon, True)
        4. vertices = len(approx)
        5. Classify:
             vertices == 3  → "Triangle"
             vertices == 4  → "Rectangle"
             vertices >= 8  → "Circle"
             else           → "Irregular"

    HINT: 0.02 * perimeter is a heuristic. If your objects are very small or
    very noisy, try epsilon = 0.01 * perimeter (stricter = more vertices).
    If a known-rectangle classifies as "Irregular", epsilon is too small.

    HINT: A true rectangle has 4 vertices, but noise may give 5-7.
    Consider treating 4 <= vertices <= 6 as "Rectangle" for real-world images.

    Returns:
        One of: "Triangle", "Rectangle", "Circle", "Irregular"
    """
    # TODO: Implement the 5-step pipeline above.
    raise NotImplementedError("classify_shape")


# ==============================  NEW  ======================================
# Drawing — overlay bounding boxes + shape labels
# ==============================


def draw_boxes(
    image: np.ndarray,
    contours: list[np.ndarray],
    shapes: list[str],
    props_list: list[dict],
) -> None:
    """
    Draw bounding boxes (axis-aligned + rotated) and shape labels in-place.

    For each contour:
        1. Axis-aligned rect (blue)   — cv2.rectangle
        2. Rotated rect (green)       — cv2.drawContours + cv2.boxPoints
        3. Shape label near centroid   — cv2.putText

    HINT: cv2.boxPoints(rotated_rect) returns 4 corner points as float32.
    Convert to int with np.int32 before drawing.

    HINT: Use cv2.FONT_HERSHEY_SIMPLEX, scale≈0.4-0.5, thickness=1.
    Place the label slightly above the centroid so the text doesn't
    overlap the crosshair.

    This function draws on `image` in-place (no return value).
    """
    # TODO: For each (contour, shape, props):
    #   1. Extract bbox_x/y/w/h from props, draw blue rectangle
    #   2. Build rotated rect tuple, compute boxPoints, draw green polygon
    #   3. Put shape text above centroid
    raise NotImplementedError("draw_boxes")


# ==============================  NEW  ======================================
# CSV writer
# ==============================


def save_csv(props_list: list[dict], output_path: Path) -> None:
    """
    Write property table to CSV file.

    Fields (in order):
        id, shape, area, perimeter, circularity, aspect_ratio, cx, cy

    HINT: Use the built-in `csv` module (import csv) or manually write
    with f-string + "\n".join(). Both are fine for ~20 rows.

    HINT: Round float values to 2 decimal places for readability.
    """
    # TODO: Open output_path for writing, write header row,
    #       write one data row per dict in props_list.
    raise NotImplementedError("save_csv")


# ==============================  NEW  ======================================
# Visualization — side-by-side debug grid
# ==============================


def build_debug_grid(
    gray: np.ndarray,
    edge: np.ndarray,
    labeled: np.ndarray,
    csv_path: Path,
    output_dir: Path,
) -> None:
    """
    Create a 2x2 debug grid showing the full analysis pipeline.

    Panels:
        1. Grayscale input
        2. Canny edge map
        3. Labeled result (contours + bounding boxes + shape text)
        4. CSV content rendered as a text table (read csv_path, format as string)

    HINT for panel 4: Read the CSV with open() and put its content
    inside a plt.text box with fontfamily="monospace". Use
    axes[1, 1].text(0.05, 0.95, csv_text, fontfamily="monospace", ...).

    Saves:
        output_dir / "day_18_debug.png"
    """
    # TODO: Build a 2x2 subplot grid.
    #       Panel 1-2: grayscale + edge (cmap="gray")
    #       Panel 3: labeled image (BGR→RGB for matplotlib)
    #       Panel 4: CSV text table
    raise NotImplementedError("build_debug_grid")


# ==============================  Main  ======================================


def main():
    # --- Setup output directory ---
    output_dir = PROJECT_DIR / "data" / "processed" / "day_18"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- TODO: Load image (reuse load_image from Day 17) ---
    image_path = PROJECT_DIR / "data" / "raw" / "IMG_0701.png"
    # TODO: load image, resize if too large (same logic as Day 17)

    # --- TODO: Preprocess (reuse preprocess from Day 17) ---
    # gray = cv2.cvtColor(...)
    # edge = preprocess(gray)

    # --- TODO: Find contours (reuse find_objects from Day 17) ---
    # contours = find_objects(edge, min_area=500)
    # print how many objects found

    # --- TODO: Classify shapes ---
    # shapes = [classify_shape(c) for c in contours]
    # print shape counts (how many triangles, rectangles, circles...)

    # --- TODO: Compute properties ---
    # props_list = build_property_table(contours, shapes)

    # --- TODO: Draw results ---
    # labeled = image_bgr.copy()
    # draw_boxes(labeled, contours, shapes, props_list)

    # --- TODO: Save labeled image ---
    # cv2.imwrite(output_dir / "day_18_labeled.jpg", labeled)

    # --- TODO: Save CSV report ---
    # csv_path = output_dir / "day_18_report.csv"
    # save_csv(props_list, csv_path)

    # --- TODO: Build debug grid ---
    # build_debug_grid(gray, edge, labeled, csv_path, output_dir)

    # --- TODO: Print summary ---
    print(f"\nDone. View results in {output_dir}/")


if __name__ == "__main__":
    main()
