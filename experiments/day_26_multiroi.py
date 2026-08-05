"""
Day 26 / 2026-08-05 / Module 12: Multi-ROI Analysis & Data Packaging
File: day_26_multiroi.py
Goal: Split one video frame into 3 ROIs (panorama / central work area /
      top-left inlet), run the SAME detection pipeline inside each ROI
      independently, and render a four-panel comparison + terminal report.
Deliverable: four-panel PNG + per-ROI object-count table + JSON summary
Runtime: ~10 s

Headless note: no imshow / waitKey / trackbar. Visualisation via PNG
export. See CLAUDE.md for headless policy.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "day_26"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── Config ────────────────────────────

VIDEO_PATH = PROJECT_DIR / "data" / "raw" / "滚动球.mp4"  # adjust if needed
FRAME_NUMBER = 100  # which source frame to analyze (0-based)
MIN_AREA = 200  # drop contours below this area inside each ROI
CANNY_LOW, CANNY_HIGH = 50, 150  # Canny hysteresis thresholds

# ROI name -> draw color (BGR)
ROI_COLORS = {
    "panorama": (0, 0, 255),    # red  — full frame
    "work_area": (0, 255, 0),   # green — central 60%
    "inlet": (255, 0, 0),       # blue  — top-left 25%
}


def define_roi_config(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    """
    Return roi_name -> (x, y, w, h) slices for the three analysis windows.
      - "panorama":  whole frame
      - "work_area": central 60% of the frame
      - "inlet":     top-left 25% of the frame
    All extents are computed as fractions of the frame size, so the same
    config works for any resolution.
    """
    # ── Your implementation here ──
    # Hint: fractional math on width/height, then round() to int.
    #   x, y are the top-left corner; w, h are the window extents.
    #   e.g. work_area: x = round(0.2 * width),  w = round(0.6 * width)
    #                  y = round(0.2 * height), h = round(0.6 * height)
    #   inlet:         x = 0, y = 0, w = round(0.25 * width), h = round(0.25 * height)
    #   Return a dict keyed by the same names as ROI_COLORS.
    pass  # TODO


def analyze_roi(roi_frame: np.ndarray, min_area: int) -> list[dict]:
    """
    Run the standard detection pipeline inside one ROI:
      gray -> Gaussian blur -> Canny -> findContours -> area filter.
    Returns a list of {"bbox": (x, y, w, h), "centroid": (cx, cy),
                       "area": float} — one entry per detected object.
    NOTE: this single function must serve ALL ROIs (same logic everywhere).
    """
    # ── Your implementation here ──
    # Hint: reuse the pipeline from Day 20 / Day 25 —
    #   gray    = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    #   blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #   edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    #   cnts    = cv2.findContours(edges, cv2.RETR_EXTERNAL,
    #                              cv2.CHAIN_APPROX_SIMPLE)[0]
    #   for each cnt with area >= min_area:
    #       bbox = cv2.boundingRect(cnt)
    #       M = cv2.moments(cnt)          # guard M["m00"] == 0
    #       centroid = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
    pass  # TODO


def draw_roi_boxes(frame: np.ndarray, roi_config: dict) -> np.ndarray:
    """
    Draw one rectangle per ROI on the original frame, colored per ROI_COLORS,
    and label each rectangle with its roi name via cv2.putText.
    Returns the annotated frame (mutates the input in place).
    """
    # ── Your implementation here ──
    # Hint: cv2.rectangle(frame, (x, y), (x + w, y + h), ROI_COLORS[name], 2)
    #       then cv2.putText(frame, name, (x + 5, y + 20), FONT_HERSHEY_SIMPLEX,
    #                        0.7, ROI_COLORS[name], 2)
    pass  # TODO


def draw_objects(frame: np.ndarray, objects: list[dict]) -> np.ndarray:
    """
    Draw each detected object as a green bounding box + an "#id" label.
    Returns the annotated frame (mutates the input in place).
    """
    # ── Your implementation here ──
    # Hint: for idx, obj in enumerate(objects):
    #       x, y, w, h = obj["bbox"]
    #       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #       cv2.putText(frame, f"#{idx}", (x + 2, y - 4), ...)
    pass  # TODO


def build_four_panel(
    original: np.ndarray,
    roi_config: dict,
    results: dict[str, list[dict]],
) -> np.ndarray:
    """
    Combine into one 2x2 figure:
      Panel 0: original frame + ROI boxes + labels
      Panels 1-3: each ROI's annotated analysis (green boxes + IDs)
    All panels are resized to a common display size, then stacked 2x2.
    Returns the combined BGR image.
    """
    # ── Your implementation here ──
    # Hint:
    #   1. panel0 = draw_roi_boxes(original.copy(), roi_config)
    #   2. For each (name, objects) in results.items():
    #        crop = original[y:y+h, x:x+w]          # use roi_config[name]
    #        panel = draw_objects(crop.copy(), objects)
    #        cv2.putText(panel, name, (10, 30), ...) # label each ROI panel
    #   3. Resize all four panels to a common (W, H) so the grid is clean.
    #      A fixed DISPLAY_W/DISPLAY_H (e.g. 640 x 360) is fine.
    #   4. top_row = np.hstack([panel0, panel1]); bottom_row = np.hstack([...])
    #      return np.vstack([top_row, bottom_row])
    pass  # TODO


def print_results(results: dict[str, list[dict]]) -> None:
    """
    Print a terminal table: one row per ROI with object count and the list
    of (id, bbox, area) for each detected object.
    """
    # ── Your implementation here ──
    # Hint: header line + separator + per-ROI rows via f-strings, e.g.
    #   f"{name:<10} | {len(objects):<4} | {[(i, o['bbox'], round(o['area'])) for i, o in enumerate(objects)]}"
    pass  # TODO


def main() -> None:
    """
    Pipeline:
      1. Open the video, seek to FRAME_NUMBER, read one frame
      2. define_roi_config -> draw_roi_boxes on a copy
      3. For each ROI: crop -> analyze_roi (same function for all ROIs)
      4. build_four_panel -> save PNG
      5. print_results + save a JSON summary
    """
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)
        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {FRAME_NUMBER}")

    h, w = frame.shape[:2]
    roi_config = define_roi_config(w, h)

    start = time.perf_counter()
    results: dict[str, list[dict]] = {}
    for name, (x, y, rw, rh) in roi_config.items():
        roi_frame = frame[y : y + rh, x : x + rw]
        results[name] = analyze_roi(roi_frame, MIN_AREA)
    elapsed = time.perf_counter() - start

    panel = build_four_panel(frame, roi_config, results)
    panel_path = OUTPUT_DIR / "day_26_four_panel.png"
    cv2.imwrite(str(panel_path), panel)

    print_results(results)

    summary = {
        "frame": FRAME_NUMBER,
        "elapsed_s": round(elapsed, 3),
        "results": results,
    }
    summary_path = OUTPUT_DIR / "day_26_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPanel:   {panel_path}")
    print(f"Summary: {summary_path}")
    print(f"Elapsed: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
