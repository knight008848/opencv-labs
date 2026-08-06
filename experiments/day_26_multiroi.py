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

Refactored 2026-08-06: ROI / segmentation / panel helpers now live in
src/vision.
"""

import json
import sys
import time
from pathlib import Path

import cv2

# Ensure project root is on sys.path so src/ imports resolve
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.vision import (  # noqa: E402
    RED_RANGES,
    crop_roi,
    define_roi_config,
    detect_hsv_objects,
    draw_objects,
    draw_roi_boxes,
    stack_panels,
)

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "day_26"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── Config ────────────────────────────

VIDEO_PATH = PROJECT_DIR / "data" / "raw" / "滚动球.mp4"  # adjust if needed
FRAME_NUMBER = 100  # which source frame to analyze (0-based)
# Keep blobs >= this fraction of each ROI's own area. A relative threshold
# suits multi-ROI: ROIs have very different sizes, so an absolute pixel
# count is arbitrary and frame-dependent (the ball's mask area swings 4-5x
# under motion blur / partial occlusion).
MIN_AREA_FRACTION = 0.0015  # 0.15% of the ROI area

# HSV red-color segmentation (matched to the rolling-ball target):
# the hue ranges live in src.vision.segmentation.RED_RANGES; here we tune
# only the morphology so the ball disk becomes one solid blob.
CLOSE_KERNEL = 15  # close gaps so the ball disk becomes one solid blob
OPEN_KERNEL = 3  # drop isolated salt-pixel noise

# ROI name -> draw color (BGR)
ROI_COLORS = {
    "panorama": (0, 0, 255),  # red  — full frame
    "work_area": (0, 255, 0),  # green — central 60%
    "inlet": (255, 0, 0),  # blue  — top-left 25%
}


def analyze_roi(roi_frame: cv2.typing.MatLike) -> list[dict]:
    """
    Run the standard detection pipeline inside one ROI: HSV red segmentation
    (two hue bands from src.vision) -> morph close/open -> area filter.
    NOTE: this single function must serve ALL ROIs (same logic everywhere).
    Returns a list of {"bbox", "centroid", "area"} — one entry per object.
    """
    return detect_hsv_objects(roi_frame, RED_RANGES, MIN_AREA_FRACTION, CLOSE_KERNEL, OPEN_KERNEL)


def build_four_panel(
    original: cv2.typing.MatLike,
    roi_config: dict[str, tuple[int, int, int, int]],
    results: dict[str, list[dict]],
):
    """
    Combine into one 2x2 figure:
      Panel 0: original frame + ROI boxes + labels
      Panels 1-3: each ROI's annotated analysis (green boxes + IDs)
    All panels are resized to a common display size, then stacked 2x2.
    Returns the combined BGR image.
    """
    # Panel 0: original frame with ROI boxes + labels
    panel0 = draw_roi_boxes(original.copy(), roi_config, ROI_COLORS)

    # Panels 1-3: one annotated analysis per ROI (per-ROI label color)
    roi_panels = []
    for name, rect in roi_config.items():
        crop = crop_roi(original, rect)
        panel = draw_objects(crop.copy(), results[name], show_id=True)
        cv2.putText(panel, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ROI_COLORS[name], 2)
        roi_panels.append(panel)

    return stack_panels([panel0, *roi_panels], cols=2, display_size=(640, 360))


def print_results(results: dict[str, list[dict]]) -> None:
    """
    Print a terminal table: one row per ROI with object count and the list
    of (id, bbox, area) for each detected object.
    """
    header = f"{'ROI':<12} | {'count':<5} | objects"
    print(header)
    print("-" * len(header))
    for name, objects in results.items():
        detail = [(i, o["bbox"], round(o["area"])) for i, o in enumerate(objects)]
        print(f"{name:<12} | {len(objects):<5} | {detail}")


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
    for name, rect in roi_config.items():
        roi_frame = crop_roi(frame, rect)
        results[name] = analyze_roi(roi_frame)
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
