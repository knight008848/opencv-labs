"""
Day 25 / 2026-08-03 / Module 11: Motion Detection & Tracking
File: day_25_motion.py
Goal: Detect moving objects with background subtraction (MOG2), clean the
      foreground mask, track centroids across frames, and render a
      three-panel video (original | clean mask | detection) at a fixed 24 fps.
Deliverable: three-panel video + sample PNG panels + per-frame object counts
Runtime: ~5-8 min (depends on source)

Headless note: no imshow / waitKey / trackbar. Visualisation via VideoWriter
and PNG export. See CLAUDE.md for headless policy.
"""

import time
from pathlib import Path

import cv2
import numpy as np

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "day_25_frames"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── Config ────────────────────────────

# ⚠️  Replace with your re-shot clip: fixed camera + one clearly moving object.
#     The default source below has a MOVING CAMERA — the foreground mask will
#     flood with the whole frame. It is only a "bad case" to test the pipeline.
VIDEO_PATH = PROJECT_DIR / "data" / "raw" / "rgb_79c1787d6c.mp4"

OUTPUT_FPS = 24.0     # pinned output frame rate (same convention as day 24)
MIN_AREA = 200        # drop contours smaller than this (speckle noise filter)
TRAIL_LENGTH = 30     # keep the last N centroids per tracked object
MAX_MATCH_DIST = 60   # centroids farther apart than this are not the same object
SAVE_EVERY = 60       # export one sample three-panel PNG every N output frames


def init_background_subtractor() -> cv2.BackgroundSubtractorMOG2:
    """
    Create a MOG2 background subtractor tuned for indoor scenes.
    Returns a background-subtraction model whose .apply(frame) yields a
    per-frame foreground mask (0=background, 127=shadow, 255=foreground).
    """
    # ── Your implementation here ──
    # Hint:
    #   return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    pass


def process_frame(
    frame: np.ndarray, bg_sub: cv2.BackgroundSubtractorMOG2, kernel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply background subtraction, then clean the mask with morphology.
    Returns (raw_fg_mask, clean_fg_mask) — both uint8, single-channel.
    Shadows (value 127) are NOT foreground and must be dropped.
    """
    # ── Your implementation here ──
    # Hint:
    #   raw = bg_sub.apply(frame)                   # 0 / 127 / 255
    #   fg = (raw == 255).astype(np.uint8) * 255    # keep only true foreground
    #   clean = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)   # erode + dilate
    #   return fg, clean
    pass


def find_moving_objects(fg_mask_clean: np.ndarray, min_area: int) -> list[dict]:
    """
    Find contours in the clean foreground mask and drop tiny speckles.
    Returns a list of {"bbox": (x, y, w, h), "centroid": (cx, cy)}.
    """
    # ── Your implementation here ──
    # Hint:
    #   cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #   for each cnt with cv2.contourArea(cnt) >= min_area:
    #       M = cv2.moments(cnt)                 # guard M["m00"] == 0
    #       centroid = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
    #       bbox = cv2.boundingRect(cnt)
    pass


def draw_detections(frame: np.ndarray, objects: list[dict]) -> np.ndarray:
    """
    Draw a green bounding rectangle + red centroid dot per moving object.
    Returns the annotated frame (mutates the input frame in place).
    """
    # ── Your implementation here ──
    # Hint:
    #   for obj in objects:
    #       x, y, w, h = obj["bbox"]; cx, cy = obj["centroid"]
    #       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #       cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    pass


def update_trajectories(
    objects: list[dict],
    trails: dict[int, list[tuple[int, int]]],
    next_id: int,
    max_dist: float,
    max_len: int,
) -> tuple[dict[int, list[tuple[int, int]]], int]:
    """
    Associate current centroids with existing trails via nearest-neighbour
    matching, then grow / trim / spawn / drop trails.
      - matched centroid -> append to that trail, trim to max_len
      - unmatched centroid -> new trail with a fresh id
      - old trail with no match this frame -> deleted (object left the scene)
    Returns (updated trails, next available id).
    """
    # ── Your implementation here ──
    # Hint (greedy nearest-neighbour):
    #   1. For each current centroid, compute its distance to every trail's
    #      LAST point; remember the closest trail within max_dist.
    #   2. Track which trails are already matched this frame so one trail
    #      is never reused by two objects.
    #   3. Append the centroid to the matched trail, then trim to max_len.
    #   4. Unmatched centroid -> trails[next_id] = [centroid]; next_id += 1.
    #   5. Delete any trail that got no match this frame.
    #   Return (trails, next_id).
    pass


def draw_trajectories(frame: np.ndarray, trails: dict[int, list[tuple[int, int]]]) -> np.ndarray:
    """
    Draw each trail as connected line segments.
    Trails with fewer than 2 points draw nothing -> no ghost line on first sight.
    """
    # ── Your implementation here ──
    # Hint:
    #   for pts in trails.values():
    #       if len(pts) >= 2:
    #           cv2.polylines(frame, [np.array(pts)], isClosed=False, color=(255, 0, 0), thickness=2)
    pass


def build_three_panel(original: np.ndarray, fg_mask_clean: np.ndarray, detection: np.ndarray) -> np.ndarray:
    """
    Stack three same-size panels horizontally: original | mask | detection.
    Resize each panel to a common height first so the output is manageable;
    label each region with cv2.putText. Returns the combined BGR frame.
    """
    # ── Your implementation here ──
    # Hint:
    #   height = 480
    #   to_panel = lambda img: cv2.resize(img, (..., height))
    #   mask_bgr = cv2.cvtColor(to_panel(fg_mask_clean), cv2.COLOR_GRAY2BGR)
    #   label each panel ("source" / "mask" / "detection")
    #   return np.hstack([src_panel, mask_bgr, det_panel])
    pass


def main() -> None:
    """
    Pipeline:
      1. Open video, init MOG2 + morphology kernel + trail store
      2. Single pass on the 24 fps grid: for each kept source frame
         process_frame -> find_moving_objects -> draw_detections
         -> update_trajectories -> draw_trajectories -> build_three_panel -> write
      3. Every SAVE_EVERY frames, export a sample three-panel PNG
      4. Print per-frame object count + elapsed time
    """
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    ratio = src_fps / OUTPUT_FPS              # source frames per output frame

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = PROJECT_DIR / "data" / "processed" / "day_25_annotated.mp4"

    bg_sub = init_background_subtractor()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    trails: dict[int, list[tuple[int, int]]] = {}
    next_id = 0
    writer = None

    frame_idx = 0         # source frame index
    saved = 0
    next_target = 0       # next source frame to keep (real-time 24 fps grid)
    start = time.perf_counter()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx == next_target:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out_video), fourcc, OUTPUT_FPS, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open writer: {out_video}")
                fg_raw, fg_clean = process_frame(frame, bg_sub, kernel)
                objects = find_moving_objects(fg_clean, MIN_AREA)
                detection = draw_detections(frame.copy(), objects)
                trails, next_id = update_trajectories(
                    objects, trails, next_id, MAX_MATCH_DIST, TRAIL_LENGTH
                )
                detection = draw_trajectories(detection, trails)
                panel = build_three_panel(frame, fg_clean, detection)
                writer.write(panel)
                if saved % SAVE_EVERY == 0:
                    cv2.imwrite(str(OUTPUT_DIR / f"panel_{frame_idx:06d}.png"), panel)
                print(f"  frame {frame_idx:4d}: {len(objects)} object(s)")
                saved += 1
                next_target = round(saved * ratio)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    elapsed = time.perf_counter() - start

    out_dur = saved / OUTPUT_FPS
    print(f"\nSaved {saved} panels @ {OUTPUT_FPS:.0f} fps = {out_dur:.3f}s")
    print(f"Video: {out_video}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
