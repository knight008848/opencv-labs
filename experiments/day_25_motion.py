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

Refactored 2026-08-06: detection / tracking / panel helpers now live in
src/vision.
"""

import sys
import time
from pathlib import Path

import cv2

# Ensure project root is on sys.path so src/ imports resolve
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.vision import (  # noqa: E402
    create_writer,
    draw_objects,
    draw_trajectories,
    find_objects,
    fit_to_height,
    init_background_subtractor,
    iter_kept_frames,
    open_video,
    process_frame,
    stack_panels,
    update_trajectories,
)

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "day_25_frames"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── Config ────────────────────────────

# Rolling-ball clip: one ball + an occasional hand in the middle of the video.
# The camera is not fully static (per-pixel std analysis shows ~99% of pixels
# vary over time), so the foreground mask is expected to flood while the hand
# is in view — that failure mode is exactly what today's exercise explores.
VIDEO_PATH = PROJECT_DIR / "data" / "raw" / "滚动球.mp4"

OUTPUT_FPS = 24.0  # pinned output frame rate (same convention as day 24)
MIN_AREA = 8000  # foreground blobs below this area are dropped; tuned so the
# ball (~12k px, median of the largest blob per frame in the 5-10 s window)
# survives while 50% of sub-200 px speckles are killed
TRAIL_LENGTH = 30  # keep the last N centroids per tracked object
MAX_MATCH_DIST = 60  # centroids farther apart than this are not the same object
SAVE_EVERY = 60  # export one sample three-panel PNG every N output frames


def build_three_panel(
    original: cv2.typing.MatLike,
    fg_mask_clean: cv2.typing.MatLike,
    detection: cv2.typing.MatLike,
):
    """
    Stack three same-size panels horizontally: original | mask | detection.
    Resize each panel to a common height (even dims for mp4v) first; label
    each region. Returns the combined BGR frame.
    """
    display = fit_to_height(original, 480)
    return stack_panels(
        [original, fg_mask_clean, detection],
        cols=3,
        display_size=display,
        labels=["source", "mask", "detection"],
    )


def main() -> None:
    """
    Pipeline:
      1. Open video, init MOG2 + morphology kernel + trail store
      2. Single pass on the 24 fps grid: for each kept source frame
         process_frame -> find_objects -> draw_objects -> update_trajectories
         -> draw_trajectories -> build_three_panel -> write
      3. Every SAVE_EVERY frames, export a sample three-panel PNG
      4. Print per-frame object count + elapsed time
    """
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    cap = open_video(VIDEO_PATH)
    src_fps = cap.get(cv2.CAP_PROP_FPS)

    out_video = PROJECT_DIR / "data" / "processed" / "day_25_annotated.mp4"

    bg_sub = init_background_subtractor()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    trails: dict[int, list[tuple[int, int]]] = {}
    next_id = 0
    writer = None

    saved = 0
    start = time.perf_counter()
    try:
        for frame_idx, frame, _timestamp in iter_kept_frames(cap, src_fps, OUTPUT_FPS):
            _raw_mask, fg_clean = process_frame(frame, bg_sub, kernel)
            objects = find_objects(fg_clean, MIN_AREA)
            detection = draw_objects(frame.copy(), objects, show_centroid=True)
            trails, next_id = update_trajectories(
                objects, trails, next_id, MAX_MATCH_DIST, TRAIL_LENGTH
            )
            detection = draw_trajectories(detection, trails)
            panel = build_three_panel(frame, fg_clean, detection)
            if writer is None:
                # writer size must match the three-panel layout, not the raw frame
                h, w = panel.shape[:2]
                writer = create_writer(out_video, OUTPUT_FPS, (w, h))
            writer.write(panel)
            if saved % SAVE_EVERY == 0:
                cv2.imwrite(str(OUTPUT_DIR / f"panel_{frame_idx:06d}.png"), panel)
            print(f"  frame {frame_idx:4d}: {len(objects)} object(s)")
            saved += 1
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
