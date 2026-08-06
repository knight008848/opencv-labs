"""
Day 24 / 2026-07-31 / Module 11: Video I/O (VideoCapture + VideoWriter)
File: day_24_video.py
Goal: Analyze an MP4 — print metadata, resample to a fixed 24 fps on a
      real-time timeline, overlay frame number + timestamp, re-encode to a new MP4.
Deliverable: annotated frames + annotated video + terminal summary
Runtime: ~6-7 min

Headless note: All visualisation via matplotlib savefig.
See CLAUDE.md for headless policy.

Refactored 2026-08-06: video helpers now live in src/vision/video.py.
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
    annotate_frame,
    create_writer,
    get_video_metadata,
    iter_kept_frames,
    open_video,
)

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "day_24_frames"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── Config ────────────────────────────

# Output video is pinned to a fixed 24 fps (standard playback rate).
# Frames are resampled on a real-time timeline (one frame per 1/OUTPUT_FPS s),
# so output duration matches the source and playback runs at real speed.
OUTPUT_FPS = 24.0

VIDEO_PATH = PROJECT_DIR / "data" / "raw" / "rgb_79c1787d6c.mp4"  # adjust if needed


# ──────────────────── 1. Metadata Extraction ────────────────────


def print_metadata(meta: dict) -> None:
    """
    Pretty-print the metadata dict.
    Include computed duration = frame_count / fps.
    """
    duration = meta["frame_count"] / meta["fps"]
    print(f"Video duration: {duration:.3f}s")
    print(f"Video width: {meta['width']}, height: {meta['height']}")
    print(f"Video fourcc: {meta['fourcc']}")
    print(f"Video FPS: {meta['fps']:.2f}")
    print(f"Video frame count: {meta['frame_count']}")


# ──────────────────── 2. Frame Extraction + Annotation ──────────


def extract_and_annotate(video_path: Path, meta: dict, out_dir: Path, out_video_path: Path) -> int:
    """
    Read the video in a single pass, resampling to OUTPUT_FPS on a real-time
    timeline via src.vision.iter_kept_frames:
      1. Annotate each kept frame with frame number + timestamp
      2. Save as {src_frame_idx:06d}.png in out_dir
      3. Write the annotated frame to the output video (VideoWriter @ OUTPUT_FPS)
    Return the number of annotated frames saved to disk.

    Prints elapsed extraction + annotation time.
    """
    start = time.perf_counter()
    cap = open_video(video_path)
    src_fps = meta["fps"]
    writer = None
    saved = 0
    try:
        for frame_idx, frame, timestamp in iter_kept_frames(cap, src_fps, OUTPUT_FPS):
            annotated = annotate_frame(frame, frame_idx, timestamp)
            cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.png"), annotated)
            if writer is None:  # lazy-init on first kept frame
                h, w = frame.shape[:2]  # derive size from actual frame
                writer = create_writer(out_video_path, OUTPUT_FPS, (w, h))
            writer.write(annotated)
            saved += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    elapsed = time.perf_counter() - start
    print(f"  Extraction + annotation took {elapsed:.2f}s")
    return saved


# ──────────────────────────── Main ──────────────────────────────


def main() -> None:
    """
    Pipeline:
      1. Extract + print video metadata
      2. Single pass: resample to fixed OUTPUT_FPS, annotate, write PNGs + video
      3. Verify saved frame count
    """
    print("\n" + "=" * 55)
    print("  Day 24 — Video I/O Analysis")
    print("=" * 55)

    try:
        # ── Step 1: Metadata ──
        print("\n[1/3] Reading video metadata...")
        meta = get_video_metadata(VIDEO_PATH)
        print_metadata(meta)

        # ── Step 2: Single-pass resample + annotate + assemble ──
        print("\n[2/3] Resampling to fixed 24 fps, annotating, writing video...")
        video_path = PROJECT_DIR / "data" / "processed" / "day_24_annotated.mp4"
        saved = extract_and_annotate(VIDEO_PATH, meta, OUTPUT_DIR, video_path)
        print(f"  Saved {saved} annotated frames to {OUTPUT_DIR}")
        print(f"  Saved video: {video_path}")

        # ── Step 3: Verify count + real-time duration ──
        # HEVC CAP_PROP_FRAME_COUNT is an estimate (±1 frame), so verify the
        # outcome by duration: the resampled video must play at real-time pace.
        out_dur = saved / OUTPUT_FPS
        src_dur = meta["frame_count"] / meta["fps"]
        status = "real-time pace OK" if abs(out_dur - src_dur) < 0.5 else "speed changed!"
        print(f"\n[3/3] Verification: saved {saved} frames @ {OUTPUT_FPS:.0f} fps")
        print(f"    output {out_dur:.3f}s vs source {src_dur:.3f}s → {status}")

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        return

    print("\n" + "=" * 55)
    print("  Done — check data/processed/day_24_frames/")
    print("=" * 55)


if __name__ == "__main__":
    main()
