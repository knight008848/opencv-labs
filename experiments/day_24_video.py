"""
Day 24 / 2026-07-31 / Module 11: Video I/O (VideoCapture + VideoWriter)
File: day_24_video.py
Goal: Analyze an MP4 — print metadata, resample to a fixed 24 fps on a
      real-time timeline, overlay frame number + timestamp, re-encode to a new MP4.
Deliverable: annotated frames + annotated video + terminal summary
Runtime: ~6-7 min

Headless note: All visualisation via matplotlib savefig.
See CLAUDE.md for headless policy.
"""

import time
from pathlib import Path

import cv2
import numpy as np

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


def get_video_metadata(video_path: Path) -> dict:
    """
    Open the video with cv2.VideoCapture and extract:
      - FPS          (CAP_PROP_FPS)
      - frame count  (CAP_PROP_FRAME_COUNT)
      - width        (CAP_PROP_FRAME_WIDTH)
      - height       (CAP_PROP_FRAME_HEIGHT)
      - fourcc       (CAP_PROP_FOURCC) → decoded to 4-char string
    Return a dict. Handle the case where the file doesn't exist.
    """
    # ── Your implementation here ──
    # Hint:
    #   if not video_path.exists(): raise FileNotFoundError(...)
    #   cap = cv2.VideoCapture(str(video_path))
    #   if not cap.isOpened(): raise RuntimeError(...)
    #   fps = cap.get(cv2.CAP_PROP_FPS)
    #   ...
    #   fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    #   fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    #   cap.release()
    #   return {...}
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fourcc": fourcc_str,
    }


def print_metadata(meta: dict) -> None:
    """
    Pretty-print the metadata dict.
    Include computed duration = frame_count / fps.
    """
    # ── Your implementation here ──
    duration = meta["frame_count"] / meta["fps"]
    print(f"Video duration: {duration:.3f}s")
    print(f"Video width: {meta['width']}, height: {meta['height']}")
    print(f"Video fourcc: {meta['fourcc']}")
    print(f"Video FPS: {meta['fps']:.2f}")
    print(f"Video frame count: {meta['frame_count']}")


# ──────────────────── 2. Frame Extraction + Annotation ──────────


def annotate_frame(frame: np.ndarray, frame_idx: int, timestamp: float) -> np.ndarray:
    """
    Draw frame number + timestamp text on the frame (top-left corner).
    Use cv2.putText with a readable size.
    Timestamp format: f"{timestamp:.3f}s"
    Return annotated frame.
    """
    # ── Your implementation here ──
    # Hint: cv2.putText(frame, text, org, font, scale, color, thickness)
    #       text = f"frame={frame_idx:05d}  t={timestamp:.3f}s"
    text = f"frame={frame_idx:05d}  t={timestamp:.3f}s"
    frame = cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


def extract_and_annotate(video_path: Path, meta: dict, out_dir: Path, out_video_path: Path) -> int:
    """
    Read the video in a single pass, resampling to OUTPUT_FPS on a real-time
    timeline (keep one frame per 1/OUTPUT_FPS s of source time):
      1. Compute timestamp = src_frame_idx / src_fps
      2. Annotate with frame number + timestamp
      3. Save as {src_frame_idx:06d}.png in out_dir
      4. Write the annotated frame to the output video (VideoWriter @ OUTPUT_FPS)
    Return the number of annotated frames saved to disk.

    Prints elapsed extraction + annotation time.
    """
    # ── Your implementation here ──
    # Hint (single pass — read, annotate, write PNG + video frame):
    #   cap = cv2.VideoCapture(str(video_path))
    #   src_fps = meta["fps"]
    #   ratio = src_fps / OUTPUT_FPS          # source frames per output frame (60/24=2.5)
    #   frame_idx = 0; saved = 0; next_target = 0; writer = None
    #   while True:
    #       ret, frame = cap.read()
    #       if not ret: break
    #       if frame_idx == next_target:      # keep the frame closest to this output tick
    #           if writer is None:            # lazy-init on first kept frame
    #               h, w = frame.shape[:2]    # derive size from actual frame
    #               writer = cv2.VideoWriter(str(out_video_path), fourcc, OUTPUT_FPS, (w, h))
    #           timestamp = frame_idx / src_fps   # real timeline time
    #           annotated = annotate_frame(frame, frame_idx, timestamp)
    #           cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.png"), annotated)
    #           writer.write(annotated)
    #           saved += 1
    #           next_target = round(saved * ratio)   # advance to next output tick
    #       frame_idx += 1
    #   cap.release(); writer.release()
    #   return saved
    start = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    src_fps = meta["fps"]
    ratio = src_fps / OUTPUT_FPS  # source frames per output frame (60/24 = 2.5)
    frame_idx = 0  # current source frame index
    saved = 0
    next_target = 0  # next source frame index to keep
    writer = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx == next_target:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out_video_path), fourcc, OUTPUT_FPS, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer: {out_video_path}")
                timestamp = frame_idx / src_fps  # real timeline time
                annotated = annotate_frame(frame, frame_idx, timestamp)
                cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.png"), annotated)
                writer.write(annotated)
                saved += 1
                next_target = round(saved * ratio)  # next keep = round(out_k * ratio)
            frame_idx += 1
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
