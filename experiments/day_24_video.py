"""
Day 24 / 2026-07-31 / Module 11: Video I/O (VideoCapture + VideoWriter)
File: day_24_video.py
Goal: Analyze an MP4 — print metadata, extract every 5th frame,
      overlay frame number + timestamp, re-encode to a new MP4.
Deliverable: annotated frames + annotated video + terminal summary
Runtime: ~2 min

Headless note: All visualisation via matplotlib savefig.
See CLAUDE.md for headless policy.
"""

import math
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

SAMPLE_INTERVAL = 5  # save every 5th frame

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


def extract_and_annotate(video_path: Path, meta: dict, out_dir: Path) -> list[np.ndarray]:
    """
    Read the video, every SAMPLE_INTERVAL-th frame:
      1. Compute timestamp = frame_idx / fps
      2. Annotate with frame number + timestamp
      3. Save as {frame_idx:06d}.png in out_dir
      4. Append to list of annotated frames
    Return the list of annotated frames (for video assembly).

    Prints elapsed extraction + annotation time.
    """
    # ── Your implementation here ──
    # Hint:
    #   cap = cv2.VideoCapture(str(video_path))
    #   annotated_frames = []
    #   frame_idx = 0
    #   while True:
    #       ret, frame = cap.read()
    #       if not ret: break
    #       if frame_idx % SAMPLE_INTERVAL == 0:
    #           timestamp = frame_idx / meta["fps"]
    #           annotated = annotate_frame(frame, frame_idx, timestamp)
    #           cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.png"), annotated)
    #           annotated_frames.append(annotated)
    #       frame_idx += 1
    #   cap.release()
    #   return annotated_frames
    start = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    annotated_frames = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % SAMPLE_INTERVAL == 0:
                timestamp = frame_idx / meta["fps"]
                annotated = annotate_frame(frame, frame_idx, timestamp)
                cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.png"), annotated)
                annotated_frames.append(annotated)
            frame_idx += 1
    finally:
        cap.release()
    elapsed = time.perf_counter() - start
    print(f"  Extraction + annotation took {elapsed:.2f}s")
    return annotated_frames


# ──────────────────── 3. Video Assembly ─────────────────────────


def assemble_video(frames: list[np.ndarray], meta: dict, out_path: Path) -> None:
    """
    Combine annotated frames into a new MP4 using VideoWriter.
    - fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    - FPS = meta["fps"]  (source speed)
    - Size = (w, h) from frames[0].shape — derived from actual frames
    Print the number of frames written and total time.
    """
    # ── Your implementation here ──
    # Hint:
    #   h, w = frames[0].shape[:2]
    #   writer = cv2.VideoWriter(str(out_path), fourcc, meta["fps"], (w, h))
    #   for frame in frames: writer.write(frame)
    #   writer.release()
    #   Verify output file exists and is non-zero size.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), fourcc, meta["fps"], (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write video to {out_path}")


# ──────────────────────────── Main ──────────────────────────────


def main() -> None:
    """
    Pipeline:
      1. Extract + print video metadata
      2. Extract every 5th frame, annotate, save PNGs
      3. Assemble annotated frames into new MP4
      4. Print summary (frames processed, elapsed time)
    """
    print("\n" + "=" * 55)
    print("  Day 24 — Video I/O Analysis")
    print("=" * 55)

    try:
        # ── Step 1: Metadata ──
        print("\n[1/4] Reading video metadata...")
        meta = get_video_metadata(VIDEO_PATH)
        print_metadata(meta)

        # ── Step 2: Extract + annotate frames ──
        print("\n[2/4] Extracting every 5th frame and annotating...")
        frames = extract_and_annotate(VIDEO_PATH, meta, OUTPUT_DIR)
        print(f"  Saved {len(frames)} annotated frames to {OUTPUT_DIR}")

        # ── Step 3: Assemble video ──
        print("\n[3/4] Assembling annotated video...")
        video_path = PROJECT_DIR / "data" / "processed" / "day_24_annotated.mp4"
        assemble_video(frames, meta, video_path)
        print(f"  Saved: {video_path}")

        # ── Step 4: Verify count ──
        expected = math.ceil(meta["frame_count"] / SAMPLE_INTERVAL)
        print(f"\n[4/4] Verification: saved {len(frames)} frames, expected {expected}")

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        return

    print("\n" + "=" * 55)
    print("  Done — check data/processed/day_24_frames/")
    print("=" * 55)


if __name__ == "__main__":
    main()
