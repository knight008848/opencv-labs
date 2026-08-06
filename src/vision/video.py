"""
Video I/O helpers extracted from day_24_video.py.

Provides safe open/metadata/write helpers plus a real-time frame-grid
iterator so any script can resample a source clip to a fixed output fps
(one frame per 1/output_fps s of source time) in a single pass.
"""

from pathlib import Path

import cv2
import numpy as np

DEFAULT_FOURCC = "mp4v"


def open_video(video_path: Path | str) -> cv2.VideoCapture:
    """Open a video file with error handling; raise if missing or unreadable."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
    return cap


def get_video_metadata(video_path: Path | str) -> dict:
    """
    Open the video and extract fps / frame_count / width / height / fourcc.

    The caller is responsible for the returned dict only; the capture is
    opened and released inside this function.
    """
    cap = open_video(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    finally:
        cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fourcc": fourcc_str,
    }


def even_size(width: int, height: int) -> tuple[int, int]:
    """Round width/height down to even numbers.

    mp4v (YUV 4:2:0) rejects odd frame sizes, so every VideoWriter size
    should pass through here first.
    """
    return width - width % 2, height - height % 2


def create_writer(
    out_path: Path | str,
    fps: float,
    size: tuple[int, int],
    fourcc: str = DEFAULT_FOURCC,
) -> cv2.VideoWriter:
    """Create an opened VideoWriter; raise a RuntimeError if it fails to open."""
    out_path = Path(out_path)
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(out_path), fourcc_code, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")
    return writer


def iter_kept_frames(cap: cv2.VideoCapture, src_fps: float, output_fps: float):
    """
    Single-pass generator over a capture resampled to a fixed output fps.

    Yields ``(src_frame_idx, frame, timestamp)`` for the frames kept on a
    real-time timeline: the k-th kept frame is the source frame closest to
    ``t = k / output_fps`` seconds. This mirrors the day 24 / day 25
    ``next_target = round(saved * ratio)`` grid so output duration matches
    the source and playback runs at real speed. The grid index is clamped to
    strictly increase, so upsampling (output_fps > src_fps) yields every
    source frame instead of stalling.

    Streaming by design: frames are yielded one at a time and dropped, so a
    long source never materialises in memory.
    """
    if src_fps <= 0:
        raise ValueError(f"src_fps must be positive, got {src_fps}")
    ratio = src_fps / output_fps  # source frames per output frame
    frame_idx = 0
    saved = 0
    next_target = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == next_target:
            timestamp = frame_idx / src_fps  # real timeline time
            yield frame_idx, frame, timestamp
            saved += 1
            next_target = max(next_target + 1, round(saved * ratio))
        frame_idx += 1


def annotate_frame(frame: np.ndarray, frame_idx: int, timestamp: float) -> np.ndarray:
    """Draw frame number + timestamp text in the top-left corner.

    Timestamp format is ``f"{timestamp:.3f}s"``. Mutates the input frame in
    place and returns it (same convention as the other draw_* helpers).
    """
    text = f"frame={frame_idx:05d}  t={timestamp:.3f}s"
    return cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
