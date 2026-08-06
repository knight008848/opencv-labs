"""Tests for src/vision/video.py — extracted from day_24_video.py.

Synthetic video fixtures keep the suite hermetic (no data/raw dependency).
"""

import numpy as np
import pytest

from src.vision.video import (
    annotate_frame,
    create_writer,
    even_size,
    get_video_metadata,
    iter_kept_frames,
    open_video,
)


def make_test_video(path, fps=30.0, n_frames=10, width=320, height=240):
    """Write a small solid-color mp4 so tests don't depend on data/raw."""
    writer = cv2_create_writer(path, fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), i, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def cv2_create_writer(path, fps, size):
    import cv2

    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


# ──────────────────── open_video ────────────────────


def test_open_video_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_video(tmp_path / "nope.mp4")


def test_open_video_opens_real_file(tmp_path):
    path = tmp_path / "v.mp4"
    make_test_video(path)
    cap = open_video(path)
    try:
        assert cap.isOpened()
    finally:
        cap.release()


# ──────────────────── get_video_metadata ────────────────────


def test_get_video_metadata(tmp_path):
    path = tmp_path / "v.mp4"
    make_test_video(path, fps=30.0, n_frames=10, width=320, height=240)
    meta = get_video_metadata(path)
    assert meta["fps"] == pytest.approx(30.0)
    assert meta["frame_count"] == 10
    assert meta["width"] == 320
    assert meta["height"] == 240
    # fourcc is a 4-char codec tag; only verify it decodes to 4 chars
    assert len(meta["fourcc"]) == 4


def test_get_video_metadata_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_video_metadata(tmp_path / "nope.mp4")


# ──────────────────── even_size ────────────────────


def test_even_size():
    assert even_size(320, 240) == (320, 240)
    assert even_size(321, 241) == (320, 240)
    assert even_size(1, 1) == (0, 0)


# ──────────────────── create_writer ────────────────────


def test_create_writer_creates_file(tmp_path):
    path = tmp_path / "out.mp4"
    writer = create_writer(path, 24.0, (320, 240))
    try:
        assert writer.isOpened()
    finally:
        writer.release()
    assert path.exists()


# ──────────────────── iter_kept_frames ────────────────────


def test_iter_kept_frames_sampling_grid(tmp_path):
    """30fps source resampled to 24fps keeps every frame at round(k*ratio)."""
    path = tmp_path / "v.mp4"
    make_test_video(path, fps=30.0, n_frames=10, width=320, height=240)
    cap = cv2_VideoCapture(path)
    try:
        kept = list(iter_kept_frames(cap, src_fps=30.0, output_fps=24.0))
    finally:
        cap.release()

    kept_indices = [idx for idx, _, _ in kept]
    # ratio = 30/24 = 1.25 → round(0, 1.25, 2.5, 3.75, ...) = 0,1,2,4,5,6,8,9
    assert kept_indices == [0, 1, 2, 4, 5, 6, 8, 9]
    # timestamp is source-frame index / source fps (real timeline)
    for idx, frame, timestamp in kept:
        assert timestamp == pytest.approx(idx / 30.0)
        assert frame.shape == (240, 320, 3)


def test_iter_kept_frames_upsample_yields_all_frames(tmp_path):
    """10fps -> 30fps upsample must not stall: every source frame is yielded."""
    path = tmp_path / "v.mp4"
    make_test_video(path, fps=10.0, n_frames=3, width=320, height=240)
    cap = cv2_VideoCapture(path)
    try:
        kept = list(iter_kept_frames(cap, src_fps=10.0, output_fps=30.0))
    finally:
        cap.release()
    assert [idx for idx, _, _ in kept] == [0, 1, 2]
    for idx, _, timestamp in kept:
        assert timestamp == pytest.approx(idx / 10.0)


def test_iter_kept_frames_rejects_nonpositive_fps(tmp_path):
    """A zero/negative src_fps must fail loudly, not silently stall."""
    path = tmp_path / "v.mp4"
    make_test_video(path, fps=10.0, n_frames=3, width=320, height=240)
    cap = cv2_VideoCapture(path)
    try:
        with pytest.raises(ValueError):
            list(iter_kept_frames(cap, src_fps=0.0, output_fps=30.0))
    finally:
        cap.release()


# ──────────────────── annotate_frame ────────────────────


def test_annotate_frame_preserves_shape():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = annotate_frame(frame, 42, 1.375)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    # red text (BGR channel 2) must have been painted near the top-left
    assert (out[:, :, 2] > 0).sum() > 0


def cv2_VideoCapture(path):
    import cv2

    return cv2.VideoCapture(str(path))
