"""Tests for src/vision/background.py — extracted from day_25_motion.py."""

import numpy as np

from src.vision.background import (
    clean_foreground_mask,
    init_background_subtractor,
    process_frame,
)


def test_init_background_subtractor_returns_mog2():
    bg_sub = init_background_subtractor()
    assert hasattr(bg_sub, "apply")


def test_init_background_subtractor_parameters():
    bg_sub = init_background_subtractor(history=100, var_threshold=8, detect_shadows=True)
    assert hasattr(bg_sub, "apply")


def test_clean_foreground_mask_drops_shadows():
    """Value 127 (shadow) must NOT survive as foreground."""
    raw = np.zeros((50, 50), dtype=np.uint8)
    raw[0:10, :] = 255  # true foreground
    raw[10:20, :] = 127  # shadow — must be dropped
    kernel = np.ones((3, 3), np.uint8)
    fg, clean = clean_foreground_mask(raw, kernel)
    assert fg.dtype == np.uint8
    assert fg.ndim == 2
    # shadow band is zero in both outputs
    assert np.all(fg[10:20, :] == 0)
    assert np.all(clean[10:20, :] == 0)
    # true foreground survives
    assert np.all(fg[0:10, :] == 255)


def test_clean_foreground_mask_opens_noise():
    """Morphological OPEN removes small speckles but keeps large blobs."""
    raw = np.zeros((100, 100), dtype=np.uint8)
    raw[20:80, 20:80] = 255  # large blob
    raw[5, 5] = 255  # single-pixel speckle
    kernel = np.ones((7, 7), np.uint8)
    _, clean = clean_foreground_mask(raw, kernel)
    assert np.all(clean[5, 5] == 0), "speckle should be eroded away"
    assert clean[20:80, 20:80].sum() > 0, "large blob should survive"


def _bg_frame(color=120):
    return np.full((100, 100, 3), color, dtype=np.uint8)


def test_process_frame_static_background_goes_black():
    """After warming up on a static scene, a static frame yields an empty mask."""
    bg_sub = init_background_subtractor()
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(10):
        process_frame(_bg_frame(), bg_sub, kernel)
    fg, clean = process_frame(_bg_frame(), bg_sub, kernel)
    assert fg.sum() < 100  # allow a tiny warm-up residue, not a real object


def test_process_frame_detects_moving_object():
    """A bright square that appears after warm-up shows up in the clean mask."""
    bg_sub = init_background_subtractor()
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(10):
        process_frame(_bg_frame(), bg_sub, kernel)
    moving = _bg_frame().copy()
    moving[30:50, 30:50] = (255, 255, 255)
    fg, clean = process_frame(moving, bg_sub, kernel)
    assert clean[30:50, 30:50].sum() > 0
    assert fg[30:50, 30:50].sum() > 0


def test_process_frame_returns_uint8_pair():
    bg_sub = init_background_subtractor()
    kernel = np.ones((3, 3), np.uint8)
    fg, clean = process_frame(_bg_frame(), bg_sub, kernel)
    for m in (fg, clean):
        assert m.dtype == np.uint8
        assert m.ndim == 2
        assert m.shape == (100, 100)
