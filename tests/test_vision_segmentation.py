"""Tests for src/vision/segmentation.py — extracted from day_26_multiroi.py."""

import numpy as np

from src.vision.segmentation import RED_RANGES, detect_hsv_objects, hsv_mask


def _red_frame():
    """100x100 BGR frame with a red block at (20:40, 20:40)."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:40, 20:40] = (0, 0, 255)  # BGR red
    return frame


# ──────────────────── hsv_mask ────────────────────


def test_hsv_mask_detects_red_block():
    mask = hsv_mask(_red_frame(), RED_RANGES)
    assert mask.shape == (100, 100)
    assert mask[20:40, 20:40].sum() > 0
    assert mask[60:80, 60:80].sum() == 0, "non-red region must be empty"


def test_hsv_mask_ignores_non_target_colors():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:40, 20:40] = (0, 255, 0)  # BGR green — not in RED_RANGES
    mask = hsv_mask(frame, RED_RANGES)
    assert mask.sum() == 0


def test_hsv_mask_close_merges_nearby_blobs():
    """Close (kernel 9) bridges the 5 px gap between two red disks."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[45:55, 20:35] = (0, 0, 255)
    frame[45:55, 40:55] = (0, 0, 255)  # 5 px gap between x=35 and x=40
    mask = hsv_mask(frame, RED_RANGES, close_kernel=9)
    assert mask[45:55, 36:39].sum() > 0, "gap should be closed by the kernel"


def test_hsv_mask_open_removes_speckle():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[50, 50] = (0, 0, 255)  # single red pixel
    mask = hsv_mask(frame, RED_RANGES, open_kernel=3)
    assert mask[50, 50] == 0, "isolated pixel should be eroded away"


# ──────────────────── detect_hsv_objects ────────────────────


def test_detect_hsv_objects_finds_one_object():
    objs = detect_hsv_objects(_red_frame(), RED_RANGES, min_area_fraction=0.0)
    assert len(objs) == 1
    assert set(objs[0].keys()) == {"bbox", "centroid", "area"}
    assert objs[0]["bbox"] == (20, 20, 20, 20)


def test_detect_hsv_objects_area_fraction_filters_small():
    """A tiny red block under the ROI-area fraction must be dropped."""
    frame = _red_frame()
    frame[80, 80] = (0, 0, 255)  # 1 px red speckle
    objs = detect_hsv_objects(frame, RED_RANGES, min_area_fraction=0.001)
    assert len(objs) == 1  # only the big block survives
