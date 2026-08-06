"""Tests for src/vision/roi.py — extracted from day_26_multiroi.py."""

import numpy as np

from src.vision.roi import (
    DEFAULT_ROI_FRACTIONS,
    crop_roi,
    define_roi_config,
    draw_roi_boxes,
)


def test_default_roi_fractions_shape():
    """Three named ROI specs, each a (x_frac, y_frac, w_frac, h_frac) tuple."""
    assert set(DEFAULT_ROI_FRACTIONS) == {"panorama", "work_area", "inlet"}
    for spec in DEFAULT_ROI_FRACTIONS.values():
        assert len(spec) == 4


def test_define_roi_config_defaults():
    cfg = define_roi_config(1000, 800)
    assert cfg["panorama"] == (0, 0, 1000, 800)
    assert cfg["work_area"] == (200, 160, 600, 480)  # central 60%
    assert cfg["inlet"] == (0, 0, 250, 200)  # top-left 25%


def test_define_roi_config_custom_fractions():
    cfg = define_roi_config(100, 100, {"half": (0, 0, 0.5, 0.5)})
    assert cfg["half"] == (0, 0, 50, 50)


def test_define_roi_config_empty_dict_is_honored():
    """An explicitly empty dict must mean 'no ROIs', not fall back to defaults."""
    cfg = define_roi_config(100, 100, {})
    assert cfg == {}


def test_crop_roi_crops_exact_region():
    frame = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
    crop = crop_roi(frame, (10, 20, 30, 40))
    assert crop.shape == (40, 30)
    assert crop[0, 0] == frame[20, 10]
    assert crop[-1, -1] == frame[59, 39]


def test_draw_roi_boxes_paints_rectangles():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cfg = {"a": (10, 10, 30, 30), "b": (50, 50, 20, 20)}
    colors = {"a": (0, 0, 255), "b": (0, 255, 0)}
    out = draw_roi_boxes(frame, cfg, colors)
    assert out.shape == frame.shape
    assert np.array_equal(out[10, 15], (0, 0, 255)), "ROI a border should be red"
    assert np.array_equal(out[50, 55], (0, 255, 0)), "ROI b border should be green"


def test_draw_roi_boxes_default_colors():
    """Without a colors map the helper still paints (cyclic palette)."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cfg = {"a": (10, 10, 30, 30)}
    out = draw_roi_boxes(frame, cfg)
    assert np.any(out > 0), "a red rectangle should have been painted"


def test_draw_roi_boxes_missing_color_key_falls_back():
    """A colors map missing one ROI name must fall back, not raise KeyError."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cfg = {"a": (10, 10, 30, 30), "b": (50, 50, 20, 20)}
    out = draw_roi_boxes(frame, cfg, {"a": (0, 255, 0)})  # "b" has no color
    assert out.shape == frame.shape
    assert np.any(out > 0), "both boxes should still be painted"
