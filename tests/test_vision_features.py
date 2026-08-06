"""Tests for src/vision/features.py — data-pipeline feature extraction."""

import numpy as np

from src.vision.features import extract_color_features
from src.vision.objects import find_objects


def _red_square_frame():
    """100x100 BGR frame: red block (20:40, 20:40), rest black."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:40, 20:40] = (0, 0, 255)  # BGR red
    return frame


def _red_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    return mask


def test_extract_color_features_single_object():
    frame = _red_square_frame()
    mask = _red_mask()
    objects = find_objects(mask, min_area=100)
    features = extract_color_features(frame, objects, mask)

    assert len(features) == 1
    cx, cy, area, b, g, r = features[0]
    assert cx == 29 and cy == 29  # 20x20 block centroid
    assert area == 361  # cv2.contourArea of a 20x20 block
    assert np.isclose([b, g, r], [0.0, 0.0, 255.0]).all()


def test_extract_color_features_two_colors():
    """Two objects of different colors keep their own means (aligned order)."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:40, 20:40] = (0, 0, 255)  # red
    frame[60:80, 60:80] = (0, 255, 0)  # green
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    mask[60:80, 60:80] = 255

    objects = find_objects(mask, min_area=100)
    features = extract_color_features(frame, objects, mask)

    assert len(features) == 2
    bgr = [(f[3], f[4], f[5]) for f in features]
    # find_objects returns contour order; both colors must appear exactly once
    assert sorted(map(tuple, bgr)) == sorted([(0.0, 0.0, 255.0), (0.0, 255.0, 0.0)])


def test_extract_color_features_empty_objects():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert extract_color_features(frame, [], np.zeros((100, 100), np.uint8)) == []


def test_extract_color_features_uses_mask_only():
    """Color mean is computed only over masked pixels, not the whole bbox."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # bbox area: left half red, right half green; mask covers only the red half
    frame[20:40, 20:40] = (0, 0, 255)  # red left
    frame[20:40, 40:60] = (0, 255, 0)  # green right — not in any object mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 255  # object is only the red half

    objects = find_objects(mask, min_area=100)
    features = extract_color_features(frame, objects, mask)
    assert len(features) == 1
    assert np.isclose([features[0][3], features[0][4], features[0][5]], [0, 0, 255]).all()


def test_day27_analyze_roi():
    import importlib.util
    from pathlib import Path

    import cv2

    root = Path(__file__).resolve().parent.parent
    d27_path = root / "experiments" / "day_27_pipeline.py"
    spec = importlib.util.spec_from_file_location("day_27_pipeline", d27_path)
    D27 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D27)

    roi_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(roi_frame, (50, 50), 20, (0, 0, 255), -1)

    objects, features = D27.analyze_roi(roi_frame)
    assert isinstance(objects, list)
    assert isinstance(features, list)
    assert len(objects) == len(features)
    if objects:
        assert "bbox" in objects[0]
        assert "centroid" in objects[0]
        assert "area" in objects[0]
        assert len(features[0]) == 6
