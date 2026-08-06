"""Tests for src/vision/panel.py — extracted from day_25/26 panel builders."""

import numpy as np

from src.vision.panel import fit_to_height, stack_panels


def _bgr(w, h, value):
    return np.full((h, w, 3), value, dtype=np.uint8)


# ──────────────────── fit_to_height ────────────────────


def test_fit_to_height_preserves_aspect_ratio():
    w, h = fit_to_height(np.zeros((360, 640, 3), dtype=np.uint8), 480)
    # 640 * 480/360 = 853.33 -> 853 -> even 852
    assert (w, h) == (852, 480)


def test_fit_to_height_returns_even_dims():
    w, h = fit_to_height(np.zeros((101, 99, 3), dtype=np.uint8), 240)
    assert w % 2 == 0 and h % 2 == 0


# ──────────────────── stack_panels ────────────────────


def test_stack_panels_horizontal_row():
    panels = [_bgr(40, 30, 50), _bgr(40, 30, 100), _bgr(40, 30, 150)]
    grid = stack_panels(panels, cols=3, display_size=(40, 30))
    assert grid.shape == (30, 120, 3)


def test_stack_panels_two_by_two():
    panels = [_bgr(40, 30, v) for v in range(4)]
    grid = stack_panels(panels, cols=2, display_size=(40, 30))
    assert grid.shape == (60, 80, 3)


def test_stack_panels_pads_missing_cells():
    panels = [_bgr(40, 30, 200)]
    grid = stack_panels(panels, cols=2, display_size=(40, 30))
    assert grid.shape == (30, 80, 3)
    # the padded right cell stays black
    assert np.all(grid[:, 40:] == 0)


def test_stack_panels_grayscale_auto_bgr():
    gray = np.full((30, 40), 128, dtype=np.uint8)
    grid = stack_panels([gray], cols=1, display_size=(40, 30))
    assert grid.ndim == 3
    assert grid.shape[2] == 3


def test_stack_panels_even_display_size():
    panels = [_bgr(41, 31, 10)]
    grid = stack_panels(panels, cols=1, display_size=(41, 31))
    assert grid.shape[1] % 2 == 0 and grid.shape[0] % 2 == 0


def test_stack_panels_labels_painted():
    panels = [_bgr(40, 30, 0)]
    labeled = stack_panels(panels, cols=1, display_size=(40, 30), labels=["alpha"])
    assert np.any(labeled > 0), "label text should be painted (red by default)"


def test_stack_panels_empty_raises():
    try:
        stack_panels([], cols=1, display_size=(40, 30))
    except ValueError:
        pass
    else:
        raise AssertionError("empty panel list should raise ValueError")
