"""Tests for src/vision/objects.py — extracted from day_25_motion.py."""

import numpy as np

from src.vision.objects import (
    draw_objects,
    draw_trajectories,
    find_objects,
    update_trajectories,
)


def _mask_with_blob(x=20, y=10, w=20, h=20):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    return mask


# ──────────────────── find_objects ────────────────────


def test_find_objects_basic():
    objs = find_objects(_mask_with_blob(), min_area=100)
    assert len(objs) == 1
    o = objs[0]
    assert o["bbox"] == (20, 10, 20, 20)
    # moments centroid of a 20x20 block at x=20..39, y=10..29
    assert o["centroid"] == (29, 19)
    # cv2.contourArea of a 20x20 block follows the outer boundary, so it
    # returns 19x19 = 361 (the border ring is excluded) — cv2's standard quirk
    assert o["area"] == 361


def test_find_objects_filters_small_speckles():
    mask = _mask_with_blob()
    mask[60, 60] = 255  # 1 px speckle
    objs = find_objects(mask, min_area=100)
    assert len(objs) == 1


def test_find_objects_empty_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    assert find_objects(mask, min_area=0) == []


def test_find_objects_returns_three_keys():
    objs = find_objects(_mask_with_blob())
    assert set(objs[0].keys()) == {"bbox", "centroid", "area"}


# ──────────────────── draw_objects ────────────────────


def test_draw_objects_bbox():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    obj = [{"bbox": (10, 10, 20, 20), "centroid": (20, 20), "area": 400}]
    out = draw_objects(frame, obj)
    assert out.shape == frame.shape
    assert np.array_equal(out[10, 15], (0, 255, 0)), "top border should be green"
    assert np.array_equal(out[25, 10], (0, 255, 0)), "left border should be green"


def test_draw_objects_centroid_red_dot():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    obj = [{"bbox": (10, 10, 20, 20), "centroid": (20, 20), "area": 400}]
    out = draw_objects(frame, obj, show_centroid=True)
    assert np.array_equal(out[20, 20], (0, 0, 255)), "centroid dot should be red"


def test_draw_objects_noop_without_objects():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_objects(frame, [])
    assert np.array_equal(out, frame)


# ──────────────────── draw_trajectories ────────────────────


def test_draw_trajectories_short_trail_noop():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    trails = {0: [(0, 0)]}
    assert np.array_equal(draw_trajectories(frame, trails), frame)


def test_draw_trajectories_paints_blue_line():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    trails = {0: [(10, 10), (50, 10), (90, 10)]}
    out = draw_trajectories(frame, trails)
    assert (out[:, :, 0] > 0).any(), "blue channel should be painted"


# ──────────────────── update_trajectories ────────────────────


def test_update_trajectories_matches_and_extends():
    trails = {0: [(0, 0)]}
    objs = [{"centroid": (1, 1)}]
    trails, next_id = update_trajectories(objs, trails, next_id=1, max_dist=10, max_len=5)
    assert trails == {0: [(0, 0), (1, 1)]}
    assert next_id == 1


def test_update_trajectories_spawns_new_trail():
    trails = {}
    objs = [{"centroid": (50, 50)}]
    trails, next_id = update_trajectories(objs, trails, next_id=0, max_dist=10, max_len=5)
    assert trails == {0: [(50, 50)]}
    assert next_id == 1


def test_update_trajectories_drops_stale_trail():
    trails = {0: [(0, 0), (5, 5)]}
    objs = [{"centroid": (100, 100)}]  # far from trail 0 -> new id
    trails, next_id = update_trajectories(objs, trails, next_id=5, max_dist=10, max_len=5)
    assert 0 not in trails, "stale trail must be deleted"
    assert trails[5] == [(100, 100)]
    assert next_id == 6


def test_update_trajectories_trims_to_max_len():
    trails = {0: [(0, 0), (1, 1), (2, 2), (3, 3)]}
    objs = [{"centroid": (4, 4)}]
    trails, _ = update_trajectories(objs, trails, next_id=1, max_dist=10, max_len=3)
    assert trails[0] == [(2, 2), (3, 3), (4, 4)]


def test_update_trajectories_one_trail_one_object():
    """A trail may match at most one object in a frame (no double-use)."""
    trails = {0: [(0, 0)]}
    objs = [{"centroid": (1, 1)}, {"centroid": (2, 2)}]
    trails, next_id = update_trajectories(objs, trails, next_id=7, max_dist=10, max_len=5)
    # only the closest object matched trail 0; the other spawned trail 7
    assert trails[0] == [(0, 0), (1, 1)]
    assert trails[7] == [(2, 2)]
    assert next_id == 8
