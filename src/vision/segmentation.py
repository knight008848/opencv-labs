"""
HSV color segmentation extracted from day_26_multiroi.py.

Hue wraps around 0°, so colors like red need TWO hue bands; the ranges are
OR-ed together into one mask. Morphology then closes small gaps and drops
salt-pixel noise before contours are extracted.
"""

import cv2
import numpy as np

from src.vision.objects import find_objects

# Red target (e.g. the rolling-ball): two hue bands straddling 0°.
RED_RANGES = [
    ((0, 80, 60), (10, 255, 255)),
    ((170, 80, 60), (180, 255, 255)),
]


def hsv_mask(
    frame_bgr: np.ndarray,
    ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    close_kernel: int = 0,
    open_kernel: int = 0,
) -> np.ndarray:
    """
    BGR -> HSV -> OR of inRange over ``ranges`` -> optional morphology.

    A kernel size of 0 disables that morphology step. Returns a uint8 mask
    where matched pixels are 255.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, lower, upper)
    if close_kernel > 0:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8)
        )
    if open_kernel > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8))
    return mask


def detect_hsv_objects(
    frame_bgr: np.ndarray,
    ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    min_area_fraction: float = 0.0,
    close_kernel: int = 0,
    open_kernel: int = 0,
) -> list[dict]:
    """
    Segment ``frame_bgr`` by hue and extract the objects inside it.

    A relative ``min_area_fraction`` (of the frame's own area) suits scenes
    where ROIs / object sizes vary — the same threshold stays meaningful
    across windows of very different sizes. Returns the ``find_objects``
    schema: list of {"bbox", "centroid", "area"}.
    """
    mask = hsv_mask(frame_bgr, ranges, close_kernel, open_kernel)
    min_area = min_area_fraction * frame_bgr.shape[0] * frame_bgr.shape[1]
    return find_objects(mask, min_area)
