"""
Feature extraction for the data pipeline (day 27).

Converts detected objects into fixed-length numeric feature vectors so a
video becomes a structured table. The vector for one object is
``[cx, cy, area, mean_b, mean_g, mean_r]`` — position + size + mean color
of the object's foreground pixels.
"""

import cv2
import numpy as np


def extract_color_features(
    frame_bgr: np.ndarray,
    objects: list[dict],
    mask: np.ndarray,
) -> list[list[float]]:
    """
    Compute a 6-dim feature vector per detected object.

    For each object (from ``find_objects``) the vector is
    ``[cx, cy, area, mean_b, mean_g, mean_r]`` where the color means are
    averaged over the object's foreground pixels (the ``mask`` values inside
    its bounding box), not over the whole box.

    Returns a list aligned 1:1 with ``objects``.
    """
    features: list[list[float]] = []
    for obj in objects:
        cx, cy = obj["centroid"]
        area = obj["area"]
        x, y, w, h = obj["bbox"]
        roi = frame_bgr[y : y + h, x : x + w]
        roi_mask = mask[y : y + h, x : x + w]
        # cv2.mean over a mask returns (mean_b, mean_g, mean_r, _)
        mean_b, mean_g, mean_r, _ = cv2.mean(roi, roi_mask)
        features.append([cx, cy, area, mean_b, mean_g, mean_r])
    return features
