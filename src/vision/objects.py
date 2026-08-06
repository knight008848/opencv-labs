"""
Detection, drawing and tracking helpers extracted from day_25_motion.py.

The object schema used across modules is a plain dict:
    {"bbox": (x, y, w, h), "centroid": (cx, cy), "area": float}
find_objects produces it; draw_objects renders it; update_trajectories
associates it across frames.
"""

import cv2
import numpy as np


def find_objects(mask: np.ndarray, min_area: float = 0) -> list[dict]:
    """
    Find external contours in a binary mask, dropping blobs below min_area.

    Returns a list of {"bbox", "centroid", "area"} — one entry per object.
    Degenerate contours (zero-area moments) are skipped.
    """
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    objects = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue  # degenerate contour — cannot compute a centroid
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        bbox = cv2.boundingRect(cnt)
        objects.append({"bbox": bbox, "centroid": centroid, "area": area})
    return objects


def draw_objects(
    frame: np.ndarray,
    objects: list[dict],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_centroid: bool = False,
    show_id: bool = False,
) -> np.ndarray:
    """
    Draw a bounding rectangle per object, plus optional decorations.

    - ``show_centroid`` paints a red dot at the object centroid
    - ``show_id`` paints a "#idx" label above the box (idx = list position)
    Mutates and returns the input frame.
    """
    for idx, obj in enumerate(objects):
        x, y, w, h = obj["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        if show_centroid:
            cx, cy = obj["centroid"]
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        if show_id:
            cv2.putText(
                frame,
                f"#{idx}",
                (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
    return frame


def draw_trajectories(frame: np.ndarray, trails: dict[int, list[tuple[int, int]]]) -> np.ndarray:
    """
    Draw each trail as connected blue line segments.

    Trails with fewer than 2 points draw nothing -> no ghost line on first sight.
    Mutates and returns the input frame.
    """
    for pts in trails.values():
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts)], isClosed=False, color=(255, 0, 0), thickness=2)
    return frame


def update_trajectories(
    objects: list[dict],
    trails: dict[int, list[tuple[int, int]]],
    next_id: int,
    max_dist: float,
    max_len: int,
) -> tuple[dict[int, list[tuple[int, int]]], int]:
    """
    Associate current centroids with existing trails via greedy
    nearest-neighbour matching, then grow / trim / spawn / drop trails.

      - matched centroid -> append to that trail, trim to max_len
      - unmatched centroid -> new trail with a fresh id
      - old trail with no match this frame -> deleted (left the scene)

    A trail is matched by at most one object per frame. Returns
    (updated trails, next available id).
    """
    preexisting = set(trails.keys())  # trails that existed before this frame
    matched_trails = set()
    for obj in objects:
        cx, cy = obj["centroid"]
        min_dist = float("inf")
        min_trail = None
        for trail_id, trail in trails.items():
            if trail_id in matched_trails:
                continue  # one trail may match at most one object this frame
            if len(trail) == 0:
                continue
            last_pt = trail[-1]
            dist = np.linalg.norm(np.array((cx, cy)) - np.array(last_pt))
            if dist < min_dist:
                min_dist = dist
                min_trail = trail_id
        if min_dist <= max_dist:
            matched_trails.add(min_trail)
            trails[min_trail].append((cx, cy))
            trails[min_trail] = trails[min_trail][-max_len:]
        else:
            trails[next_id] = [(cx, cy)]
            next_id += 1
    for trail_id in list(trails.keys()):  # snapshot keys before deleting
        if trail_id not in matched_trails and trail_id in preexisting:
            del trails[trail_id]
    return trails, next_id
