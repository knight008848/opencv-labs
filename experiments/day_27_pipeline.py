"""
Day 27 / 2026-08-06 / Data Pipeline v0.1 (skeleton — implement the stages)
File: day_27_pipeline.py
Goal: MP4 -> structured observations:
      1. Frame sampling: keep every FRAME_STEP-th frame
      2. Multi-ROI analysis: run the same detection pipeline per ROI
      3. Cross-frame tracking: match objects per ROI (centroid < 50 px)
      4. Output: output.json (per-frame per-ROI detections),
         features.npz ([cx, cy, area, b, g, r] per object per frame),
         summary.txt (timeline of object appear/disappear frames)
      5. Visualization: annotated video (detection boxes + trajectory lines)
Deliverable: output.json + features.npz + summary.txt + annotated video
Runtime: depends on source

Headless note: no imshow / waitKey. Visualization via VideoWriter.
See CLAUDE.md for headless policy.

Refactoring note: the building blocks (sampler, ROI, detection, features,
tracking, drawing) live in src/vision — this script orchestrates them and
only you fill in the per-stage glue below the "Your implementation here"
markers.
"""

import json
import sys
import time
from pathlib import Path

import cv2  # noqa: F401 — used in render_frame
import numpy as np

# Ensure project root is on sys.path so src/ imports resolve
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# noqa: E402, F401 — the "unused" imports below are the tools the stages
# you fill in are expected to call; they are listed up front so you know
# what src.vision offers instead of hunting for it.
from src.vision import (  # noqa: E402, F401
    RED_RANGES,
    create_writer,
    crop_roi,
    define_roi_config,
    draw_objects,
    draw_roi_boxes,
    draw_trajectories,
    even_size,
    extract_color_features,
    find_objects,
    get_video_metadata,
    hsv_mask,
    iter_every_n_frames,
    open_video,
    update_trajectories,
)

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "day_27"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── Config ────────────────────────────

VIDEO_PATH = PROJECT_DIR / "data" / "raw" / "滚动球.mp4"  # adjust if needed

FRAME_STEP = 5  # keep every N-th frame (configurable, default 5)
MIN_AREA_FRACTION = 0.0015  # blobs >= this fraction of the ROI area survive
CLOSE_KERNEL = 15  # HSV mask: close gaps so the ball is one solid blob
OPEN_KERNEL = 3  # HSV mask: drop isolated salt noise
MAX_MATCH_DIST = 50  # centroids closer than this are the same object (px)
TRAIL_LENGTH = 30  # keep the last N centroid points per tracked object
OUTPUT_FPS = 24.0  # visualization video frame rate

# ROI name -> draw color (BGR)
ROI_COLORS = {
    "panorama": (0, 0, 255),  # red  — full frame
    "work_area": (0, 255, 0),  # green — central 60%
    "inlet": (255, 0, 0),  # blue  — top-left 25%
}


# ──────────────────── Stage 1: Per-ROI analysis ─────────────────


def analyze_roi(roi_frame: np.ndarray) -> tuple[list[dict], list[list[float]]]:
    """
    Detection + feature extraction inside one ROI (ROI-local coordinates).

    Returns ``(objects, features)``:
      objects:  list of {"bbox": (x, y, w, h), "centroid": (cx, cy), "area": float}
      features: list of [cx, cy, area, mean_b, mean_g, mean_r] — one vector
                per object, aligned with ``objects``.
    """
    mask = hsv_mask(roi_frame, RED_RANGES, CLOSE_KERNEL, OPEN_KERNEL)
    min_area = MIN_AREA_FRACTION * roi_frame.shape[0] * roi_frame.shape[1]
    objects = find_objects(mask, min_area)
    features = extract_color_features(roi_frame, objects, mask)
    return objects, features


# ──────────────── Stage 2: Frame rendering (visualization) ───────


def render_frame(
    frame: np.ndarray,
    roi_config: dict[str, tuple[int, int, int, int]],
    roi_objects: dict[str, list[dict]],
    state: dict,
    roi_colors: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    """
    Annotate a full frame: ROI boxes + per-object detection boxes + trails.

    Detection coordinates inside a ROI are local to that ROI; they must be
    shifted by the ROI origin (x, y) before being drawn on the full frame.
    Returns the annotated BGR frame (mutates a copy of the input).
    """
    vis = draw_roi_boxes(frame.copy(), roi_config, roi_colors)

    for name, objects in roi_objects.items():
        ox, oy, _rw, _rh = roi_config[name]

        shifted_objects = []
        for obj in objects:
            shifted_obj = obj.copy()
            x, y, w, h = obj["bbox"]
            shifted_obj["bbox"] = (x + ox, y + oy, w, h)
            cx, cy = obj["centroid"]
            shifted_obj["centroid"] = (cx + ox, cy + oy)
            shifted_objects.append(shifted_obj)

        vis = draw_objects(vis, shifted_objects, show_id=True)

        trails = state[name]["trails"]
        shifted_trails = {}
        for obj_id, trail in trails.items():
            shifted_trails[obj_id] = [(cx + ox, cy + oy) for (cx, cy) in trail]

        vis = draw_trajectories(vis, shifted_trails)

    return vis


# ──────────────── Stage 3: Cross-frame tracking glue ─────────────


def update_timeline(
    timeline: dict[int, dict], trails: dict[int, list[tuple[int, int]]], frame_idx: int
) -> None:
    """
    Record when each tracked object first / last appeared.

    ``timeline`` maps trail id -> {"first": int, "last": int}. For every trail
    alive in this frame: create the entry on first sight, otherwise advance
    ``last``. A trail that stops matching keeps its last-seen frame — that is
    its disappearance time.
    """
    # ── Your implementation here ──
    # Hint:
    #   for trail_id in trails:
    #       if trail_id not in timeline:
    #           timeline[trail_id] = {"first": frame_idx, "last": frame_idx}
    #       else:
    #           timeline[trail_id]["last"] = frame_idx
    for trail_id in trails:
        if trail_id not in timeline:
            timeline[trail_id] = {"first": frame_idx, "last": frame_idx}
        else:
            timeline[trail_id]["last"] = frame_idx


# ──────────────── Stage 4: Structured outputs ────────────────────


def pack_features(
    all_features: dict[str, list[list[list[float]]]], roi_names: list[str], n_frames: int
) -> dict[str, np.ndarray]:
    """
    Turn per-frame feature lists into fixed 2D/1D arrays for npz.

    ``all_features[name]`` is one list per frame; each frame is a list of
    feature vectors [cx, cy, area, b, g, r]. Returns a dict of arrays:
      - name:            (n_frames, max_objs, 6), NaN-padded
      - f"{name}_counts": (n_frames,) — real object count per frame
    """
    # ── Your implementation here ──
    # Hint:
    #   for each ROI: max_objs = max(len(v) for v in frame_lists)
    #   arr = np.full((n_frames, max_objs, 6), np.nan)
    #   for i, vecs in enumerate(frame_lists):
    #       arr[i, :len(vecs)] = vecs; counts[i] = len(vecs)
    for name in roi_names:
        max_objs = max(len(v) for v in all_features[name])
        arr = np.full((n_frames, max_objs, 6), np.nan)
        counts = np.zeros(n_frames, dtype=np.int32)
        for i, vecs in enumerate(all_features[name]):
            n = len(vecs)  # 先把长度取出来，算一次
            if n:  # 只有有物体时才填数据
                arr[i, :n] = vecs
            counts[i] = n  # 无论有没有，计数都要记（0 也算一种结果）
        all_features[name] = arr
        all_features[f"{name}_counts"] = counts

    return all_features


def write_summary(
    summary_path: Path,
    meta: dict,
    state: dict,
    frame_step: int,
) -> None:
    """
    Write a human-readable summary: total sampled frames, total objects,
    and a per-ROI timeline of object first/last seen frames.
    """
    # ── Your implementation here ──
    # Hint: build a list of lines, then write "\n".join(lines).
    #   - total objects = sum(len(st["timeline"]) for st in state.values())
    #   - per ROI, sort timeline items and format "trail X: frames A -> B"
    lines = []
    total_objects = sum(len(st["timeline"]) for st in state.values())
    lines.append(f"Total sampled frames: {meta['frame_count'] // frame_step}")
    lines.append(f"Total objects: {total_objects}")
    for name in state:
        lines.append(f"{name}: {len(state[name]['timeline'])} trails")
        for trail_id, timeline in sorted(state[name]["timeline"].items()):
            lines.append(f"trail {trail_id}: frames {timeline['first']} -> {timeline['last']}")
    with summary_path.open("w") as f:
        f.write("\n".join(lines))


# ──────────────────────────── Main ──────────────────────────────


def main() -> None:
    """
    Pipeline orchestration:
      1. Open video, read metadata, build ROI config
      2. For each sampled frame: per-ROI analyze + track + render, write video
      3. Emit output.json, features.npz, summary.txt
    """
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    cap = open_video(VIDEO_PATH)
    meta = get_video_metadata(VIDEO_PATH)
    h, w = meta["height"], meta["width"]
    roi_config = define_roi_config(w, h)
    roi_names = list(roi_config)

    # Per-ROI tracking state: trails, next id, appear/disappear timeline
    state: dict[str, dict] = {
        name: {"trails": {}, "next_id": 0, "timeline": {}} for name in roi_names
    }

    out_video = OUTPUT_DIR / "day_27_annotated.mp4"
    writer = create_writer(out_video, OUTPUT_FPS, even_size(w, h))

    frames_json: list[dict] = []
    all_features: dict[str, list] = {name: [] for name in roi_names}

    start = time.perf_counter()
    try:
        for frame_idx, frame in iter_every_n_frames(cap, FRAME_STEP):
            frame_rois: dict[str, list[dict]] = {}
            for name, (ox, oy, rw, rh) in roi_config.items():
                roi_frame = crop_roi(frame, (ox, oy, rw, rh))
                objects, features = analyze_roi(roi_frame)

                st = state[name]
                st["trails"], st["next_id"] = update_trajectories(
                    objects, st["trails"], st["next_id"], MAX_MATCH_DIST, TRAIL_LENGTH
                )
                update_timeline(st["timeline"], st["trails"], frame_idx)

                frame_rois[name] = objects
                all_features[name].append(features)

            rendered = render_frame(frame, roi_config, frame_rois, state, ROI_COLORS)
            writer.write(rendered)
            frames_json.append({"frame_idx": frame_idx, "rois": frame_rois})
    finally:
        cap.release()
        writer.release()
    elapsed = time.perf_counter() - start

    # ── Stage 4: write structured outputs ──
    output_path = OUTPUT_DIR / "output.json"
    with open(output_path, "w") as f:
        json.dump(
            {"source": VIDEO_PATH.name, "frame_step": FRAME_STEP, "frames": frames_json},
            f,
            indent=2,
        )

    arrays = pack_features(all_features, roi_names, len(frames_json))
    np.savez(OUTPUT_DIR / "features.npz", **arrays)

    summary_path = OUTPUT_DIR / "summary.txt"
    write_summary(summary_path, meta, state, FRAME_STEP)

    print(f"\nFrames sampled: {len(frames_json)} (step {FRAME_STEP})")
    print(f"Output:  {output_path}")
    print(f"Features:{OUTPUT_DIR / 'features.npz'}")
    print(f"Summary: {summary_path}")
    print(f"Video:   {out_video}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
