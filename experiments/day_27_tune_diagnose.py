"""
Day 27 / 2026-08-07 / Detection fragmentation diagnosis
File: day_27_tune_diagnose.py
Goal: explain why the day-27 pipeline yields ~108 trails in "panorama"
      for a single rolling ball, and quantify the three candidate causes:
        (1) HSV-mask noise  -> several blobs in one frame
        (2) morphology too weak -> ball split into fragments
        (3) track loss      -> ball moves > max_dist between sampled frames
Outputs:
  - stdout  : metrics for the current config + a small parameter sweep
  - PNG     : diagnose_grid.png — sample frames (BGR | mask | blobs)
Runtime: ~30 s
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path so src/ imports resolve
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.vision import (  # noqa: E402
    RED_RANGES,
    crop_roi,
    define_roi_config,
    find_objects,
    get_video_metadata,
    hsv_mask,
    iter_every_n_frames,
    open_video,
    update_trajectories,
)

# Paths / constants (mirror day_27_pipeline.py so the diagnosis matches it)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "day_27"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_PATH = PROJECT_DIR / "data" / "raw" / "滚动球.mp4"

ROI_NAME = "panorama"
FRAME_STEP = 5

# Baseline config currently used by the pipeline
CUR_CLOSE = 15
CUR_OPEN = 3
CUR_MIN_FRAC = 0.003
CUR_MAX_DIST = 250
CUR_MAX_LEN = 30


def load_sampled_roi(video_path: Path, step: int, roi_name: str) -> list[tuple[int, np.ndarray]]:
    """Return [(frame_idx, roi_frame_bgr)] sampled every ``step`` frames."""
    cap = open_video(video_path)
    meta = get_video_metadata(video_path)
    rect = define_roi_config(meta["width"], meta["height"])[roi_name]
    try:
        return [(idx, crop_roi(frame, rect)) for idx, frame in iter_every_n_frames(cap, step)]
    finally:
        cap.release()


def blob_stats(
    frames: list[tuple[int, np.ndarray]],
    close: int,
    open_k: int,
    min_frac: float,
) -> tuple[list[int], list[float]]:
    """
    Per-frame blob count and largest-blob area for the given mask config.

    Returns (counts, largest_areas); a frame with no blobs gets area 0.
    """
    counts: list[int] = []
    largest: list[float] = []
    for _idx, frame in frames:
        mask = hsv_mask(frame, RED_RANGES, close, open_k)
        min_area = min_frac * frame.shape[0] * frame.shape[1]
        objs = find_objects(mask, min_area)
        counts.append(len(objs))
        largest.append(max((o["area"] for o in objs), default=0.0))
    return counts, largest


def track_metrics(
    frames: list[tuple[int, np.ndarray]],
    close: int,
    open_k: int,
    min_frac: float,
    max_dist: float,
    max_len: int,
) -> dict:
    """
    Run the day-27 tracker and return aggregate trail metrics.

    ``next_id`` only ever grows, so it equals the total number of trails
    spawned during the run (dead trails are deleted but the counter stays).
    """
    trails: dict[int, list[tuple[int, int]]] = {}
    next_id = 0
    peak_alive = 0
    max_trail_len = 0
    for _idx, frame in frames:
        mask = hsv_mask(frame, RED_RANGES, close, open_k)
        min_area = min_frac * frame.shape[0] * frame.shape[1]
        objs = find_objects(mask, min_area)
        trails, next_id = update_trajectories(objs, trails, next_id, max_dist, max_len)
        peak_alive = max(peak_alive, len(trails))
        max_trail_len = max(max_trail_len, *(len(t) for t in trails.values()), 0)
    return {
        "total_trails": next_id,
        "peak_alive": peak_alive,
        "max_trail_len": max_trail_len,
    }


def largest_blob_displacement(
    frames: list[tuple[int, np.ndarray]], close: int, open_k: int, min_frac: float
) -> list[float]:
    """
    Centroid displacement of the largest blob between consecutive sampled
    frames where the largest blob exists in BOTH frames.

    If this p95 comfortably exceeds max_dist, trails break because the ball
    outruns the matching distance between sampled frames.
    """
    centroids: list[tuple[int, int]] = []
    for _idx, frame in frames:
        mask = hsv_mask(frame, RED_RANGES, close, open_k)
        min_area = min_frac * frame.shape[0] * frame.shape[1]
        objs = find_objects(mask, min_area)
        if not objs:
            centroids.append(None)  # type: ignore[arg-type]
            continue
        big = max(objs, key=lambda o: o["area"])
        centroids.append(big["centroid"])  # type: ignore[arg-type]
    dists = []
    prev = None
    for c in centroids:
        if c is not None and prev is not None:
            dists.append(float(np.linalg.norm(np.array(c) - np.array(prev))))
        prev = c
    return dists


def make_grid_image(
    frames: list[tuple[int, np.ndarray]], close: int, open_k: int, min_frac: float
) -> np.ndarray:
    """Side-by-side (BGR | mask | blobs) grid for a few sampled frames."""
    rows = []
    for idx, frame in frames:
        mask = hsv_mask(frame, RED_RANGES, close, open_k)
        min_area = min_frac * frame.shape[0] * frame.shape[1]
        objs = find_objects(mask, min_area)
        vis = frame.copy()
        for o in objs:
            x, y, w, h = o["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx, cy = o["centroid"]
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cell = np.hstack([vis, mask_bgr, vis])
        label = np.full((40, cell.shape[1], 3), 255, np.uint8)
        cv2.putText(
            label, f"sampled frame idx {idx}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2
        )
        rows.append(np.vstack([label, cell]))
    return np.vstack(rows)


def summarize(label: str, counts: list[int], largest: list[float], dists: list[float]) -> None:
    """Print one line of summary statistics for a config."""
    cnt = np.array(counts, dtype=int)
    arr = np.array(largest, dtype=float)
    dd = np.array(dists, dtype=float) if dists else np.array([0.0])
    pos = arr[arr > 0]
    if pos.size:
        area_line = (
            f"  largest-blob area   : min={pos.min():.0f} "
            f"p50={np.median(pos):.0f} max={arr.max():.0f}"
        )
    else:
        area_line = "  largest-blob area   : none"
    print(f"[{label}]")
    print(f"  frames with >1 blob : {int((cnt > 1).sum())}/{len(cnt)}")
    print(f"  frames with 0 blobs : {int((cnt == 0).sum())}/{len(cnt)}")
    print(area_line)
    print(
        f"  largest-blob step   : p50={np.median(dd):.1f} p95={np.percentile(dd, 95):.1f} "
        f"max={dd.max():.1f} px (max_dist={CUR_MAX_DIST})"
    )


def main() -> None:
    print(f"Loading sampled frames (step={FRAME_STEP}) from {VIDEO_PATH.name}...")
    frames = load_sampled_roi(VIDEO_PATH, FRAME_STEP, ROI_NAME)
    n = len(frames)
    print(f"  {n} sampled frames, ROI={ROI_NAME}")

    # --- Baseline config diagnostics ---
    counts, largest = blob_stats(frames, CUR_CLOSE, CUR_OPEN, CUR_MIN_FRAC)
    dists = largest_blob_displacement(frames, CUR_CLOSE, CUR_OPEN, CUR_MIN_FRAC)
    base = track_metrics(frames, CUR_CLOSE, CUR_OPEN, CUR_MIN_FRAC, CUR_MAX_DIST, CUR_MAX_LEN)
    print("\n=== Baseline config ===")
    print(
        f"  total_trails={base['total_trails']}  peak_alive={base['peak_alive']}  "
        f"max_trail_len={base['max_trail_len']}"
    )
    summarize("baseline", counts, largest, dists)

    # --- Grid image for visual inspection ---
    pick = frames[:: max(1, n // 6)][:6]
    grid = make_grid_image(pick, CUR_CLOSE, CUR_OPEN, CUR_MIN_FRAC)
    out = OUTPUT_DIR / "diagnose_grid.png"
    cv2.imwrite(str(out), grid)
    print(f"\nGrid image: {out}")

    # --- Parameter sweep: total trails vs config ---
    print("\n=== Sweep: total trails (closer to 1 = less fragmentation) ===")
    header = (
        f"  {'close':>5} {'min_frac':>8} {'max_dist':>8} | {'trails':>6} {'peak':>4} {'maxlen':>6}"
    )
    print(header)
    for close in (15, 21, 31):
        for mf in (0.0015, 0.004, 0.008):
            for md in (50, 120):
                r = track_metrics(frames, close, CUR_OPEN, mf, md, CUR_MAX_LEN)
                print(
                    f"  {close:>5} {mf:>8.4f} {md:>8} | {r['total_trails']:>6} "
                    f"{r['peak_alive']:>4} {r['max_trail_len']:>6}"
                )


if __name__ == "__main__":
    main()
