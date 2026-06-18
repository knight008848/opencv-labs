"""
Day 08: Batch Image Resize — letterbox all images to 224×224
Date: 2026-06-18
Goal: Read all jpg/png from data/raw/, resize to 224×224 keeping aspect ratio,
      pad with black borders (letterbox), save to data/processed/day_08_batch/
Runtime: < 5 s for ~10 images
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "day_08_batch"
TARGET_SIZE = (224, 224)  # (width, height) — matches resize() arg order

# ═══════════════════════════════════════════════════════════════════════════════
# 1. letterbox_resize() — resize with aspect-ratio preservation
# ═══════════════════════════════════════════════════════════════════════════════
#
# 思路：
#   - 计算缩放比例 scale = min(target_w / w, target_h / h)
#   - 用 cv2.resize 缩放到 (w*scale, h*scale)，插值按方向选 AREA 或 CUBIC
#   - 创建黑色 target 画布，居中贴入缩放后的图
#   - 返回 (结果图, scale, padding_pixels)


def letterbox_resize(img: np.ndarray, target_w: int, target_h: int) -> tuple[np.ndarray, float, int]:
    """Resize img to fit within target_w×target_h, pad with black to center.

    Args:
        img:      input image (BGR or grayscale)
        target_w: desired output width  (224)
        target_h: desired output height (224)

    Returns:
        (output_image, scale_factor, padding_pixels)
          padding_pixels = target_w * target_h - new_w * new_h
    """
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)

    # 缩放 — 按方向选择插值，clamp 防止 round 到 0
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # 建黑色画布，兼容灰度/彩色
    canvas_shape = (target_h, target_w, img.shape[2]) if img.ndim == 3 else (target_h, target_w)
    canvas = np.zeros(canvas_shape, dtype=img.dtype)

    # 居中贴入
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y+new_h, x:x+new_w] = resized

    padding = target_w * target_h - new_w * new_h
    return canvas, scale, padding



# ═══════════════════════════════════════════════════════════════════════════════
# 2. collect_images() — discover all jpg/png files under a directory
# ═══════════════════════════════════════════════════════════════════════════════


def collect_images(directory: Path) -> list[Path]:
    """Return a sorted list of .jpg/.png file paths (case-insensitive)."""
    return sorted([p for p in directory.glob("*")
                   if p.suffix.lower() in (".jpg", ".png")])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. process_batch() — iterate all images, letterbox, save, collect stats
# ═══════════════════════════════════════════════════════════════════════════════


def process_batch(image_paths: list[Path], output_dir: Path, target_w: int, target_h: int) -> dict[str, tuple[int, int, float, int]]:
    """Process every image: letterbox → save → return stats per file.

    Returns:
        Map of filename → (orig_w, orig_h, scale, padding_pixels)
    """

    stats = {}

    for image in image_paths:
        img = cv2.imread(str(image))
        if img is None:
            print(f"  [SKIP] Cannot read {image.name}")
            continue
        print(f"  {image.name}")
        h, w = img.shape[:2]
        resized, scale, padding = letterbox_resize(img, target_w, target_h)
        ok = cv2.imwrite(str(output_dir / image.name), resized)
        if not ok:
            print(f"  [WARN] Failed to write {image.name}")
        stats[image.name] = (w, h, scale, padding)
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 4. print_results() — terminal summary table
# ═══════════════════════════════════════════════════════════════════════════════


def print_results(stats: dict[str, tuple[int, int, float, int]]) -> None:
    """Print per-image summary: filename, original size, scale, padding."""
    print()
    for fname, (orig_w, orig_h, scale, padding) in stats.items():
        print(f"{fname}: {orig_w}×{orig_h} → scale {scale:.2f}, padding {padding} pixels")
    print(f"\n{len(stats)} image(s) processed → {OUTPUT_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. main() — pipeline orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Discover → process → report pipeline."""
    # 1. Discover images
    if not RAW_DIR.exists():
        print(f"Input directory not found: {RAW_DIR}")
        sys.exit(1)
    images = collect_images(RAW_DIR)
    if not images:
        print(f"No jpg/png files found in {RAW_DIR}")
        sys.exit(1)
    print(f"Found {len(images)} image(s) in {RAW_DIR}\n")

    # 2. Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Process batch

    stats = process_batch(images, OUTPUT_DIR, *TARGET_SIZE)

    # 4. Terminal report
    print_results(stats)


if __name__ == "__main__":
    main()
