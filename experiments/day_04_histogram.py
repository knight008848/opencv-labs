"""
Day 04: Histogram Analysis & Equalization
Date: 2026-06-15
Goal: Compute grayscale histogram, assess exposure quality, apply histogram
      equalization (grayscale + color via HSV), generate a 3-panel report figure.
Runtime: < 2 s
"""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
RESULT_DIR = Path(__file__).resolve().parent / "results"
INPUT_PATH = RESULT_DIR / "frame_1008.jpg"
OUTPUT_PATH = RESULT_DIR / "day04_histogram_report.jpg"


# ── TODO 1: assess_exposure() ───────────────────────────────────────────────
# 需求：根据灰度直方图判断曝光情况
# 输入：gray (2D uint8 ndarray)
# 输出：(label, mean_brightness, info_dict)
#
# 提示：
# - 用 np.mean() 和 np.percentile() 提取统计量
# - 不要简单以 127 为界——欠曝图的 mean 可能 50-60，过曝图可能 200+
# - 建议用百分位数判断：p75 < 85 → 欠曝（最亮的 25% 像素也暗）
#                        p25 > 170 → 过曝（最暗的 25% 像素也亮）
#                        p50 < 60 → 欠曝（中位数太暗）
#                        p50 > 195 → 过曝（中位数太亮）
# - info 字典建议包含: mean, p10, p25, p50, p75, p90, dark_frac, bright_frac
# - label 用英文 (避免 matplotlib 中文字体问题)


def assess_exposure(gray: np.ndarray) -> tuple[str, float, dict]:
    """Classify exposure based on histogram percentiles.

    Returns (label, mean_brightness, debug_info).
    """
    mean_val = float(np.mean(gray))
    p10, p25, p50, p75, p90 = np.percentile(gray, [10, 25, 50, 75, 90])

    dark_frac = (gray < 50).mean()    # 像素值<50 的占比
    bright_frac = (gray > 200).mean()  # 像素值>200 的占比

    # ── Classify exposure ────────────────────────────────────────────────
    # p75 < 85  → even the brightest quartile is dim  (broad underexposure)
    # p50 < 60  → median is dark despite p75 ≥ 85     (bimodal: dark mass +
    #              small bright patch that inflates p75)
    if p75 < 85 or p50 < 60:
        label = "underexposed"
    # p25 > 170 → even the darkest quartile is bright  (broad overexposure)
    # p50 > 195 → median is bright despite p25 ≤ 170   (bimodal: bright mass +
    #              small dark patch that deflates p25)
    elif p25 > 170 or p50 > 195:
        label = "overexposed"
    else:
        label = "normal"

    info = {
        "mean": float(mean_val),
        "p10": float(p10),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "dark_frac": float(dark_frac),
        "bright_frac": float(bright_frac),
    }

    return label, mean_val, info


# ── TODO 2: equalize_color() ────────────────────────────────────────────────
# 需求：对彩色图做直方图均衡化
# 输入：BGR 图像 (3D uint8 ndarray)
# 输出：均衡化后的 BGR 图像
#
# 提示：
# - cv2.equalizeHist() 只接受单通道灰度图
# - 彩色图必须在 HSV 空间只处理 V 通道，否则颜色失真
# - 步骤：BGR → HSV → 取 V 通道做 equalizeHist → 写回 HSV → 转 BGR


def equalize_color(img: np.ndarray) -> np.ndarray:
    """Equalize a color image safely — operate on V channel in HSV space."""

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



# ── TODO 3: build_histogram_figure() ────────────────────────────────────────
# 需求：生成一张 3 格报告图
# 布局：
#   ┌─────────────┬─────────────┐
#   │   Original   │  Equalized  │
#   │   (color)    │   (color)   │
#   ├─────────────┴─────────────┤
#   │   Histogram overlay       │
#   │   (gray, before vs after) │
#   └───────────────────────────┘
#
# 提示：
# - 用 fig.add_subplot(2, 2, 1) / (2, 2, 2) 做上方两格
# - 用 fig.add_subplot(2, 1, 2) 做下方跨列直方图
# - cv2.calcHist() 返回 (256, 1)，需要用 .ravel() 展平再给 matplotlib
# - 用 fill_between() 做半透明叠加直方图
# - 用 axvline() 标出均衡化前后的 mean 值
# - imshow 时记得 BGR → RGB 转换


def build_histogram_figure(
    img: np.ndarray,
    gray: np.ndarray,
    equ_gray: np.ndarray,
    equ_color: np.ndarray,
    exposure_label: str,
) -> plt.Figure:
    """Assemble the 3-panel report figure."""
    fig = plt.figure(figsize=(14, 10))

    # ── Top-left: original color ─────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original", fontsize=12)
    ax1.axis("off")

    # ── Top-right: equalized color ───────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(cv2.cvtColor(equ_color, cv2.COLOR_BGR2RGB))
    ax2.set_title("Equalized (HSV-V)", fontsize=12)
    ax2.axis("off")

    # ── Bottom: histogram overlay ────────────────────────────────────
    ax3 = fig.add_subplot(2, 1, 2)

    hist_orig = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist_equ = cv2.calcHist([equ_gray], [0], None, [256], [0, 256]).ravel()
    bins = np.arange(256)

    ax3.fill_between(bins, hist_orig, alpha=0.45, color="steelblue", label="Original")
    ax3.fill_between(bins, hist_equ, alpha=0.45, color="darkorange", label="Equalized")
    ax3.plot(bins, hist_orig, color="steelblue", linewidth=0.8)
    ax3.plot(bins, hist_equ, color="darkorange", linewidth=0.8)

    mean_orig = float(np.mean(gray))
    mean_equ = float(np.mean(equ_gray))
    ax3.axvline(mean_orig, color="steelblue", linestyle="--", linewidth=1.5,
                label=f"Mean orig = {mean_orig:.0f}")
    ax3.axvline(mean_equ, color="darkorange", linestyle="--", linewidth=1.5,
                label=f"Mean equ = {mean_equ:.0f}")

    ax3.set_xlim(0, 255)
    ax3.set_xlabel("Pixel intensity")
    ax3.set_ylabel("Pixel count")
    ax3.legend(fontsize=9, loc="upper right")
    ax3.set_title(f"Grayscale histogram — Exposure: {exposure_label}", fontsize=12)

    fig.tight_layout()
    return fig


# ── TODO 4: main() 接线 ─────────────────────────────────────────────────────
# Wire everything together: load -> equalize -> assess -> print -> save
#
# Steps:
# 1. imread INPUT_PATH -> img
# 2. cvtColor -> gray
# 3. equalizeHist(gray) -> equ_gray
# 4. equalize_color(img) -> equ_color
# 5. assess_exposure(gray) -> label, mean_val, info
# 6. Print terminal report (name, shape, label, mean, percentiles)
# 7. build_histogram_figure(img, gray, equ_gray, equ_color, label) -> fig
# 8. fig.savefig(OUTPUT_PATH, dpi=150)
#
# Checklist:
# - [x] Histogram drawn with matplotlib (not OpenCV bare-bones version)
# - [x] Exposure logic is more nuanced than a single 127 cutoff
# - [x] Report figure has appropriate resolution, no label clipping


def main():
    img = cv2.imread(str(INPUT_PATH))
    if img is None:
        print(f"Error: failed to read image at {INPUT_PATH}")
        sys.exit(1)
    print(f"shape: {img.shape}")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ_gray = cv2.equalizeHist(img_gray)
    equ_color = equalize_color(img)

    exposure_label, mean_val, info = assess_exposure(img_gray)

    # ── Terminal report ──────────────────────────────────────────────
    print("=" * 54)
    print(f"  Image : {INPUT_PATH.name}")
    print(f"  Shape : {img.shape[1]}×{img.shape[0]}  ({img.shape[2]} channels)")
    print(f"  Exposure : {exposure_label}  (mean = {mean_val:.1f})")
    print("-" * 54)
    print(f"  Percentiles  |  p10={info['p10']:5.0f}  p25={info['p25']:5.0f}  "
          f"p50={info['p50']:5.0f}  p75={info['p75']:5.0f}  p90={info['p90']:5.0f}")
    print(f"  Dark  fraction (<  50) : {info['dark_frac']:.3f}")
    print(f"  Bright fraction (> 200) : {info['bright_frac']:.3f}")
    print("=" * 54)

    fig = build_histogram_figure(
        img,
        img_gray,
        equ_gray,
        equ_color,
        exposure_label,
    )
    fig.savefig(OUTPUT_PATH, dpi=150)


if __name__ == "__main__":
    main()
