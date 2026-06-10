"""
Day 01: Hello Pixels — extract frame from MP4 and explore image matrix
Date: 2026-06-10
Goal: Read a video, extract a single frame, inspect its matrix properties
Runtime: ~15 min
"""
import sys
from pathlib import Path

import cv2
import numpy as np

result_dir = Path(__file__).resolve().parent / "results"
video_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "test.mp4"
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    sys.exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {width}x{height} @ {fps}fps")

# Seek directly to frame 500 (0-indexed: 499)
cap.set(cv2.CAP_PROP_POS_FRAMES, 499)
ret, frame = cap.read()
cap.release()

if ret:
    result_dir.mkdir(exist_ok=True)
    frame_path = result_dir / "0500.jpg"
    cv2.imwrite(str(frame_path), frame)
    print(f"Frame 500 saved to {frame_path}")
else:
    print("Error: Could not read frame 500")


img = cv2.imread(str(frame_path))
print(f"shape:  {img.shape}")      # (高, 宽, 通道数)
print(f"dtype:  {img.dtype}")      # uint8
print(f"总像素数: {img.size}")      # 高×宽×3

print(f"文件大小: {frame_path.stat().st_size} bytes")

# 取中心像素
cy, cx = img.shape[0] // 2, img.shape[1] // 2
b, g, r = img[cy, cx]
print(f"中心像素 [{cy}, {cx}] 的 BGR: ({b}, {g}, {r})")

# 三通道各自的平均值
mean_b, mean_g, mean_r = np.mean(img, axis=(0, 1))
print(f"BGR 各通道均值: B={mean_b:.1f} G={mean_g:.1f} R={mean_r:.1f}")


# --- Exercise 2: find the brightest pixel ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
y, x = np.unravel_index(np.argmax(gray), gray.shape)
b, g, r = img[y, x]
print(f" brightest pixel @ [{y}, {x}]  gray={gray[y,x]}  BGR=({b}, {g}, {r})")

# --- top-5 brightest pixels ---
gray_flat = gray.ravel()
top5_idx = np.argpartition(gray_flat, -5)[-5:]              # partition: smallest 5 to the left
top5_idx = top5_idx[np.argsort(gray_flat[top5_idx])[::-1]]  # sort descending
ys, xs = np.unravel_index(top5_idx, gray.shape)
print(" top-5 brightest pixels:")
for i, (yi, xi) in enumerate(zip(ys, xs), 1):
    b, g, r = img[yi, xi]
    print(f"   {i}. [{yi:4d}, {xi:4d}]  gray={gray[yi,xi]:3d}  BGR=({b}, {g}, {r})")

# --- Exercise 3: watermark at bottom-right corner ---
text = "OpenCV Day 01"
(h, w) = img.shape[:2]
pos = (w - 400, h - 30)  # bottom-right, offset for text width
cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
cv2.imwrite(str(frame_path.with_stem("0500_watermarked")), img)
print(f" watermarked image saved")