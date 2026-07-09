"""
Day 16 (Review) / 2026-07-09
Goal: When to use erode/dilate/opening/closing — 4 real scenarios.
Runtime: ~3 min
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def make_scene() -> np.ndarray:
    """Create a binary image with 4 common problems."""
    img = np.zeros((300, 400), dtype=np.uint8)

    # 1. A big solid object (coin, workpiece)
    cv2.circle(img, (100, 120), 60, 255, -1)

    # 2. A thin line (scratch, wire, crack)
    cv2.line(img, (10, 230), (390, 230), 255, 2)
    cv2.line(img, (10, 240), (390, 240), 255, 2)

    # 3. Broken line (gap simulates a disconnected trace)
    cv2.line(img, (10, 270), (185, 270), 255, 2)
    cv2.line(img, (195, 270), (390, 270), 255, 2)

    # 4. Hole inside the circle
    cv2.circle(img, (100, 120), 20, 0, -1)

    # 5. Salt noise (white dots on black bg)
    salt_mask = np.random.rand(*img.shape) < 0.005
    bg_mask = img == 0
    img[salt_mask & bg_mask] = 255

    # 6. Pepper noise (black dots inside white)
    pepper_mask = np.random.rand(*img.shape) < 0.02
    fg_mask = img == 255
    img[pepper_mask & fg_mask] = 0

    return img


binary = make_scene()

# 3x3 rect kernel
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

eroded = cv2.erode(binary, k, iterations=1)
dilated = cv2.dilate(binary, k, iterations=1)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)    # erode -> dilate
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)   # dilate -> erode

# ------------------------------------------------
# Side-by-side comparison — 1 row, 5 columns
# ------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(15, 4.0))

titles = [
    "Original",
    "Erode",
    "Dilate",
    "Open",
    "Close",
]
subtitles = [
    "",
    "[OK] Noise gone\n[!!] Object shrinks",
    "[OK] Gaps filled\n[!!] Noise grows",
    "[OK] Noise gone\n[OK] Size preserved",
    "[OK] Holes filled\n[OK] Gaps closed",
]
images = [binary, eroded, dilated, opened, closed]

for idx, (img, title, sub) in enumerate(zip(images, titles, subtitles)):
    ax = axes[idx]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{title}\n{sub}", fontsize=7.5, linespacing=1.3)
    ax.axis("off")

plt.tight_layout(pad=0.5)
output_dir = Path("data/processed/day_16_review")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(str(output_dir / "morphology_applications.png"), dpi=150)
plt.close(fig)
print(f"Saved {output_dir / 'morphology_applications.png'}")
print()

# ------------------------------------------------
# Count white pixels for each operation
# ------------------------------------------------
print(f"{'Operation':<12} {'White px':>8} {'Δ vs original':>12}")
print("-" * 35)
orig_count = np.count_nonzero(binary)
for name, result in [
    ("original", binary), ("erode", eroded), ("dilate", dilated),
    ("open", opened), ("close", closed),
]:
    count = np.count_nonzero(result)
    delta = count - orig_count
    print(f"{name:<12} {count:>8} {delta:+>12d}")

print()
print("Key takeaway:")
print("  Open  (erode→dilate): removes SALT noise, preserves object size.")
print("  Close (dilate→erode): fills HOLES, connects BROKEN lines.")
print("  Erode alone:          removes noise BUT shrinks the object.")
print("  Dilate alone:         fills holes  BUT grows the object.")
