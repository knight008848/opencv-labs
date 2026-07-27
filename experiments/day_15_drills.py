"""
Day 15 / 2026-06-29 / Speed Drills — 3 quick exercises for Module 7
  1. Three thresholding methods on one image (side-lit)
  2. Erode → Dilate demo (noise removal + size recovery)
  3. Opening / Closing demo (gap repair + salt-noise removal)
Runtime: ~10 min total
"""

import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path("../data/processed/day_15")
OUT.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Drill 1: Three thresholding methods
# ──────────────────────────────────────────────
def drill_threshold_comparison():
    print("[Drill 1] Three thresholding methods on a side-lit image...")

    # Make a synthetic "document": white text on gray bg, side-lit
    bg = np.full((300, 600), 160, dtype=np.uint8)  # gray bg
    text = np.zeros((300, 600), dtype=np.uint8)
    cv2.putText(text, "HELLO OPENCV", (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)

    # side light: brightness gradient left→right
    gradient = np.linspace(0, 80, 600, dtype=np.uint8)  # 0→80
    light = bg.copy().astype(np.int16)
    light += gradient  # left dim, right bright
    light = np.clip(light, 0, 255).astype(np.uint8)
    # place text (use max to overlay white text)
    img = np.maximum(light, text).astype(np.uint8)

    # Three thresholding methods
    _, global_t = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # 2x2 comparison grid (300 rows img + 10 gap + 300 rows bottom = 610)
    panel = np.zeros((610, 1220), dtype=np.uint8)
    panel[:300, :600] = img
    panel[:300, 620:1220] = global_t
    panel[310:610, :600] = otsu
    panel[310:610, 620:1220] = adaptive

    # Annotate each quadrant with method name
    for x, y, label in [
        (10, 25, "Original"),
        (630, 25, "Global (127)"),
        (10, 335, "Otsu"),
        (630, 335, "Adaptive (Gaussian, 31)"),
    ]:
        cv2.putText(panel, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 1)

    path = str(OUT / "drill1_threshold.png")
    cv2.imwrite(path, panel)
    print(f"  Saved {path}")
    print("  White pixel ratios:")
    print(f"    global:   {cv2.countNonZero(global_t) / global_t.size:.1%}")
    print(f"    otsu:     {cv2.countNonZero(otsu) / otsu.size:.1%}")
    print(f"    adaptive: {cv2.countNonZero(adaptive) / adaptive.size:.1%}")
    print()


# ──────────────────────────────────────────────
# Drill 2: Erode → Dilate (opening principle)
# ──────────────────────────────────────────────
def drill_erode_dilate():
    print("[Drill 2] Erode then dilate to see opening principle...")

    # Binary image: 3 big white rectangles + 20 random noise dots
    img = np.zeros((400, 400), dtype=np.uint8)
    img[50:150, 50:150] = 255  # rect 1
    img[50:150, 250:350] = 255  # rect 2
    img[250:350, 100:300] = 255  # rect 3 (wide)
    # noise: random white dots
    np.random.seed(42)
    for _ in range(20):
        y, x = np.random.randint(0, 400, 2)
        cv2.circle(img, (x, y), 3, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    eroded = cv2.erode(img, kernel, iterations=2)
    opened = cv2.dilate(eroded, kernel, iterations=2)

    # Side-by-side: original | eroded | dilated-back
    h_panel = np.hstack([img, eroded, opened])
    # annotate widths with cv2.rectangle-ish comments
    path = str(OUT / "drill2_erode_dilate.png")
    cv2.imwrite(path, h_panel)
    print(f"  Saved {path}")
    print("  White pixel counts:")
    print(f"    original:  {cv2.countNonZero(img)}")
    print(f"    eroded:    {cv2.countNonZero(eroded)}  (noise gone, rects shrunk)")
    print(f"    dilate↑:   {cv2.countNonZero(opened)}  (rects recovered, noise stays gone)")
    print()


# ──────────────────────────────────────────────
# Drill 3: Opening & Closing on defects
# ──────────────────────────────────────────────
def drill_open_close():
    print("[Drill 3] Closing repairs gaps, Opening removes salt noise...")

    np.random.seed(99)

    img = np.zeros((300, 600), dtype=np.uint8)

    # ── Left half: a line with a 2-pixel break → Closing repairs it ──
    cv2.line(img, (50, 100), (250, 100), 255, 3)
    # Erase a 2-pixel gap in the middle
    img[98:103, 138:162] = 0
    cv2.putText(img, "GAP", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)

    # ── Right half: black bg + white salt noise → Opening removes noise ──
    # Draw a white reference rectangle (clean shape that should survive opening)
    img[50:250, 350:550] = 255
    # Sprinkle small white noise dots around the rect on the right half
    for _ in range(15):
        y, x = np.random.randint(0, 290, 2)
        img[y, x + 310] = 255
    cv2.putText(img, "NOISE", (360, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # 2x2 panel:  original-left  |  original-right
    #             closed-left    |  opened-right
    panel = np.zeros((600, 600), dtype=np.uint8)
    panel[:300, :300] = img[:, :300]  # top-left:  original left half
    panel[:300, 300:] = img[:, 300:]  # top-right: original right half
    panel[300:, :300] = closed[:, :300]  # bot-left:  closing repairs gap
    panel[300:, 300:] = opened[:, 300:]  # bot-right: opening removes noise

    # Annotate
    for x, y, label in [
        (10, 20, "Original"),
        (310, 20, "Original"),
        (10, 320, "Closed"),
        (310, 320, "Opened"),
    ]:
        cv2.putText(panel, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1)

    path = str(OUT / "drill3_open_close.png")
    cv2.imwrite(path, panel)
    print(f"  Saved {path}")

    # Verify: closing should ADD white pixels (filled gap)
    left_original = cv2.countNonZero(img[:, :300])
    left_closed = cv2.countNonZero(closed[:, :300])
    gap_fixed = left_closed > left_original

    # Verify: opening should REMOVE white pixels (noise dots gone)
    right_original = cv2.countNonZero(img[:, 300:])
    right_opened = cv2.countNonZero(opened[:, 300:])
    noise_removed = right_opened < right_original

    print("  Observations:")
    print(
        f"    Left  (Closing): gap filled?     {gap_fixed}  "
        f"(white px {left_original} -> {left_closed})"
    )
    print(
        f"    Right (Opening): noise removed?  {noise_removed}  "
        f"(white px {right_original} -> {right_opened})"
    )
    print()


if __name__ == "__main__":
    drills = [
        ("Threshold comparison", drill_threshold_comparison),
        ("Erode → Dilate", drill_erode_dilate),
        ("Opening & Closing", drill_open_close),
    ]
    failed = 0
    for name, fn in drills:
        try:
            fn()
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            failed += 1

    if failed:
        print(f"\n{failed}/{len(drills)} drills failed.", file=sys.stderr)
        sys.exit(1)
    else:
        print("All 3 drills done. View results in data/processed/day_15/")
