"""
Day 02: Color Spaces — BGR, RGB, HSV, Gray
Date: 2026-06-11
Goal: Understand color space conversion and saturation/color measurement
Runtime: ~5 min
"""
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np



def main():
    result_dir = Path(__file__).resolve().parent / "results"
    frame_path = result_dir / "frame_1008.jpg"

    if not frame_path.exists():
        print(f"Error: Could not find frame at {frame_path}")
        sys.exit(1)
    else:
        img = cv2.imread(str(frame_path))
        print(f"shape:  {img.shape}")
        print(f"dtype:  {img.dtype}")
        print(f"file size (original JPG): {frame_path.stat().st_size:,} bytes")

        # --- Conversions ---
        img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # =====================================================================
        # Task 1: 4-in-1 comparison — Original(BGR) | RGB | HSV | Gray
        # =====================================================================
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle("Color Space Converter — 4-in-1 Comparison", fontsize=14, y=0.98)

        # Panel 1: BGR (convert to RGB just for display so colors look right)
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("BGR (original)\nOpenCV default", fontsize=11)
        axes[0].axis("off")

        # Panel 2: RGB
        axes[1].imshow(img_rgb)
        axes[1].set_title("RGB\nRed-Green-Blue order", fontsize=11)
        axes[1].axis("off")

        # Panel 3: HSV (show H channel as hue wheel, S as saturation map)
        # Display the HSV image directly — colors look psychedelic because
        # matplotlib treats H/S/V values as R/G/B, but that's the point:
        # it visually proves HSV is a DIFFERENT representation
        axes[2].imshow(img_hsv)
        axes[2].set_title("HSV (raw)\nHue-Saturation-Value space", fontsize=11)
        axes[2].axis("off")

        # Panel 4: Gray
        axes[3].imshow(img_gray, cmap="gray")
        axes[3].set_title("Gray\nSingle channel, 0-255", fontsize=11)
        axes[3].axis("off")

        plt.tight_layout()
        cmp4_path = result_dir / "day02_4in1_comparison.jpg"
        fig.savefig(str(cmp4_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {cmp4_path.name}")

        # =====================================================================
        # Task 2: Pseudo-color — 3 colormaps on the grayscale image
        # =====================================================================
        colormaps = [
            (cv2.COLORMAP_JET,    "JET",    "blue→cyan→yellow→red"),
            (cv2.COLORMAP_HOT,    "HOT",    "black→red→yellow→white"),
            (cv2.COLORMAP_BONE,   "BONE",   "black→white (blue tint)"),
        ]

        colored = {}
        for cmap_id, name, desc in colormaps:
            colored[name] = cv2.applyColorMap(img_gray, cmap_id)

        fig2, axes2 = plt.subplots(1, 4, figsize=(18, 5))
        fig2.suptitle("Pseudo-Color Mapping — 3 Colormaps vs Original Gray", fontsize=14, y=0.98)

        # Panel 0: original gray (reference)
        axes2[0].imshow(img_gray, cmap="gray")
        axes2[0].set_title("Original Gray\n(reference)", fontsize=11)
        axes2[0].axis("off")

        for i, (cmap_id, name, desc) in enumerate(colormaps):
            # applyColorMap returns BGR, convert to RGB for display
            colored_rgb = cv2.cvtColor(colored[name], cv2.COLOR_BGR2RGB)
            axes2[i + 1].imshow(colored_rgb)
            axes2[i + 1].set_title(f"COLORMAP_{name}\n{desc}", fontsize=10)
            axes2[i + 1].axis("off")

        plt.tight_layout()
        cmp3_path = result_dir / "day02_3colormaps.jpg"
        fig2.savefig(str(cmp3_path), dpi=120, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {cmp3_path.name}")

        # =====================================================================
        # Task 3: File size reduction from BGR to Gray
        # =====================================================================
        bgr_path  = result_dir / "_tmp_bgr.jpg"
        gray_path = result_dir / "_tmp_gray.jpg"
        cv2.imwrite(str(bgr_path), img)
        cv2.imwrite(str(gray_path), img_gray)

        if not bgr_path.exists() or not gray_path.exists():
            print("Warning: failed to write temp files for size comparison")
        else:
            size_bgr  = bgr_path.stat().st_size
            size_gray = gray_path.stat().st_size
            reduction = (1 - size_gray / size_bgr) * 100

            print(f"\n--- File Size Comparison ---")
            print(f"  BGR  (3 channels): {size_bgr:>8,} bytes")
            print(f"  Gray (1 channel):  {size_gray:>8,} bytes")
            print(f"  Reduction:         {reduction:>7.1f}%")

            # Cleanup temp files
            bgr_path.unlink(missing_ok=True)
            gray_path.unlink(missing_ok=True)

        print("\nDone — Day 02 complete.")
        plt.close("all")  # safety net: close any leaked figures


if __name__ == "__main__":
    main()
