"""
Day 15 / 2026-06-29 / Module 7: Thresholding
Goal: Compare global, Otsu, and adaptive thresholding across 3 lighting conditions.
      Deliverable: 4-in-1 comparison panels + markdown report.
Runtime: ~60 min
"""

from pathlib import Path

import cv2
import numpy as np


def make_test_images(output_dir: Path) -> list[dict]:
    """
    Generate 3 synthetic test images simulating different lighting conditions.
    All use realistic white-paper + black-ink appearance.
    """
    np.random.seed(2026)
    h, w = 400, 700

    lines = [
        (0.20, "OpenCV 30-Day Challenge"),
        (0.32, "Module 7: Thresholding"),
        (0.44, "Global vs Otsu vs Adaptive"),
        (0.56, "Each method has its niche."),
        (0.68, "Know when to use which."),
    ]

    # --- uniform: white paper (255), black ink (0) ---
    paper = np.full((h, w), 255, dtype=np.uint8)
    for y_ratio, line in lines:
        y = int(h * y_ratio)
        cv2.putText(paper, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)  # color=0 (black)
    uniform = paper.copy()

    # --- side_light: right side shadowed, paper darkens ---
    # gradient: 255 (left) → 150 (right), text stays 0
    grad = np.linspace(255, 150, w, dtype=np.uint8)
    side = np.minimum(paper.copy(), grad[None, :])
    side_light = side.copy()

    # --- very_dark: overall dim, paper→gray, text→near-black ---
    # Multiply everything by 0.35. Paper 255→89, text 0→0.
    dark = (uniform.astype(np.float32) * 0.35).astype(np.uint8)
    very_dark = dark.copy()

    images = [
        {"name": "uniform", "gray": uniform},
        {"name": "side_light", "gray": side_light},
        {"name": "very_dark", "gray": very_dark},
    ]

    for item in images:
        path = output_dir / f"{item['name']}_original.png"
        cv2.imwrite(str(path), item["gray"])

    return images


def apply_thresholds(gray: np.ndarray) -> dict[str, np.ndarray]:
    """
    Apply 3 thresholding methods × 2 modes (BINARY + BINARY_INV).

    Returns:
        {"gray": gray,
         "global": ..., "otsu": ..., "adaptive": ...,
         "global_inv": ..., "otsu_inv": ..., "adaptive_inv": ...}
    """
    _, global_t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    _, global_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )

    return {
        "gray": gray,
        "global": global_t,
        "otsu": otsu,
        "adaptive": adaptive,
        "global_inv": global_inv,
        "otsu_inv": otsu_inv,
        "adaptive_inv": adaptive_inv,
    }


def white_pixel_ratio(binary: np.ndarray) -> float:
    """Return fraction of pixels that are 255 (white)."""
    # HINT: cv2.countNonZero works on single-channel uint8.
    return cv2.countNonZero(binary) / binary.size


def build_comparison_panel(results: dict[str, np.ndarray], title: str) -> np.ndarray:
    """
    Create a 4×2 grid comparing BINARY (left) vs BINARY_INV (right).

    Layout:
      Row 0: Original gray (spans both columns)
      Row 1: BINARY global   | BINARY_INV global
      Row 2: BINARY otsu     | BINARY_INV otsu
      Row 3: BINARY adaptive | BINARY_INV adaptive
    Each binary cell annotated with white-pixel ratio.
    """
    gray = results["gray"]
    h, w = gray.shape
    gap = 10
    rows, cols = 4, 2

    panel_h = h * rows + (rows - 1) * gap
    panel_w = w * cols + (cols - 1) * gap
    panel = np.zeros((panel_h, panel_w), dtype=np.uint8)

    # Row 0: original gray spanning both columns
    panel[:h, :w] = gray

    # Rows 1-3: BINARY (left) vs BINARY_INV (right)
    pairs = [
        (1, "global", "global_inv"),
        (2, "otsu", "otsu_inv"),
        (3, "adaptive", "adaptive_inv"),
    ]
    for row, key_bin, key_inv in pairs:
        y0 = row * (h + gap)
        panel[y0 : y0 + h, :w] = results[key_bin]
        panel[y0 : y0 + h, w + gap :] = results[key_inv]

    # Annotate row labels
    labels = [
        (5, 20, f"Original ({title})"),
        (5, h + gap + 20, "BINARY global"),
        (w + gap + 5, h + gap + 20, "BINARY_INV global"),
        (5, 2 * (h + gap) + 20, "BINARY otsu"),
        (w + gap + 5, 2 * (h + gap) + 20, "BINARY_INV otsu"),
        (5, 3 * (h + gap) + 20, "BINARY adaptive"),
        (w + gap + 5, 3 * (h + gap) + 20, "BINARY_INV adaptive"),
    ]
    for x, y, label in labels:
        cv2.putText(
            panel, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 160, 1
        )  # gray visible on both black & white

    # Ratio annotations for all 6 binary results
    ratio_positions = [
        (w + gap + 5, h + gap + 40, "global_inv"),
        (5, h + gap + 40, "global"),
        (w + gap + 5, 2 * (h + gap) + 40, "otsu_inv"),
        (5, 2 * (h + gap) + 40, "otsu"),
        (w + gap + 5, 3 * (h + gap) + 40, "adaptive_inv"),
        (5, 3 * (h + gap) + 40, "adaptive"),
    ]
    for x, y, key in ratio_positions:
        mask = results[key]
        ratio = cv2.countNonZero(mask) / mask.size
        cv2.putText(
            panel, f"white {ratio:.1%}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 160, 1
        )  # gray visible on both black & white

    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    return panel_bgr


def write_report(results: list[dict], output_dir: Path) -> str:
    """Generate a markdown report with BINARY vs BINARY_INV comparison."""
    lines = [
        "# Day 15 — Thresholding Comparison Report (White Paper + Black Ink)",
        "",
        "| Lighting   | BIN-global | BIN-otsu | BIN-adapt | INV-global | INV-otsu | INV-adapt | Winner  |",
        "|------------|-----------|----------|-----------|------------|----------|-----------|---------|",
    ]

    analysis = {
        "uniform": [
            "- White paper (255), black ink (0), even lighting.",
            "- BINARY with T=127 works: paper > 127 → white, text ≤ 127 → black.",
            "- BINARY_INV inverts the polarity: text becomes white, background becomes black.",
            "- Lesson: BINARY is natural for white-paper/black-ink; BINARY_INV is useful if you want white text for contour analysis.",
        ],
        "side_light": [
            "- White paper has a shadow gradient (255→150 right side), text stays black (0).",
            "- BINARY with T=127 still works because paper is always ≥150 > 127.",
            "- BINARY_INV inverts: text becomes white on a black background — cleaner for findContours.",
            "- Lesson: side-lighting is harmless here because paper is always brighter than ink.",
        ],
        "very_dark": [
            "- Overall dim: paper ≈ 89, text = 0. Paper falls below T=127.",
            "- BINARY global (127) fails: all pixels ≤ 127 → all black (no information).",
            "- **Otsu** rescues it: finds the valley between peaks at 0 and 89, correctly splits paper from ink.",
            "- BINARY_INV global (127) also fails: all pixels ≤ 127 → all white (no information).",
            "- Lesson: when lighting is low, fixed thresholds fail — Otsu or adaptive are necessary.",
        ],
    }

    for item in results:
        name = item["name"]
        r = item["ratios"]

        def score(ratio):
            """0 = extreme (all-black or all-white), higher = more plausible."""
            if ratio <= 0.001 or ratio >= 0.999:
                return -1
            return 1.0 - abs(ratio - 0.10)

        valid = {m: s for m, rt in r.items() if (s := score(rt)) >= 0}
        winner = max(valid, key=valid.get) if valid else "???"

        lines.append(
            f"| {name:11} | {r['global']:9.1%} | {r['otsu']:7.1%} | {r['adaptive']:8.1%} |"
            f" {r['global_inv']:8.1%} | {r['otsu_inv']:6.1%} | {r['adaptive_inv']:7.1%} |"
            f" **{winner}** |"
        )

    lines.extend(["", "## Per-condition analysis", ""])

    for item in results:
        name = item["name"]
        r = item["ratios"]
        lines.append(f"### {name}")
        lines.append("")
        lines.extend(analysis.get(name, ["- No analysis available."]))
        lines.append("")
        lines.append("| Method        | White pixel ratio |")
        lines.append("|---------------|-------------------|")
        for method in ["global", "otsu", "adaptive", "global_inv", "otsu_inv", "adaptive_inv"]:
            lines.append(f"| {method:13} | {r[method]:6.1%}          |")
        lines.append("")

    lines.extend(
        [
            "## Summary",
            "",
            "| Condition   | Recommended | Mode        | Why |",
            "|-------------|-------------|-------------|-----|",
            "| Uniform     | global/T=127 | BINARY      | White paper (255) > 127, black ink (0) ≤ 127. Clean split with zero computation. |",
            "| Side light  | global/T=127 | BINARY_INV  | Same split quality, but INV inverts to white-text-on-black — optimal for contour analysis. |",
            "| Very dark   | Otsu         | BINARY      | Only Otsu finds the valley in the compressed histogram. Fixed thresholds (127) produce all-black or all-white. |",
            "",
            f"*Images saved to `{output_dir}/`*",
        ]
    )

    report = "\n".join(lines)
    return report


def main():
    output_dir = Path("../data/processed/day_15")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: generate or load test images
    print("[1/3] Generating test images...")
    images = make_test_images(output_dir)

    # Step 2: threshold each image
    print("[2/3] Applying thresholding methods...")
    panels = []
    report_data = []
    for item in images:
        thresh_results = apply_thresholds(item["gray"])
        panel = build_comparison_panel(thresh_results, item["name"])
        panels.append(panel)

        ratios = {
            name: white_pixel_ratio(mask) for name, mask in thresh_results.items() if name != "gray"
        }
        report_data.append({"name": item["name"], "ratios": ratios})

    # Step 3: save panels and write report
    print("[3/3] Saving results and report...")
    for panel, item in zip(panels, images):
        path = output_dir / f"{item['name']}_comparison.png"
        cv2.imwrite(str(path), panel)
        print(f"  Saved {path}")

    report = write_report(report_data, output_dir)
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved {report_path}")
    print("Done. View results in data/processed/day_15/")


if __name__ == "__main__":
    main()
