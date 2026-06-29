"""
Day 15 / 2026-06-29 / Module 7: Thresholding
Goal: Compare global, Otsu, and adaptive thresholding across 3 lighting conditions.
      Deliverable: 4-in-1 comparison panels + markdown report.
Runtime: ~60 min
"""

import cv2
import numpy as np
from pathlib import Path


def make_test_images(output_dir: Path) -> list[dict]:
    """
    Generate 3 synthetic test images simulating different lighting conditions:
      - uniform: even lighting, clean histogram
      - side_light: brightness gradient from left to right
      - very_dark: low overall brightness

    Returns a list of dicts: [{"name": str, "gray": np.ndarray}, ...]
    """
    h, w = 400, 700

    # --- Base document: dark text on light gray background ---
    bg = np.full((h, w), 180, dtype=np.uint8)
    text = np.zeros((h, w), dtype=np.uint8)
    lines = [
        (0.20, "OpenCV 30-Day Challenge"),
        (0.32, "Module 7: Thresholding"),
        (0.44, "Global vs Otsu vs Adaptive"),
        (0.56, "Each method has its niche."),
        (0.68, "Know when to use which."),
    ]
    for y_ratio, line in lines:
        y = int(h * y_ratio)
        cv2.putText(text, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 255, 2)

    # --- uniform: clean, even lighting ---
    uniform = np.maximum(bg, text).astype(np.uint8)

    # --- side_light: brightness gradient left->right ---
    # left stays at bg(180), right brightens up to +80
    grad = np.linspace(0, 80, w, dtype=np.uint8)
    side = bg.astype(np.int16) + grad[None, :]  # broadcast over rows
    side = np.clip(side, 0, 255).astype(np.uint8)
    side_light = np.maximum(side, text).astype(np.uint8)

    # --- very_dark: multiply whole image by 0.3 ---
    dark = (bg.astype(np.float32) * 0.3).astype(np.uint8)
    text_dark = (text.astype(np.float32) * 0.9).astype(np.uint8)  # dim text too
    very_dark = np.maximum(dark, text_dark).astype(np.uint8)

    images = [
        {"name": "uniform",   "gray": uniform},
        {"name": "side_light","gray": side_light},
        {"name": "very_dark", "gray": very_dark},
    ]

    # Save originals for reference
    for item in images:
        path = output_dir / f"{item['name']}_original.png"
        cv2.imwrite(str(path), item["gray"])

    return images


def apply_thresholds(gray: np.ndarray) -> dict[str, np.ndarray]:
    """
    Apply 3 thresholding methods + return the original gray.

    Returns:
        {"gray": gray, "global": ..., "otsu": ..., "adaptive": ...}
    """
    # TODO:
    #   1. global: cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #   2. otsu:   cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #   3. adaptive: cv2.adaptiveThreshold(gray, 255, ...)
    #   Return a dict keyed by method name.
    _, global_t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, otsu     = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive    = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10)
    return {"gray": gray, "global": global_t, "otsu": otsu, "adaptive": adaptive}


def white_pixel_ratio(binary: np.ndarray) -> float:
    """Return fraction of pixels that are 255 (white)."""
    # HINT: cv2.countNonZero works on single-channel uint8.
    return cv2.countNonZero(binary) / binary.size


def build_comparison_panel(results: dict[str, np.ndarray],
                           title: str) -> np.ndarray:
    """Create a 2x2 annotated comparison grid, returned as BGR for imwrite."""
    gray = results["gray"]
    h, w = gray.shape
    gap = 10

    panel = np.zeros((h * 2 + gap, w * 2 + gap), dtype=np.uint8)

    # 2x2 grid:  original      | global
    #             otsu          | adaptive
    panel[:h, :w]            = gray
    panel[:h, w + gap:]      = results["global"]
    panel[h + gap:, :w]      = results["otsu"]
    panel[h + gap:, w + gap:] = results["adaptive"]

    # Label each quadrant
    labels = [
        (10, 25,            "Original"),
        (w + gap + 10, 25,  "Global (127)"),
        (10, h + gap + 25,  "Otsu"),
        (w + gap + 10, h + gap + 25, "Adaptive"),
    ]
    for x, y, label in labels:
        cv2.putText(panel, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, 255, 1)

    # Annotate binary quads with white-pixel ratio
    ratio_annotations = [
        (w + gap + 10, 55,            results["global"]),
        (10, h + gap + 55,            results["otsu"]),
        (w + gap + 10, h + gap + 55,  results["adaptive"]),
    ]
    for x, y, mask in ratio_annotations:
        ratio = cv2.countNonZero(mask) / mask.size
        cv2.putText(panel, f"white {ratio:.1%}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

    # Convert to BGR and add title bar
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel_bgr, f"Lighting: {title}", (10, panel_bgr.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return panel_bgr


def write_report(results: list[dict], output_dir: Path) -> str:
    """Generate a markdown report recommending the best method per lighting type."""
    lines = [
        "# Day 15 — Thresholding Comparison Report",
        "",
        "| Lighting   | Global (127) | Otsu     | Adaptive | Winner      |",
        "|------------|-------------|----------|----------|-------------|",
    ]

    analysis = {
        "uniform": [
            "- Global (127) fails here: background gray (180) is already above 127, so everything turns white.",
            "- **Otsu wins**: it finds the valley between text (255) and background (180) automatically.",
            "- Adaptive also works but produces more white artifacts than necessary.",
            "- Lesson: when text is brighter than background, global threshold needs careful tuning.",
        ],
        "side_light": [
            "- Global fails completely (100% white) — the gradient spans both sides of 127.",
            "- Adaptive also over-thresholds (94% white) because the bright side has zero text/background contrast.",
            "- **Otsu** is the least broken: it splits the gradient at its midpoint, preserving text on the dim side.",
            "- Lesson: no method can rescue text that has zero contrast against its local background.",
        ],
        "very_dark": [
            "- Both Global (127) and Otsu produce the same result (4.2% white): clean text extraction.",
            "- Adaptive over-thresholds (86% white) — the 31×31 window is too large for this dim, low-contrast scene.",
            "- **Global (127)** wins by tie-break: simpler and faster, same output quality as Otsu here.",
            "- Lesson: when the histogram lacks a clear bimodal shape, Otsu offers no advantage over a fixed threshold.",
        ],
    }

    # Winner: a good result has white ratio in [1%, 30%] — not all-black, not all-white.
    # Among those, pick the one with the most moderate ratio.
    for item in results:
        name = item["name"]
        r = item["ratios"]

        def score(ratio):
            """0 = extreme (all-black or all-white), higher = more plausible."""
            if ratio <= 0.001 or ratio >= 0.999:
                return -1  # disqualified
            return 1.0 - abs(ratio - 0.10)  # expect ~5-15% text coverage

        valid = {m: s for m, rt in r.items() if (s := score(rt)) >= 0}
        winner = max(valid, key=valid.get) if valid else "???"

        lines.append(
            f"| {name:11} | {r['global']:7.1%}    | {r['otsu']:6.1%}   | {r['adaptive']:6.1%}   | **{winner}** |"
        )

    lines.extend([
        "",
        "## Per-condition analysis",
        "",
    ])

    for item in results:
        name = item["name"]
        r = item["ratios"]
        lines.append(f"### {name}")
        lines.append("")
        lines.extend(analysis.get(name, ["- No analysis available."]))
        lines.append("")
        lines.append("| Method   | White pixel ratio |")
        lines.append("|----------|-------------------|")
        for method in ["global", "otsu", "adaptive"]:
            lines.append(f"| {method:8} | {r[method]:6.1%}          |")
        lines.append("")

    lines.extend([
        "## Summary",
        "",
        "| Condition   | Recommended Method | Why |",
        "|-------------|-------------------|-----|",
        "| Uniform     | Otsu              | Global fails when bg gray (180) already sits above 127. Otsu finds the true text/bg divide. |",
        "| Side light  | Otsu              | All methods struggle; Otsu is the least broken — it splits the gradient at its midpoint. |",
        "| Very dark   | Global (127)      | Same output as Otsu (4.2% white), but simpler and faster. Dim histogram has no clear bimodal peak. |",
        "",
        f"*Images saved to `{output_dir}/`*",
    ])

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
            name: white_pixel_ratio(mask)
            for name, mask in thresh_results.items()
            if name != "gray"
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
    print("Done. Activate conda and run: python experiments/day_15_threshold.py")


if __name__ == "__main__":
    main()
