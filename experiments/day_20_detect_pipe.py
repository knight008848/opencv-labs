"""
Day 20 / 2026-07-20 / Module 9 Concept C: Pipeline Design
File: day_20_detect_pipe.py  (object detection pipeline)
Goal: Build a complete object detection pipeline —
      load → gray → blur → Canny → morphology → contours → filter → analyze.
      Save intermediate results per step + final JSON output.
Deliverable: annotated images + JSON result files for 3+ test images.
Runtime: ~1.5 h

Headless note: All visualisation via matplotlib savefig.
See CLAUDE.md for headless policy.
"""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


# =========================  Configuration  ====================================

PIPELINE_CONFIG = {
    # Preprocessing
    "blur_ksize": (5, 5),  # GaussianBlur kernel size (odd, odd)
    "canny_t1": 50,  # Canny low threshold
    "canny_t2": 150,  # Canny high threshold
    # Morphology
    "morph_kernel_size": 3,  # kernel size for morphology operations
    "morph_iterations": 1,  # number of iterations
    "morph_operation": cv2.MORPH_CLOSE,  # close = dilate then erode (connects gaps)
    # Contour filtering
    "min_area": 50,  # discard contours smaller than this (px²)
    "max_area": 1e6,  # discard contours larger than this (px²)
    # ApproxPolyDP
    "approx_epsilon_factor": 0.02,  # fraction of arcLength for epsilon
    # Image loading
    "max_size": 1200,  # downscale longest side to this
    # Output
    "output_dir": PROJECT_DIR / "data" / "processed" / "day_20",
}


# ======================  Step 1 — Load Image  =================================


def load_image(path: str | Path, max_size: int | None = 1200) -> np.ndarray:
    """
    Load an image in BGR color.  Downscale if the longest side exceeds
    *max_size* (disabled when *max_size* is None).
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    if max_size is not None:
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    return img


# ===================  Step 2 — Convert to Grayscale  ==========================


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to single-channel grayscale.

    HINT: cv2.cvtColor with COLOR_BGR2GRAY.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


# ===================  Step 3 — Gaussian Blur  =================================


def apply_blur(gray: np.ndarray, ksize: tuple[int, int]) -> np.ndarray:
    """
    Reduce noise with Gaussian blur before edge detection.

    HINT: cv2.GaussianBlur. ksize must be odd (e.g. (5,5)).
    """
    return cv2.GaussianBlur(gray, ksize, 0)


# ===================  Step 4 — Canny Edge Detection  ==========================


def detect_edges(blurred: np.ndarray, low: int, high: int) -> np.ndarray:
    """
    Run Canny edge detection on the blurred grayscale image.

    HINT: cv2.Canny.
    low / high are the hysteresis thresholds (low / high).
    """
    return cv2.Canny(blurred, low, high)


# ===================  Step 5 — Morphology Close  ==============================


def morph_close(
    edges: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply morphological closing (dilate → erode) to connect broken edge segments.

    HINT:
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
      cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    Closing connects nearby white regions — useful for filling small gaps
    in detected edges so contours become complete.
    """
    if edges is None:
        return edges

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=iterations)


# ===================  Step 6 — Find Contours  =================================


def find_contours(
    binary: np.ndarray,
    min_area: float = 50.0,
    max_area: float = 1e6,
) -> list[np.ndarray]:
    """
    Extract external contours from a binary image, sorted by area descending.

    HINT:
      cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      Filter by min_area < cv2.contourArea(c) < max_area
      Sort descending by area.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered = [
        c for c in contours if cv2.contourArea(c) >= min_area and cv2.contourArea(c) < max_area
    ]
    return sorted(filtered, key=cv2.contourArea, reverse=True)


# =================  Step 7 — Geometric Analysis  ==============================


def compute_properties(cnt: np.ndarray) -> dict:
    """
    Compute a full set of geometric properties for a single contour.

    Use the following cv2 / numpy functions (each is one line):
        - cv2.contourArea(cnt)
        - cv2.arcLength(cnt, closed=True)
        - cv2.moments(cnt)   → cx, cy from m10/m00, m01/m00
        - cv2.boundingRect(cnt)  → (x, y, w, h)
        - cv2.minAreaRect(cnt)   → ((cx, cy), (w, h), angle) — rotated

    HINT: circularity = 4 * pi * area / (perimeter ** 2)
          aspect_ratio = w / h  (use boundingRect's w, h)

    Returns dict with keys:
        area, perimeter, cx, cy, bbox_x, bbox_y, bbox_w, bbox_h,
        rotated_cx, rotated_cy, rotated_w, rotated_h, rotated_angle,
        circularity, aspect_ratio
    """

    area = cv2.contourArea(cnt)

    perimeter = cv2.arcLength(cnt, True)

    # - If m00 == 0, set cx/cy = 0 (prevent ZeroDivisionError)
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        cx, cy = 0, 0
    else:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

    bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)
    ((rotated_cx, rotated_cy), (rotated_w, rotated_h), rotated_angle) = cv2.minAreaRect(cnt)

    circularity = 4 * np.pi * area / (perimeter**2)
    aspect_ratio = rotated_w / rotated_h

    return {
        "area": area,
        "perimeter": perimeter,
        "cx": cx,
        "cy": cy,
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "rotated_cx": rotated_cx,
        "rotated_cy": rotated_cy,
        "rotated_w": rotated_w,
        "rotated_h": rotated_h,
        "rotated_angle": rotated_angle,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
    }


def classify_shape(cnt: np.ndarray, epsilon_factor: float = 0.02) -> str:
    """
    Classify a contour as 'Triangle', 'Rectangle', 'Circle', or 'Irregular'.

    Pipeline:
        1. perimeter = cv2.arcLength(cnt, True)
        2. epsilon = 0.02 * perimeter  (starting value — tune if misclassifies)
        3. approx = cv2.approxPolyDP(cnt, epsilon, True)
        4. vertices = len(approx)
        5. Classify:
             vertices == 3  → "Triangle"
             vertices == 4  → "Rectangle"
             vertices >= 8  → "Circle"
             else           → "Irregular"

    HINT: 0.02 * perimeter is a heuristic. If your objects are very small or
    very noisy, try epsilon = 0.01 * perimeter (stricter = more vertices).
    If a known-rectangle classifies as "Irregular", epsilon is too small.

    HINT: A true rectangle has 4 vertices, but noise may give 5-7.
    Consider treating 4 <= vertices <= 6 as "Rectangle" for real-world images.

    Returns:
        One of: "Triangle", "Rectangle", "Circle", "Irregular"
    """

    perimeter = cv2.arcLength(cnt, True)
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        return "Rectangle"
    elif vertices >= 8:
        return "Circle"
    else:
        return "Irregular"


# ===================  Step 8 — Annotate Image  ================================


def get_color_palette(n: int) -> list[tuple[int, int, int]]:
    """
    Return n distinct BGR colors for drawing contours.

    Generates evenly-spaced hues in HSV, converts to BGR.
    Returns empty list when n == 0.
    """
    if n == 0:
        return []
    hsv = np.zeros((1, n, 3), dtype=np.uint8)
    hsv[0, :, 0] = np.linspace(0, 179, n, dtype=np.uint8)
    hsv[0, :, 1] = 255
    hsv[0, :, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return [tuple(int(v) for v in c) for c in bgr[0]]


def draw_annotations(
    image_bgr: np.ndarray,
    contours: list[np.ndarray],
    properties: list[dict],
    shapes: list[str],
) -> np.ndarray:
    """
    Draw contour outlines + bounding boxes + labels on a copy of the image.

    HINT:
      Use cv2.drawContours for outlines.
      Use cv2.rectangle for bounding boxes (blue, thickness=2).
      Use cv2.putText for "ID:area,shape" labels near the centroid.
      Generate colours from a smooth HSV palette (see Day 17).

    Returns BGR image ready for imwrite.
    """
    image = image_bgr.copy()
    colors = get_color_palette(len(contours))

    for i, c in enumerate(contours):
        color = colors[i]

        # Draw contour outline
        cv2.drawContours(image, contours, i, color, thickness=2)

        # Draw axis-aligned bounding box (blue)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Label near centroid (already computed in properties[i])
        cx, cy = properties[i]["cx"], properties[i]["cy"]
        cv2.putText(
            image,
            f"ID:{i} Area:{int(cv2.contourArea(c))} Shape:{shapes[i]}",
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    return image


# ===================  Pipeline Orchestrator  ==================================


def run_pipeline(
    image_path: Path,
    config: dict,
) -> dict:
    """
    Run the full 7-step detection pipeline on a single image.

    Each step saves its intermediate result to a steps/ subdirectory.
    Returns a result dict containing:
      - "image_name": str
      - "output_paths": dict of step_name → file path
      - "objects": list of dicts with properties + shape
      - "total_objects": int
    """
    output_dir = Path(config["output_dir"])
    steps_dir = output_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    image_name = image_path.stem
    result = {
        "image_name": image_name,
        "output_paths": {},
        "objects": [],
        "total_objects": 0,
    }

    try:
        # ---- Step 1: Load ----
        print(f"  [1/7] Loading {image_path.name}...")
        img = load_image(image_path, config["max_size"])
        cv2.imwrite(str(steps_dir / f"{image_name}_01_load.jpg"), img)
        result["output_paths"]["step_01_load"] = str(steps_dir / f"{image_name}_01_load.jpg")

        # ---- Step 2: Grayscale ----
        print("  [2/7] Converting to grayscale...")
        gray = to_grayscale(img)
        cv2.imwrite(str(steps_dir / f"{image_name}_02_gray.jpg"), gray)
        result["output_paths"]["step_02_gray"] = str(steps_dir / f"{image_name}_02_gray.jpg")

        # ---- Step 3: Blur ----
        print("  [3/7] Applying Gaussian blur...")
        blurred = apply_blur(gray, config["blur_ksize"])
        cv2.imwrite(str(steps_dir / f"{image_name}_03_blur.jpg"), blurred)
        result["output_paths"]["step_03_blur"] = str(steps_dir / f"{image_name}_03_blur.jpg")

        # ---- Step 4: Canny ----
        print("  [4/7] Detecting edges...")
        edges = detect_edges(blurred, config["canny_t1"], config["canny_t2"])
        cv2.imwrite(str(steps_dir / f"{image_name}_04_canny.jpg"), edges)
        result["output_paths"]["step_04_canny"] = str(steps_dir / f"{image_name}_04_canny.jpg")

        # ---- Step 5: Morphology ----
        print("  [5/7] Applying morphology...")
        closed = morph_close(edges, config["morph_kernel_size"], config["morph_iterations"])
        cv2.imwrite(str(steps_dir / f"{image_name}_05_morph.jpg"), closed)
        result["output_paths"]["step_05_morph"] = str(steps_dir / f"{image_name}_05_morph.jpg")

        # ---- Step 6: Contours ----
        print("  [6/7] Finding contours...")
        contours = find_contours(closed, config["min_area"], config["max_area"])
        result["total_objects"] = len(contours)

        # ---- Step 7: Analyze ----
        print("  [7/7] Analyzing geometry...")
        properties = []
        shapes = []
        for cnt in contours:
            props = compute_properties(cnt)
            shape = classify_shape(cnt, config["approx_epsilon_factor"])
            props["shape"] = shape
            properties.append(props)
            shapes.append(shape)
            result["objects"].append(props)

        # ---- Draw annotations ----
        annotated = draw_annotations(img, contours, properties, shapes)
        ann_path = str(output_dir / f"{image_name}_annotated.jpg")
        cv2.imwrite(ann_path, annotated)
        result["output_paths"]["annotated"] = ann_path

        print(f"      → {len(contours)} objects detected, annotated saved.")

    except Exception as e:
        print(f"  [ERROR] Pipeline failed for {image_name}: {e}")
        result["error"] = str(e)

    return result


# =====================  Build Debug Grid  =====================================


def build_debug_grid(
    step_paths: dict[str, str],
    annotated_path: str,
    objects: list[dict],
    output_path: Path,
) -> None:
    """
    Create a 2×3 or 3×2 grid showing intermediate pipeline results.

    Panels (in order they appear in the grid):
      Load | Gray  | Blur
      Canny | Morph | Annotated

    HINT: plt.subplots(2, 3). Use cv2.imread to load each step image.
    Convert BGR → RGB before displaying with imshow.
    """
    step_keys = [
        "step_01_load",
        "step_02_gray",
        "step_03_blur",
        "step_04_canny",
        "step_05_morph",
        "annotated",
    ]
    display_names = {
        "step_01_load": "1. Load",
        "step_02_gray": "2. Grayscale",
        "step_03_blur": "3. Gaussian Blur",
        "step_04_canny": "4. Canny Edges",
        "step_05_morph": "5. Morph Close",
        "annotated": "6. Annotated",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Detection Pipeline — Intermediate Results", fontsize=14)

    for ax, key in zip(axes.flat, step_keys):
        path = step_paths.get(key, "")
        if path and Path(path).exists():
            img_bgr = cv2.imread(str(path))
            if len(img_bgr.shape) == 2:
                ax.imshow(img_bgr, cmap="gray")
            else:
                ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(display_names.get(key, key))
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved debug grid: {output_path}")


# =====================  Save JSON Results  ====================================


def save_json_report(results: list[dict], output_path: Path) -> None:
    """
    Write a JSON report containing results for all processed images.

    HINT: json.dump(results, f, indent=2, ensure_ascii=False)
    Each result dict has keys: image_name, total_objects, objects[...].
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ============================  Main  ==========================================


def main():
    """
    Pipeline orchestration:
      1. Prepare config and output directories.
      2. Collect test images (type_test.png + 2 real photos).
      3. Run pipeline on each image with try/except.
      4. Build debug grid for each result.
      5. Save combined JSON report.
      6. Print terminal summary.
    """
    output_dir = PIPELINE_CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Collect test images ---
    raw_dir = PROJECT_DIR / "data" / "raw"
    test_images = [
        raw_dir / "type_test.png",  # Synthetic: geometric shapes
        raw_dir / "IMG_0701.png",  # Real: desktop scene 1
        raw_dir / "IMG_0705.png",  # Real: desktop scene 2
    ]

    # Verify all files exist
    valid_images = [p for p in test_images if p.exists()]
    missing = [p.name for p in test_images if not p.exists()]
    if missing:
        print(f"[WARN] Missing images (skipping): {missing}")

    if not valid_images:
        print("[ERROR] No valid test images found. Exiting.")
        return

    print(f"\n{'=' * 55}")
    print("  Day 20 — Object Detection Pipeline")
    print(f"  Images to process: {[p.name for p in valid_images]}")
    print(f"{'=' * 55}\n")

    # --- Run pipeline on each image ---
    all_results = []
    for img_path in valid_images:
        print(f"--- Processing: {img_path.name} ---")
        result = run_pipeline(img_path, PIPELINE_CONFIG)

        # Build debug grid
        debug_path = output_dir / f"{img_path.stem}_debug.png"
        build_debug_grid(
            result["output_paths"],
            result["output_paths"].get("annotated", ""),
            result["objects"],
            debug_path,
        )

        all_results.append(result)
        print(f"    Debug grid: {debug_path}\n")

    # --- Save combined JSON ---
    json_path = output_dir / "day_20_report.json"
    save_json_report(all_results, json_path)

    # --- Terminal summary ---
    print(f"{'=' * 55}")
    print("  Day 20 Pipeline — Summary")
    print(f"{'=' * 55}")
    for res in all_results:
        err = res.get("error")
        if err:
            print(f"  ❌ {res['image_name']}: FAILED — {err}")
        else:
            print(f"  ✅ {res['image_name']}: {res['total_objects']} objects detected")
    print(f"\n  JSON report : {json_path}")
    print(f"  Output dir  : {output_dir}/")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
