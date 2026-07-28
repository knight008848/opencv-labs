"""
Day 23 / 2026-07-28 / Module 10: Feature Matching (ORB + Ratio Test + RANSAC)
File: day_23_matcher.py
Goal: Object localizer — match synthetic template (reusing generate_texture_image)
      in a cluttered scene using ORB features, Ratio Test, and RANSAC homography.
Deliverable: comparison panel (4 stages) + green-box annotation + confidence report
Runtime: ~1 min

Headless note: All visualisation via matplotlib savefig.
See CLAUDE.md for headless policy.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path so src/ imports resolve
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils import generate_texture_image  # noqa: E402

# Resolve paths
PROJECT_DIR = _project_root
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── Config ────────────────────────────

ORB_NFEATURES = 500  # Number of features to detect
RATIO_THRESHOLD = 0.75  # Lowe's ratio test threshold
RANSAC_REPROJ_THRESH = 5.0  # RANSAC reprojection error (pixels)
RANSAC_CONFIDENCE = 0.99  # RANSAC confidence level

CANVAS_W, CANVAS_H = 600, 400
SCENE_W, SCENE_H = 900, 600


# ──────────────────── 1. Synthetic Image Generation ────────────
# Reuses generate_texture_image from src/utils to create both template
# and scene, keeping the script zero-external-dependency.


def make_template(w: int = 200, h: int = 200) -> np.ndarray:
    """
    Create a synthetic template image by reusing ``generate_texture_image``.

    Returns a grayscale image with checkerboard + geometric shapes (rectangle,
    circle, triangle) that provides rich corner/edge features for ORB detection.
    """
    return generate_texture_image(w, h)


def make_scene(template: np.ndarray) -> np.ndarray:
    """
    Create a cluttered scene containing the template at a random transform.

    Uses ``generate_texture_image()`` as the base background — its built-in
    shapes (rectangle, circle, triangle) act as natural distractors. The
    template is placed at a random offset with 0-45° rotation, making the
    matching problem non-trivial.

    Returns:
        grayscale scene image of shape (SCENE_H, SCENE_W)
    """
    rng = np.random.RandomState(42)
    scene = generate_texture_image(SCENE_W, SCENE_H)

    # ── Place template at random position with rotation ──
    th, tw = template.shape[:2]
    ox = rng.randint(300, SCENE_W - tw - 50)
    oy = rng.randint(150, SCENE_H - th - 30)
    angle = rng.uniform(0, 45)

    center = (tw / 2, th / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a, sin_a = abs(rot_mat[0, 0]), abs(rot_mat[0, 1])
    new_w = int(th * sin_a + tw * cos_a) + 4
    new_h = int(th * cos_a + tw * sin_a) + 4
    rot_mat[0, 2] += (new_w - tw) / 2
    rot_mat[1, 2] += (new_h - th) / 2
    template_rot = cv2.warpAffine(template, rot_mat, (new_w, new_h))

    # Clip to scene bounds
    rw, rh = min(new_w, SCENE_W - ox), min(new_h, SCENE_H - oy)
    template_rot = template_rot[:rh, :rw]

    # Blend template into scene using a mask to avoid dark halo
    roi = scene[oy : oy + rh, ox : ox + rw]
    mask = template_rot > 0
    roi[mask] = cv2.addWeighted(roi[mask], 0.0, template_rot[mask], 1.0, 0)
    scene[oy : oy + rh, ox : ox + rw] = roi

    return scene


# ──────────────────── 2. Feature Detection (复用 Day 22) ────────


def detect_keypoints(gray: np.ndarray, nfeatures: int = ORB_NFEATURES):
    """
    Run ORB detection on a grayscale image.

    Returns:
        (keypoints, descriptors)
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    return orb.detectAndCompute(gray, None)


# ──────────────────── 3. Feature Matching ──────────────────────
# TODO: Implement the three matching stages below.


def match_raw(desc1: np.ndarray, desc2: np.ndarray) -> list[cv2.DMatch]:
    """
    Brute-force matching between two descriptor sets.
    Returns ALL matches sorted by distance (best first).

    Steps:
      1. Create cv2.BFMatcher with normType=cv2.NORM_HAMMING
      2. Call .match(desc1, desc2) — returns list of DMatch objects
      3. Sort by distance (ascending)

    Returns:
        sorted list of DMatch objects
    """
    # ── Your implementation here ──
    pass  # TODO


def match_ratio_test(
    desc1: np.ndarray, desc2: np.ndarray, ratio: float = RATIO_THRESHOLD
) -> list[cv2.DMatch]:
    """
    KNN matching with Lowe's Ratio Test.

    Steps:
      1. BFMatcher(NORM_HAMMING) → .knnMatch(desc1, desc2, k=2)
      2. For each (m, n) pair: keep m if m.distance < ratio * n.distance
      3. Return the list of accepted matches

    Returns:
        list of DMatch objects that passed Ratio Test
    """
    # ── Your implementation here ──
    pass  # TODO


def match_ransac(
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
    reproj_thresh: float = RANSAC_REPROJ_THRESH,
    confidence: float = RANSAC_CONFIDENCE,
):
    """
    Compute homography from good matches using RANSAC.

    Steps:
      1. Extract matched point pairs:
         src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
         dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
      2. Call cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                 reproj_thresh, confidence)
      3. Compute:
         - inlier_mask: mask.ravel().astype(bool)  (True = inlier)
         - inlier_matches: [m for i, m in enumerate(good_matches) if inlier_mask[i]]
         - inlier_ratio: inlier_count / total_matches
      4. Return H, inlier_matches, inlier_ratio

    Returns:
        (H, inlier_matches, inlier_ratio)
        H = 3x3 homography matrix (or None if not found)
    """
    # ── Your implementation here ──
    # Edge case: if len(good_matches) < 4, RANSAC can't work → return (None, [], 0.0)
    pass  # TODO


# ──────────────────── 4. Draw Helpers ──────────────────────────


def draw_keypoints_side_by_side(
    img1: np.ndarray, kp1: list, img2: np.ndarray, kp2: list
) -> np.ndarray:
    """
    Create a side-by-side visualization of keypoints on both images.
    Use cv2.drawKeypoints with DRAW_RICH_KEYPOINTS flags.
    Stack horizontally with np.hstack.
    """
    # ── Your implementation here ──
    pass  # TODO


def draw_matches(
    img1: np.ndarray,
    kp1: list,
    img2: np.ndarray,
    kp2: list,
    matches: list[cv2.DMatch],
    max_matches: int = 50,
) -> np.ndarray:
    """
    Draw match lines between two images side by side.

    Use cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_matches], None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    Returns:
        image with match lines drawn
    """
    # ── Your implementation here ──
    pass  # TODO


def draw_object_box(scene: np.ndarray, template_shape: tuple, H: np.ndarray | None) -> np.ndarray:
    """
    Project template corners into scene using homography, draw green box.

    Steps:
      1. If H is None, draw "MATCH FAILED" text on scene and return.
      2. Define template corners:
         corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
      3. Project: cv2.perspectiveTransform(corners, H)
      4. Draw: cv2.polylines(scene_color, [projected_corners],
                             isClosed=True, color=(0,255,0), thickness=3)

    Args:
        scene: grayscale scene image
        template_shape: (w, h) of template
        H: 3x3 homography matrix (or None if matching failed)

    Returns:
        BGR image with green box / failure text
    """
    # ── Your implementation here ──
    # Note: Convert grayscale to BGR first for colored annotations.
    pass  # TODO


# ──────────────────── 5. Comparison Panel ──────────────────────
# TODO: Arrange all outputs into a clear 2×2 grid.


def build_comparison_panel(
    img1: np.ndarray,
    kp1: list,
    img2: np.ndarray,
    kp2: list,
    raw_matches: list,
    ratio_matches: list,
    inlier_matches: list,
    scene_annotated: np.ndarray,
    inlier_ratio: float,
    output_path: Path,
) -> None:
    """
    Create a 2×2 grid showing the four stages of matching:

    Top-left:     Side-by-side keypoints
    Top-right:    Raw matches (first 50)
    Bottom-left:  Ratio Test matches
    Bottom-right: RANSAC inliers + green box annotation

    Each subplot title includes match count. Bottom-right title
    also includes inlier_ratio.
    """
    # ── Your implementation here ──
    # Hint: plt.subplots(2, 2, figsize=(12, 10))
    #       For each axis: axis.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #                      axis.set_title(...)
    #                      axis.axis("off")
    #
    #       Save with: fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    #                  plt.close(fig)
    pass  # TODO


# ──────────────────── 6. Terminal Report ───────────────────────


def print_match_summary(n_raw: int, n_ratio: int, n_inlier: int, inlier_ratio: float) -> None:
    """
    Print a formatted table of matching results and confidence assessment.

    Confidence levels:
      - inlier_ratio > 0.40 → "High confidence"
      - inlier_ratio > 0.20 → "Medium confidence"
      - inlier_ratio > 0.10 → "Low confidence"
      - else                → "Match failed — object may not be in scene"

    Also check if n_ratio < 4 before RANSAC (not enough points to compute H).
    """
    # ── Your implementation here ──
    # Print like:
    #   Matching Results:
    #     Raw matches:        XXX
    #     After Ratio Test:   XXX  (XX.X% retained)
    #     After RANSAC:       XXX  (inlier ratio = XX.X%)
    #     Confidence:         Medium confidence
    pass  # TODO


# ──────────────────────────── Main ──────────────────────────────


def main() -> None:
    """
    Pipeline:
      1. Generate synthetic template + scene
      2. Detect ORB keypoints on both
      3. Match: raw → Ratio Test → RANSAC
      4. Draw green box (or failure text)
      5. Build 2×2 comparison panel → save PNG
      6. Print match summary to terminal
    """
    print("\n" + "=" * 55)
    print("  Day 23 — ORB Feature Matching & Object Localization")
    print("=" * 55)

    try:
        # ── Step 1: Generate synthetic images ──
        print("\n[1/6] Generating synthetic template and scene...")
        template = make_template()

        # TODO: verify template has sufficient features
        #   kp_t, _ = detect_keypoints(template)
        #   print(f"  Template keypoints: {len(kp_t)}")
        #   If len(kp_t) < 20, warn the user.

        scene = make_scene(template)
        print(f"  Template: {template.shape[1]}x{template.shape[0]}")
        print(f"  Scene:    {scene.shape[1]}x{scene.shape[0]}")

        # ── Step 2: Detect keypoints ──
        print("\n[2/6] Detecting ORB keypoints...")
        kp1, desc1 = detect_keypoints(template)
        kp2, desc2 = detect_keypoints(scene)
        print(f"  Template: {len(kp1)} keypoints, descriptors: {desc1.shape}")
        print(f"  Scene:    {len(kp2)} keypoints, descriptors: {desc2.shape}")

        # ── Step 3: Three-stage matching ──
        print("\n[3/6] Matching features...")

        raw_matches = match_raw(desc1, desc2)
        print(f"  Raw matches:       {len(raw_matches)}")

        ratio_matches = match_ratio_test(desc1, desc2)
        print(f"  After Ratio Test:  {len(ratio_matches)}")

        H, inlier_matches, inlier_ratio = match_ransac(kp1, kp2, ratio_matches)
        if H is not None and len(inlier_matches) > 0:
            print(
                f"  After RANSAC:       {len(inlier_matches)} (inlier ratio = {inlier_ratio:.1%})"
            )
        else:
            print("  RANSAC:            FAILED (not enough good matches)")

        # ── Step 4: Draw green box ──
        print("\n[4/6] Drawing object box annotation...")
        scene_annotated = draw_object_box(scene, template.shape, H)
        box_path = OUTPUT_DIR / "day_23_object_box.png"
        cv2.imwrite(str(box_path), scene_annotated)
        print(f"  Saved: {box_path}")

        # ── Step 5: Build comparison panel ──
        print("\n[5/6] Building 2×2 comparison panel...")
        panel_path = OUTPUT_DIR / "day_23_match_panel.png"
        build_comparison_panel(
            template,
            kp1,
            scene,
            kp2,
            raw_matches,
            ratio_matches,
            inlier_matches,
            scene_annotated,
            inlier_ratio,
            panel_path,
        )
        print(f"  Saved: {panel_path}")

        # ── Step 6: Print summary ──
        print("\n[6/6] Match summary:")
        print_match_summary(
            len(raw_matches),
            len(ratio_matches),
            len(inlier_matches) if inlier_matches else 0,
            inlier_ratio,
        )

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        return

    print("\n" + "=" * 55)
    print("  Done — check data/processed/day_23_*.png")
    print("=" * 55)


if __name__ == "__main__":
    main()
