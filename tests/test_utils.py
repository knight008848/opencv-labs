from pathlib import Path

import cv2
import numpy as np

from src.utils import (
    any_image_reader,
    create_color_mask,
    create_comparison_grid,
    draw_labeled_bbox,
    ensure_dir,
    generate_texture_image,
    get_hsv_range,
    get_project_root,
    load_image,
    resize_keep_aspect,
    save_image,
    standardize_for_model,
    time_it,
)


def test_get_project_root():
    root = get_project_root()
    assert isinstance(root, Path)
    # The root should contain 'src' and 'tests'
    assert (root / "src").exists()
    assert (root / "tests").exists()


def test_ensure_dir(tmp_path):
    target = tmp_path / "new_folder" / "sub_folder"
    assert not target.exists()
    result = ensure_dir(target)
    assert target.exists()
    assert result == target


def test_load_image(tmp_path):
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))
    img = load_image(str(img_path))
    assert img is not None
    assert img.shape == (10, 10, 3)

    img_fail = load_image(str(tmp_path / "non_existent.png"))
    assert img_fail is None


def test_save_image(tmp_path):
    img_path = tmp_path / "nested" / "dir" / "out.png"
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    success = save_image(img, str(img_path))
    assert success
    assert img_path.exists()


def test_resize_keep_aspect_color():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)  # Blue image
    resized = resize_keep_aspect(img, target_size=100)
    assert resized.shape == (100, 100, 3)
    assert np.all(resized[0:25, :] == 0)


def test_resize_keep_aspect_grayscale():
    img = np.zeros((100, 200), dtype=np.uint8)
    img[:] = 255  # White image
    # Expected to fail due to 3-channel canvas in utils.py
    resized = resize_keep_aspect(img, target_size=100)
    assert resized.shape == (100, 100)


def test_standardize_for_model():
    img_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    img_bgr[:, :] = (0, 0, 255)  # BGR Red

    standardized = standardize_for_model(img_bgr)

    expected_red_val = (1.0 - 0.485) / 0.229
    expected_blue_val = (0.0 - 0.406) / 0.225

    # standardize_for_model converts the image to RGB format!
    # So Red is now at index 0, Blue is at index 2
    assert np.isclose(standardized[0, 0, 0], expected_red_val, atol=1e-2), "Red channel mismatch"
    assert np.isclose(standardized[0, 0, 2], expected_blue_val, atol=1e-2), "Blue channel mismatch"


def test_draw_labeled_bbox():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = (10, 10, 20, 20)
    res = draw_labeled_bbox(img, bbox, "test")
    assert res.shape == (100, 100, 3)


def test_create_comparison_grid():
    img1 = np.ones((50, 50, 3), dtype=np.uint8) * 100
    img2 = np.ones((50, 50, 3), dtype=np.uint8) * 200
    grid = create_comparison_grid([img1, img2], ["img1", "img2"], cols=2, cell_size=(100, 100))
    assert grid.shape == (100, 200, 3)


def test_create_comparison_grid_grayscale():
    img1 = np.ones((50, 50), dtype=np.uint8) * 100
    img2 = np.ones((50, 50), dtype=np.uint8) * 200
    # Expected to fail due to 3-channel grid initialization in utils.py
    grid = create_comparison_grid([img1, img2], ["img1", "img2"], cols=2, cell_size=(100, 100))
    assert grid is not None


def test_get_hsv_range():
    red_ranges = get_hsv_range("red")
    assert len(red_ranges) == 2
    unknown = get_hsv_range("unknown_color")
    assert unknown == []


def test_create_color_mask():
    hsv = np.zeros((10, 10, 3), dtype=np.uint8)
    hsv[0:5, :, :] = (5, 255, 255)

    mask = create_color_mask(hsv, "red")
    assert mask.shape == (10, 10)
    assert np.all(mask[0:5, :] == 255)
    assert np.all(mask[5:, :] == 0)


def test_time_it(capsys):
    @time_it
    def dummy_func():
        return 42

    res = dummy_func()
    assert res == 42
    captured = capsys.readouterr()
    assert "[dummy_func]" in captured.out


def test_any_image_reader(tmp_path):
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))
    res = any_image_reader(str(img_path))
    assert res is not None


# ──────────────────── generate_texture_image ─────────────────────


def test_generate_texture_image_defaults():
    """Default 600x400 → grayscale uint8 with checkerboard + shapes."""
    img = generate_texture_image()
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert img.shape == (400, 600)
    # Checkerboard guarantees both black and white
    assert img.min() == 0
    assert img.max() == 255


def test_generate_texture_image_custom_size():
    """Custom dimensions should be reflected in output shape."""
    img = generate_texture_image(w=200, h=100)
    assert img.shape == (100, 200)
    assert img.dtype == np.uint8
    assert img.ndim == 2


def test_generate_texture_image_shapes_present():
    """Verify the three overlayed shapes exist (rect / circle / triangle)."""
    img = generate_texture_image(w=600, h=400)
    # Gray rectangle at (50,50)-(150,150) → value 128
    assert img[75, 75] == 128, "Gray rectangle should be at (50,50)-(150,150)"
    # Dark circle at center (400,200) → value 64
    assert img[200, 400] == 64, "Dark circle should be centered at (400,200)"
    # Light triangle centroid area → value 192
    assert img[300, 500] == 192, "Light triangle should cover (500,300)"


def test_generate_texture_image_multi_value():
    """Checkerboard + shapes produce at least 4 distinct gray values."""
    img = generate_texture_image()
    unique = len(np.unique(img))
    assert unique >= 4, f"Expected ≥4 unique values, got {unique}"
