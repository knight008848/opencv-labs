import pytest
import cv2
import numpy as np
from pathlib import Path

from src.utils import (
    get_project_root,
    ensure_dir,
    resize_keep_aspect,
    standardize_for_model,
    create_comparison_grid,
    get_hsv_range,
    create_color_mask
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

def test_resize_keep_aspect_color():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:] = (255, 0, 0) # Blue image
    resized = resize_keep_aspect(img, target_size=100)
    assert resized.shape == (100, 100, 3)
    # Check that it's padded vertically
    # Original is 100x200 (aspect 1:2)
    # Resizing to 100 max dimension -> new size should be 50x100
    # Canvas is 100x100, so top 25 and bottom 25 should be padding (black by default)
    assert np.all(resized[0:25, :] == 0)

def test_resize_keep_aspect_grayscale():
    img = np.zeros((100, 200), dtype=np.uint8)
    img[:] = 255 # White image
    # This is expected to fail currently due to broadcasting error in utils.py
    resized = resize_keep_aspect(img, target_size=100)
    assert resized.shape == (100, 100) # Should ideally return a grayscale padded image

def test_standardize_for_model():
    # Model standardization uses ImageNet mean/std which are RGB
    # Our function takes BGR image. Let's test with a pure red image in BGR.
    img_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    img_bgr[:, :] = (0, 0, 255) # BGR Red

    # Mean = [0.485, 0.456, 0.406] (RGB)
    # If the function doesn't account for BGR, it will subtract the red mean (0.485) from the blue channel (index 0).
    standardized = standardize_for_model(img_bgr)
    
    # Let's see what we expect. If the function is CORRECT, it should handle BGR to RGB mapping internally, or subtract correct means.
    # Currently it just does (img - mean) / std directly, subtracting the 0.485 (Red mean) from the Blue channel (0).
    # We will test the desired behavior: standardization should treat the image correctly.
    # If we pass a pure red image (RGB=[255,0,0], BGR=[0,0,255]), the red channel should be normalized with 0.485 and 0.229.
    expected_red_val = (1.0 - 0.485) / 0.229 # roughly 2.248
    expected_blue_val = (0.0 - 0.406) / 0.225 # roughly -1.804
    
    # This will likely fail until we fix standardise_for_model
    assert np.isclose(standardized[0, 0, 2], expected_red_val, atol=1e-2), "Red channel mismatch"
    assert np.isclose(standardized[0, 0, 0], expected_blue_val, atol=1e-2), "Blue channel mismatch"

def test_create_comparison_grid():
    img1 = np.ones((50, 50, 3), dtype=np.uint8) * 100
    img2 = np.ones((50, 50, 3), dtype=np.uint8) * 200
    grid = create_comparison_grid([img1, img2], ["img1", "img2"], cols=2, cell_size=(100, 100))
    # Should create a 1x2 grid -> shape (100, 200, 3)
    assert grid.shape == (100, 200, 3)

def test_create_comparison_grid_grayscale():
    img1 = np.ones((50, 50), dtype=np.uint8) * 100
    img2 = np.ones((50, 50), dtype=np.uint8) * 200
    # Expected to fail due to 3-channel grid initialization in utils.py
    grid = create_comparison_grid([img1, img2], ["img1", "img2"], cols=2, cell_size=(100, 100))
    # For grayscale inputs, maybe the grid should be grayscale or the inputs converted to BGR
    # We'll assert it at least returns successfully without crashing.
    assert grid is not None

def test_get_hsv_range():
    red_ranges = get_hsv_range("red")
    assert len(red_ranges) == 2 # Red has two ranges due to hue wrap-around
    
    unknown = get_hsv_range("unknown_color")
    assert unknown == []

def test_create_color_mask():
    # Create an HSV image
    hsv = np.zeros((10, 10, 3), dtype=np.uint8)
    # Set top half to red hue (around 5), saturation 255, value 255
    hsv[0:5, :, :] = (5, 255, 255)
    
    mask = create_color_mask(hsv, "red")
    assert mask.shape == (10, 10)
    assert np.all(mask[0:5, :] == 255) # Top half should be selected
    assert np.all(mask[5:, :] == 0) # Bottom half should not be selected
