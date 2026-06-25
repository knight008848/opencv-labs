"""
Shared utilities for opencv-labs experiments.
"""

import time
from functools import wraps
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# --- Path utilities ---
def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- Image I/O ---
def load_image(path: str, color: bool = True) -> Optional[np.ndarray]:
    """Load an image with error handling."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not load image from {path}")
    return img


def save_image(img: np.ndarray, path: str) -> bool:
    """Save an image, creating parent directories as needed."""
    ensure_dir(Path(path).parent)
    return cv2.imwrite(str(path), img)


# --- Image preprocessing ---
def resize_keep_aspect(
    img: np.ndarray, target_size: int, pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """Resize image to target_size x target_size, preserving aspect ratio with padding."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


def standardize_for_model(
    img: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Standardize an RGB image for model input (ImageNet stats)."""
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    return img


# --- Visualization ---
def draw_labeled_bbox(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a bounding box with a text label."""
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def create_comparison_grid(
    images: List[np.ndarray],
    labels: List[str],
    cols: int = 3,
    cell_size: Tuple[int, int] = (300, 200),
) -> np.ndarray:
    """Create a grid of images with labels for comparison."""
    n = len(images)
    rows = (n + cols - 1) // cols
    grid_h = rows * cell_size[1]
    grid_w = cols * cell_size[0]
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

    for i, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        y, x = r * cell_size[1], c * cell_size[0]

        # Resize image to fit cell
        resized = cv2.resize(img, (cell_size[0] - 10, cell_size[1] - 30))
        grid[y + 5 : y + 5 + resized.shape[0], x + 5 : x + 5 + resized.shape[1]] = resized

        # Add label
        cv2.putText(
            grid, label, (x + 5, y + cell_size[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
        )

    return grid


# --- Performance ---
def time_it(func):
    """Decorator to measure function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"[{func.__name__}] {elapsed * 1000:.1f}ms")
        return result

    return wrapper


# --- Color utilities ---
def get_hsv_range(color_name: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Get HSV range(s) for common colors."""
    ranges = {
        "red": [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([179, 255, 255])),
        ],
        "green": [(np.array([40, 100, 50]), np.array([80, 255, 255]))],
        "blue": [(np.array([100, 100, 50]), np.array([130, 255, 255]))],
        "yellow": [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
        "orange": [(np.array([10, 100, 100]), np.array([20, 255, 255]))],
        "white": [(np.array([0, 0, 200]), np.array([179, 30, 255]))],
        "black": [(np.array([0, 0, 0]), np.array([179, 255, 50]))],
    }
    return ranges.get(color_name.lower(), [])


def create_color_mask(hsv_img: np.ndarray, color_name: str) -> np.ndarray:
    """Create a binary mask for a given color name in HSV space."""
    ranges = get_hsv_range(color_name)
    if not ranges:
        return np.zeros(hsv_img.shape[:2], dtype=np.uint8)

    mask = None
    for lower, upper in ranges:
        m = cv2.inRange(hsv_img, lower, upper)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


# --- HEIC / non-standard format I/O ---


def read_heic(path: str | Path) -> np.ndarray | None:
    """Read a HEIC image as an OpenCV BGR array.

    Requires ``pip install pillow-heif --break-system-packages`` in the
    pydata environment.  Without it the function prints instructions and
    returns None so your pipeline can fall back gracefully.

    Args:
        path:  Path to a .heic / .heif file.

    Returns:
        BGR uint8 array on success, or None if the dependency is missing
        or the file can't be read.
    """
    try:
        import pillow_heif
        from PIL import Image

        # pillow_heif must register its decoder with Pillow
        pillow_heif.register_heif_opener()
        pil_img = Image.open(str(path)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    except ImportError:
        print(
            "[read_heic] pillow-heif is not installed.\n"
            "  Run:  pip install pillow-heif --break-system-packages\n"
            "  Or convert manually: python tools/convert_heic.py"
        )
        return None
    except Exception as exc:
        print(f"[read_heic] Failed to read {path}: {exc}")
        return None


def any_image_reader(path: str | Path) -> np.ndarray | None:
    """Try to read an image regardless of format (HEIC, JPG, PNG, …).

    Tries HEIC first (if file extension matches), falls back to
    ``cv2.imread`` for standard formats.

    Args:
        path:  Path to an image file.

    Returns:
        BGR uint8 array, or None if every attempt failed.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".heic", ".heif", ".heics"}:
        result = read_heic(path)
        if result is not None:
            return result
        # fall through to cv2.imread (which will also fail, but at least
        # we tried with a useful error message)

    return cv2.imread(str(path))
