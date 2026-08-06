"""
Panel composition helpers extracted from day_25/26 panel builders.

Unifies day 25's three-panel hstack and day 26's 2x2 grid into one
``stack_panels`` grid: every panel is resized to a common display size,
grayscale inputs are promoted to BGR, and the grid's dimensions are kept
even so the result is safe for an mp4v (YUV 4:2:0) VideoWriter.
"""

import cv2
import numpy as np

from src.vision.video import even_size


def fit_to_height(img: np.ndarray, height: int) -> tuple[int, int]:
    """Return an even (width, height) preserving the image aspect ratio."""
    width = int(img.shape[1] * height / img.shape[0])
    return even_size(width, height)


def stack_panels(
    panels: list[np.ndarray],
    cols: int = 1,
    display_size: tuple[int, int] = (640, 360),
    labels: list[str] | None = None,
    label_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """
    Arrange ``panels`` into a grid with ``cols`` columns.

    Each panel is resized to ``display_size`` (rounded to even dims), then
    arranged left-to-right, top-to-bottom. Grayscale panels are converted to
    BGR. Missing cells are filled with black so the grid is always a full
    rectangle. If ``labels`` is given, each panel gets a text label at its
    top-left corner (same color for all panels).

    Returns the combined BGR grid.
    """
    if not panels:
        raise ValueError("at least one panel is required")
    w, h = even_size(*display_size)

    prepared: list[np.ndarray] = []
    for i, panel in enumerate(panels):
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(panel, (w, h))
        if labels is not None:
            cv2.putText(resized, labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)
        prepared.append(resized)

    rows = (len(prepared) + cols - 1) // cols
    while len(prepared) < rows * cols:
        prepared.append(np.zeros((h, w, 3), dtype=np.uint8))

    row_images = [np.hstack(prepared[r * cols : (r + 1) * cols]) for r in range(rows)]
    return np.vstack(row_images)
