"""
ROI (region-of-interest) helpers extracted from day_26_multiroi.py.

ROIs are defined as fractions of the frame so the same config works at any
resolution: {name: (x_frac, y_frac, w_frac, h_frac)} -> concrete pixel rects.
"""

import cv2
import numpy as np

# Day 26 layout: whole frame | central 60% | top-left 25%.
DEFAULT_ROI_FRACTIONS: dict[str, tuple[float, float, float, float]] = {
    "panorama": (0.0, 0.0, 1.0, 1.0),
    "work_area": (0.2, 0.2, 0.6, 0.6),
    "inlet": (0.0, 0.0, 0.25, 0.25),
}

# Cyclic BGR palette used when no explicit color map is given.
DEFAULT_ROI_COLORS: list[tuple[int, int, int]] = [
    (0, 0, 255),  # red
    (0, 255, 0),  # green
    (255, 0, 0),  # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
]


def define_roi_config(
    width: int,
    height: int,
    fractions: dict[str, tuple[float, float, float, float]] | None = None,
) -> dict[str, tuple[int, int, int, int]]:
    """
    Convert fractional ROI specs into pixel rects (x, y, w, h).

    ``fractions`` maps roi name -> (x_frac, y_frac, w_frac, h_frac), all in
    [0, 1] relative to the frame. Defaults to the day 26 layout.
    """
    fractions = fractions or DEFAULT_ROI_FRACTIONS
    return {
        name: (
            round(x_frac * width),
            round(y_frac * height),
            round(w_frac * width),
            round(h_frac * height),
        )
        for name, (x_frac, y_frac, w_frac, h_frac) in fractions.items()
    }


def crop_roi(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Slice the frame to the (x, y, w, h) rect — a view, not a copy."""
    x, y, w, h = rect
    return frame[y : y + h, x : x + w]


def draw_roi_boxes(
    frame: np.ndarray,
    roi_config: dict[str, tuple[int, int, int, int]],
    colors: dict[str, tuple[int, int, int]] | list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """
    Draw one labeled rectangle per ROI.

    ``colors`` may be a {name: BGR} map or a list used cyclically; if None,
    a default palette is used. Mutates and returns the input frame.
    """
    palette = colors if isinstance(colors, list) else None
    for idx, (name, (x, y, w, h)) in enumerate(roi_config.items()):
        color = (
            colors[name]
            if isinstance(colors, dict)
            else (
                palette[idx % len(palette)]
                if palette
                else DEFAULT_ROI_COLORS[idx % len(DEFAULT_ROI_COLORS)]
            )
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame
