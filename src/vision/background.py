"""
Background-subtraction pipeline extracted from day_25_motion.py.

MOG2 produces a per-frame raw mask with three values:
0 = background, 127 = shadow, 255 = foreground. Shadows are NOT foreground,
so they must be dropped before morphology cleans the mask.
"""

import cv2
import numpy as np


def init_background_subtractor(
    history: int = 500,
    var_threshold: float = 16.0,
    detect_shadows: bool = True,
) -> cv2.BackgroundSubtractorMOG2:
    """Create a MOG2 background subtractor tuned for indoor scenes."""
    return cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=detect_shadows
    )


def clean_foreground_mask(
    raw_mask: np.ndarray, kernel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn a raw MOG2 mask (0 / 127 / 255) into a usable foreground mask.

    Drops shadow pixels (127), then opens the result with ``kernel`` to
    remove speckle noise. Returns ``(fg, clean)`` — both uint8 single-channel.
    """
    fg = (raw_mask == 255).astype(np.uint8) * 255
    clean = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    return fg, clean


def process_frame(
    frame: np.ndarray,
    bg_sub: cv2.BackgroundSubtractorMOG2,
    kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply background subtraction, then clean the mask with morphology.

    Returns ``(fg_raw, fg_clean)`` — both uint8 single-channel masks.
    """
    raw = bg_sub.apply(frame)
    return clean_foreground_mask(raw, kernel)
