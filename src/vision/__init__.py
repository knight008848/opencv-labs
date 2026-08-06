"""
src.vision — reusable computer-vision pipeline components.

Extracted from experiments day 24-26 so future scripts can import just the
pieces they need instead of re-implementing them per experiment.

Public API (both import styles work):
    from src.vision.video import get_video_metadata
    from src.vision import get_video_metadata
"""

from src.vision.video import (
    DEFAULT_FOURCC,
    annotate_frame,
    create_writer,
    even_size,
    get_video_metadata,
    iter_kept_frames,
    open_video,
)

__all__ = [
    "DEFAULT_FOURCC",
    "annotate_frame",
    "create_writer",
    "even_size",
    "get_video_metadata",
    "iter_kept_frames",
    "open_video",
]
