"""
src.vision — reusable computer-vision pipeline components.

Extracted from experiments day 24-26 so future scripts can import just the
pieces they need instead of re-implementing them per experiment.

Public API (both import styles work):
    from src.vision.video import get_video_metadata
    from src.vision import get_video_metadata
"""

from src.vision.background import (
    clean_foreground_mask,
    init_background_subtractor,
    process_frame,
)
from src.vision.objects import (
    draw_objects,
    draw_trajectories,
    find_objects,
    update_trajectories,
)
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
    "clean_foreground_mask",
    "create_writer",
    "draw_objects",
    "draw_trajectories",
    "even_size",
    "find_objects",
    "get_video_metadata",
    "init_background_subtractor",
    "iter_kept_frames",
    "open_video",
    "process_frame",
    "update_trajectories",
]
