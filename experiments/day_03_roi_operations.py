"""
Day 03: ROI Operations
Date: 2026-06-11
Goal: Understand ROI operations — grid split, shuffle, border, reassemble
Runtime: < 1 s
"""

import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def split_grid(img: np.ndarray, rows: int = 3, cols: int = 3) -> list[list[np.ndarray]]:
    """Slice image into a rows×cols 2D grid, discarding excess edge pixels."""
    h, w = img.shape[:2]
    h_aligned = h - (h % rows)
    w_aligned = w - (w % cols)
    trimmed = img[:h_aligned, :w_aligned]

    block_h, block_w = h_aligned // rows, w_aligned // cols
    tiles = [[None] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * block_h, (r + 1) * block_h
            x0, x1 = c * block_w, (c + 1) * block_w
            tiles[r][c] = trimmed[y0:y1, x0:x1]
    print(
        f"split: {rows}×{cols} grid, tile {block_h}×{block_w}  "
        f"(dropped {h - h_aligned} rows, {w - w_aligned} cols)"
    )
    return tiles


def shuffle_and_border(
    tiles: list[list[np.ndarray]],
    border: int = 5,
    palette: list[tuple[int, int, int]] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    """Randomly shuffle tile positions, add colored borders.

    By default uses the global random state — each run produces a different
    shuffle and different border colors.  Pass ``seed`` for reproducibility.

    Args:
        tiles: 2D grid of image tiles.
        border: Border width in pixels.
        palette: Optional list of BGR color tuples, one per original tile position.
                 If None or too short, missing colors are auto-generated randomly.
        seed: If given, uses a dedicated Generator for reproducible output.

    Returns:
        (flat_indices, bordered_2d_grid).
    """
    rows, cols = len(tiles), len(tiles[0])
    n = rows * cols

    # Use global PRNG by default (fresh shuffle each run); local Generator if seed given
    rng = np.random.default_rng(seed) if seed is not None else None

    # Build palette: auto-generate random colors if not provided (or too short)
    if palette is None:
        palette = [
            tuple(np.random.randint(0, 256, 3).tolist())
            if rng is None
            else tuple(rng.integers(0, 256, 3).tolist())
            for _ in range(n)
        ]
    elif len(palette) < n:
        missing = n - len(palette)
        extra = [
            tuple(np.random.randint(0, 256, 3).tolist())
            if rng is None
            else tuple(rng.integers(0, 256, 3).tolist())
            for _ in range(missing)
        ]
        palette = list(palette) + extra

    flat_indices = np.random.permutation(n) if rng is None else rng.permutation(n)
    bordered = [[None] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            src_idx = flat_indices[r * cols + c]
            src_row, src_col = src_idx // cols, src_idx % cols
            color = palette[src_idx]
            bordered[r][c] = cv2.copyMakeBorder(
                tiles[src_row][src_col],
                border,
                border,
                border,
                border,
                cv2.BORDER_CONSTANT,
                value=color,
            )
    return flat_indices, bordered


def assemble(bordered: list[list[np.ndarray]]) -> np.ndarray:
    """Reassemble a 2D grid of tiles into a single image."""
    return np.vstack([np.hstack(row) for row in bordered])


def main():
    result_dir = Path(__file__).resolve().parent / "results"
    frame_path = result_dir / "frame_1008.jpg"

    if not frame_path.exists():
        print(f"Error: Could not find frame at {frame_path}")
        sys.exit(1)

    img = cv2.imread(str(frame_path))
    print(f"shape: {img.shape}")

    tiles = split_grid(img, rows=3, cols=3)
    flat_indices, bordered = shuffle_and_border(tiles, border=5)
    puzzle = assemble(bordered)

    puzzle_path = result_dir / "day03_puzzle.jpg"
    cv2.imwrite(str(puzzle_path), puzzle)
    print(f"Saved puzzle to {puzzle_path}  ({puzzle.shape[1]}×{puzzle.shape[0]})")

    # Display and save shuffled tiles
    fig, axes = plt.subplots(3, 3, figsize=(10, 6))
    for idx, src_idx in enumerate(flat_indices):
        r, c = idx // 3, idx % 3
        ax = axes[r, c]
        ax.imshow(cv2.cvtColor(bordered[r][c], cv2.COLOR_BGR2RGB))
        ax.set_title(f"from [{src_idx // 3},{src_idx % 3}]")
        ax.axis("off")
    fig.suptitle("Shuffled 3×3 Grid with Colored Borders")
    plt.tight_layout()

    grid_path = result_dir / "day03_shuffled_grid.jpg"
    fig.savefig(str(grid_path), dpi=120, bbox_inches="tight")
    print(f"Saved shuffled grid to {grid_path}")

    if os.environ.get("DISPLAY"):
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
