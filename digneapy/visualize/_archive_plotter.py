#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   _archive_plotter.py
@Time    :   2026/05/21 11:02:49
@Author  :   Alejandro Marrero (amarrerd@ull.edu.es)
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2026, Alejandro Marrero
@Desc    :   None
"""

from typing import Optional, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from digneapy.archives import GridArchive


def _archive_to_matrix(archive: GridArchive, attr: str = "p") -> np.ndarray:
    """Converts a 2d GridArchive to a (rows × cols) float matrix.

    Empty cells are filled with NaN so they render as a neutral colour.

    Args:
        archive: A GridArchive with exactly 2 dimensions.
        attr:    Instance attribute to use as cell value. Defaults to ``"p"``.

    Returns:
        2d numpy array of shape ``(dims[0], dims[1])``.
    """
    rows, cols = int(archive.dimensions[0]), int(archive.dimensions[1])
    matrix = np.full((rows, cols), np.nan, dtype=np.float64)

    if len(archive) == 0:
        return matrix

    flat_indices = np.array(list(archive.filled_cells))
    _instances = archive.retrieve_filled_cells(flat_indices)
    grid_indices = archive.int_to_grid_index(flat_indices)  # (n, 2)
    values = np.asarray(
        [getattr(instance, attr) for instance in _instances],
        dtype=np.float64,
    )
    matrix[grid_indices[:, 0], grid_indices[:, 1]] = values
    return matrix


def _axis_tick_labels(
    archive: GridArchive, dim: int, n_ticks: int = 5
) -> tuple[list[float], list[str]]:
    """Returns (positions, labels) for cell-index ticks on one axis."""
    n_cells = int(archive.dimensions[dim])
    lb = archive._lower_bounds[dim]
    ub = archive._upper_bounds[dim]
    step = max(1, n_cells // (n_ticks - 1))
    positions = list(range(0, n_cells, step))
    if positions[-1] != n_cells - 1:
        positions.append(n_cells - 1)
    labels = [f"{lb + (ub - lb) * p / (n_cells - 1):.2f}" for p in positions]
    return positions, labels


class ArchivePlotter:
    """Maintains a live matplotlib figure showing the GridArchive as a heatmap.

    The colour of each cell encodes a chosen scalar attribute of its elite
    (default: ``p``, the performance bias).  Empty cells are shown in a
    distinct neutral colour so it is easy to see how the archive fills up.

    Args:
        archive (GridArchive): The 2-D archive to visualise.
        attr (str): Instance attribute to use as colour value. Default ``"p"``.
        feat_names (Sequence[str]): Labels for the two feature axes.
        cmap (str): Matplotlib colormap name. Default ``"viridis"``.
        vmin / vmax (float | None): Fixed colour scale limits.  If ``None``
            (default), the scale is recomputed each frame from the data.
        figsize (tuple[float, float]): Figure size in inches.
        title (str): Figure window / suptitle text.
    """

    def __init__(
        self,
        archive: GridArchive,
        attr: str = "p",
        feat_names: Optional[Sequence[str]] = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        figsize: tuple[float, float] = (7, 6),
        title: str = "MAP-Elites Archive",
    ):
        if len(archive.dimensions) != 2:
            raise ValueError(
                "ArchivePlotter only supports 2d GridArchives. "
                f"Got dimensions={archive.dimensions}"
            )

        self._archive = archive
        self._attr = attr
        self._feat_names = feat_names or ["Feature 0", "Feature 1"]
        self._cmap = cmap
        self._vmin = vmin
        self._vmax = vmax

        self._fig, self._ax = plt.subplots(figsize=figsize)
        self._fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.ion()  # non-blocking interactive mode

        rows, cols = int(archive.dimensions[0]), int(archive.dimensions[1])
        empty = np.full((rows, cols), np.nan)

        # Empty-cell background (neutral grey)
        bg_cmap = mcolors.ListedColormap(["#d0d0d0"])
        self._ax.imshow(
            np.zeros_like(empty),
            cmap=bg_cmap,
            aspect="auto",
            origin="lower",
            extent=[-0.5, cols - 0.5, -0.5, rows - 0.5],
        )

        # Elite heatmap layer (NaN = transparent → shows grey background)
        cm = plt.get_cmap(self._cmap).copy()
        cm.set_bad(color="none")  # NaN → transparent
        self._im = self._ax.imshow(
            empty,
            cmap=cm,
            aspect="auto",
            origin="lower",
            extent=[-0.5, cols - 0.5, -0.5, rows - 0.5],
            interpolation="nearest",
        )

        # Colour bar
        self._cbar = self._fig.colorbar(self._im, ax=self._ax, pad=0.02)
        self._cbar.set_label(f"Elite  '{attr}'  value", fontsize=10)

        # Axis labels & ticks
        self._ax.set_xlabel(self._feat_names[0], fontsize=11)
        self._ax.set_ylabel(self._feat_names[1], fontsize=11)

        x_pos, x_lbl = _axis_tick_labels(archive, dim=0)
        y_pos, y_lbl = _axis_tick_labels(archive, dim=1)
        self._ax.set_xticks(x_pos)
        self._ax.set_xticklabels(x_lbl, fontsize=8)
        self._ax.set_yticks(y_pos)
        self._ax.set_yticklabels(y_lbl, fontsize=8)

        # Stats text box (top-left inside axes)
        self._info = self._ax.text(
            0.02,
            0.97,
            "",
            transform=self._ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

        self._fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    def update(self, generation: int = 0) -> None:
        """Redraws the heatmap with the current state of the archive.

        Call this once per generation inside your evolution loop.

        Args:
            generation: Current generation number (shown in the stats box).
        """
        matrix = _archive_to_matrix(self._archive, self._attr)

        vmin = (
            self._vmin
            if self._vmin is not None
            else np.nanmin(matrix)
            if not np.all(np.isnan(matrix))
            else 0.0
        )
        vmax = (
            self._vmax
            if self._vmax is not None
            else np.nanmax(matrix)
            if not np.all(np.isnan(matrix))
            else 1.0
        )

        self._im.set_data(matrix)
        self._im.set_clim(vmin=vmin, vmax=vmax)

        filled = len(self._archive)
        total = int(self._archive.n_cells)
        coverage = 100 * filled / total if total > 0 else 0.0

        non_nan = matrix[~np.isnan(matrix)]
        mean_p = float(np.mean(non_nan)) if non_nan.size else 0.0
        max_p = float(np.max(non_nan)) if non_nan.size else 0.0

        self._info.set_text(
            f"Generation : {generation}\n"
            f"Cells      : {filled} / {total}  ({coverage:.1f}%)\n"
            f"Mean p     : {mean_p:.4f}\n"
            f"Max  p     : {max_p:.4f}"
        )
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def save(self, path: str, dpi: int = 150) -> None:
        """Saves the current figure to *path*.

        Args:
            path: Output file path (e.g. ``"archive_gen200.png"``).
            dpi:  Resolution. Default 150.
        """
        self._fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[ArchivePlotter] Saved → {path}")

    def show(self) -> None:
        """Blocks until the figure window is closed (call at end of run)."""
        plt.ioff()
        plt.show()

    def close(self) -> None:
        """Closes the figure programmatically."""
        plt.close(self._fig)
