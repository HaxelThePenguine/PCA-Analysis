"""Small plotting helpers shared by the analysis scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.dates as mdates
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DEFAULT_GRID_COLOR = "#D9DEE5"


def style_axis(
    axis: Axes,
    *,
    grid_axis: Literal["x", "y", "both"] = "y",
    grid_color: str = DEFAULT_GRID_COLOR,
    grid_linewidth: float = 0.8,
    format_dates: bool = False,
) -> None:
    """Apply the project's restrained chart styling to one axis."""

    axis.grid(axis=grid_axis, color=grid_color, linewidth=grid_linewidth)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if format_dates:
        locator = mdates.AutoDateLocator()
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def save_figure(
    figure: Figure,
    path: Path,
    *,
    dpi: int = 180,
    tight_bbox: bool = False,
) -> None:
    """Save and close a Matplotlib figure."""

    from matplotlib import pyplot as plt

    options = {"bbox_inches": "tight"} if tight_bbox else {}
    figure.savefig(path, dpi=dpi, **options)
    plt.close(figure)
