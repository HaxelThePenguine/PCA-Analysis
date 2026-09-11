"""Charts and console output for rolling pca."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plotting import save_figure, style_axis
from utils.rolling import rolling_starts
from utils.rolling_pca import STEP_SESSIONS, WINDOWS

TRANSFORMATIONS = ("raw", "intraday_normalized", "spy_xlf_residual")
TRANSFORM_LABELS = {
    "raw": "Raw",
    "intraday_normalized": "Intraday-normalized",
    "spy_xlf_residual": "SPY/XLF residual",
}
TRANSFORM_COLORS = {
    "raw": "#2F6690",
    "intraday_normalized": "#D99A2B",
    "spy_xlf_residual": "#E07A5F",
}
GRID_COLOR = "#D9DEE5"


def plot_correlation_metrics(metrics: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot the main rolling correlation-PCA diagnostics."""
    correlation = metrics[metrics["pca_method"] == "correlation"]
    plot_specs = [
        ("pc1_explained_pct", "PC1 explained variance (%)"),
        ("first3_cumulative_pct", "First 3 PCs explained variance (%)"),
        ("effective_dimension", "Effective cross-sectional dimension"),
    ]
    figure, axes = plt.subplots(
        len(WINDOWS), len(plot_specs), figsize=(15, 8), sharex="col"
    )
    axes = np.atleast_2d(axes)
    for row, window_size in enumerate(WINDOWS):
        subset = correlation[
            correlation["window_size_sessions"] == window_size
        ].sort_values("window_end")
        for column, (metric, ylabel) in enumerate(plot_specs):
            axis = axes[row, column]
            for transformation in TRANSFORMATIONS:
                series = subset[subset["transformation"] == transformation]
                axis.plot(
                    pd.to_datetime(series["window_end"], utc=True),
                    series[metric],
                    color=TRANSFORM_COLORS[transformation],
                    linewidth=1.4,
                    label=TRANSFORM_LABELS[transformation],
                )
            axis.set_title(f"{window_size}-session window")
            axis.set_ylabel(ylabel)
            style_axis(axis, format_dates=True)
            if metric != "effective_dimension":
                axis.set_ylim(bottom=0)
            if row == len(WINDOWS) - 1:
                axis.set_xlabel("Window end")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Rolling correlation PCA diagnostics", x=0.06, ha="left", y=1.04, fontsize=14
    )
    figure.text(
        0.06,
        1.005,
        "Descriptive estimates; windows overlap and are not independent.",
        color="#666666",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(figure, out_dir / "11_rolling_correlation_metrics.png", tight_bbox=True)


def plot_loading_stability(metrics: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot sign-invariant PC1 similarity to the previous rolling window."""
    correlation = metrics[metrics["pca_method"] == "correlation"]
    similarities = correlation["pc1_similarity_to_previous"].dropna()
    lower_limit = max(0.0, np.floor((similarities.min() - 0.005) * 100) / 100)
    figure, axes = plt.subplots(len(WINDOWS), 1, figsize=(11, 6), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, window_size in zip(axes, WINDOWS):
        subset = correlation[
            correlation["window_size_sessions"] == window_size
        ].sort_values("window_end")
        for transformation in TRANSFORMATIONS:
            series = subset[subset["transformation"] == transformation]
            axis.plot(
                pd.to_datetime(series["window_end"], utc=True),
                series["pc1_similarity_to_previous"],
                color=TRANSFORM_COLORS[transformation],
                linewidth=1.4,
                label=TRANSFORM_LABELS[transformation],
            )
        axis.set_title(f"{window_size}-session window")
        axis.set_ylabel("Absolute loading similarity")
        axis.set_ylim(lower_limit, 1.005)
        style_axis(axis, format_dates=True)
    axes[-1].set_xlabel("Window end")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Rolling PC1 loading stability", x=0.08, ha="left", y=1.14, fontsize=14
    )
    figure.text(
        0.08,
        1.09,
        "Absolute dot product; 1 = identical. First window has no comparison.",
        color="#666666",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.9])
    save_figure(figure, out_dir / "11_pc1_loading_stability.png", tight_bbox=True)


def plot_benchmark_variance_removed(metrics: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot the average rolling variance share removed by SPY and XLF."""
    residual = metrics[
        (metrics["pca_method"] == "correlation")
        & (metrics["transformation"] == "spy_xlf_residual")
    ]
    figure, axis = plt.subplots(figsize=(11, 4.8))
    for window_size, color in zip(WINDOWS, ("#2F6690", "#D99A2B")):
        subset = residual[residual["window_size_sessions"] == window_size].sort_values(
            "window_end"
        )
        axis.plot(
            pd.to_datetime(subset["window_end"], utc=True),
            subset["mean_benchmark_variance_removed_pct"],
            color=color,
            linewidth=1.5,
            label=f"{window_size}-session window",
        )
    axis.set_title("Rolling variance share removed by SPY/XLF")
    axis.set_xlabel("Window end")
    axis.set_ylabel("Mean variance removed (%)")
    axis.set_ylim(0, 100)
    axis.legend(frameon=False)
    style_axis(axis, format_dates=True)
    figure.tight_layout()
    save_figure(figure, out_dir / "11_benchmark_variance_removed.png", tight_bbox=True)


def print_summary(metrics, panel, sessions, *, out_dir: Path) -> None:
    primary = metrics[metrics["pca_method"] == "correlation"]
    summary = (
        primary.groupby(["window_size_sessions", "transformation"])[
            ["pc1_explained_pct", "first3_cumulative_pct", "effective_dimension"]
        ]
        .agg(["mean", "min", "max"])
        .round(4)
    )
    print("=== ROLLING PCA ===")
    print(f"Complete panel: {panel.shape}")
    print(f"Sessions: {len(sessions)}")
    print(f"Period: {panel.index[0]} -> {panel.index[-1]}")
    print(f"Session step: {STEP_SESSIONS}")
    print("Intraday normalization: fixed full-sample profile (descriptive only)")
    print(
        "Windows: "
        + ", ".join(
            (
                f"{size} sessions ({len(rolling_starts(len(sessions), size))} windows)"
                for size in WINDOWS
            )
        )
    )
    print("\n=== CORRELATION PCA ROLLING SUMMARY ===")
    print(summary.to_string())
    print(
        "\nBootstrap confidence bands are intentionally not implemented yet; the current outputs are descriptive rolling estimates."
    )
    print(f"\nOutputs saved to: {out_dir}")
