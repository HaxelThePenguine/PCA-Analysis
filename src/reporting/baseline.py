"""Charts and console output for baseline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.pca import format_pca_summary
from utils.plotting import save_figure, style_axis

COLORS = {"cov": "#2F6B9A", "corr": "#C4933F", "grid": "#D9DEE5"}


def save_line_plot(
    filename: str,
    title: str,
    ylabel: str,
    series: Sequence[tuple[str, np.ndarray, str]],
    *,
    cumulative: bool = False,
    out_dir: Path,
) -> None:
    components = np.arange(1, len(series[0][1]) + 1)
    fig, axis = plt.subplots(figsize=(8, 5))
    for label, values, color in series:
        y = values.cumsum() * 100 if cumulative else values * 100
        axis.plot(components, y, marker="o", color=color, label=label)
    if cumulative:
        axis.axhline(
            80, color="#7B8794", linestyle="--", linewidth=0.9, label="80% threshold"
        )
        axis.set_ylim(0, 105)
    axis.set_title(title)
    axis.set_xlabel("Number of components" if cumulative else "Principal component")
    axis.set_ylabel(ylabel)
    axis.set_xticks(components)
    axis.legend(frameon=False)
    style_axis(axis, grid_color=COLORS["grid"])
    fig.tight_layout()
    save_figure(fig, out_dir / filename, tight_bbox=True)


def save_pc1_plot(
    cov_loadings: pd.DataFrame, corr_loadings: pd.DataFrame, *, out_dir: Path
) -> None:
    order = corr_loadings["PC1"].sort_values().index
    positions = np.arange(len(order))
    width = 0.38
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(
        positions - width / 2,
        cov_loadings.loc[order, "PC1"],
        height=width,
        color=COLORS["cov"],
        label="Covariance",
    )
    axis.barh(
        positions + width / 2,
        corr_loadings.loc[order, "PC1"],
        height=width,
        color=COLORS["corr"],
        label="Correlation",
    )
    axis.axvline(0, color="#1F2933", linewidth=0.8)
    axis.set_title("Baseline PCA: PC1 eigenvector weights")
    axis.set_xlabel("Eigenvector weight")
    axis.set_yticks(positions)
    axis.set_yticklabels(order)
    axis.legend(frameon=False)
    style_axis(axis, grid_axis="x", grid_color=COLORS["grid"])
    fig.tight_layout()
    save_figure(fig, out_dir / "07_pc1_loadings.png", tight_bbox=True)


def print_summary(
    returns,
    covariance_result,
    correlation_result,
    ticker_summary,
    covariance_diff,
    correlation_diff,
    score_error,
    *,
    out_dir: Path,
) -> None:
    print("=== BASELINE PCA CORE ===")
    print(f"Panel: {returns.shape} | NaN: 0")
    print(f"Date: {returns.index.min()} -> {returns.index.max()}")
    print(f"Covariance check: {covariance_diff:.3e}")
    print(f"Score/eigenvalue check: {score_error:.3e}")
    print(
        f"First 3 components: {100 * covariance_result.explained[:3].sum():.4f}% explained"
    )
    print(f"Correlation check: {correlation_diff:.3e}")
    print(f"Correlation eigenvalue sum: {correlation_result.eigenvalues.sum():.12f}")
    print("\n=== COVARIANCE PCA ===")
    print(format_pca_summary(covariance_result.summary))
    print("\nPC1-PC3 loadings:")
    print(covariance_result.weights.iloc[:, :3].round(4).to_string())
    print("\n=== CORRELATION PCA ===")
    print(format_pca_summary(correlation_result.summary))
    print("\nPC1-PC3 loadings:")
    print(correlation_result.weights.iloc[:, :3].round(4).to_string())
    print("\n=== EXPLAINED VARIANCE BY STOCK, FIRST 3 PCS ===")
    print(ticker_summary[["explained_pct"]].round(2).to_string())
    print(f"\nOutputs saved to: {out_dir}")


def make_plots(covariance_result, correlation_result, *, out_dir: Path) -> None:
    """Render the baseline comparison from two already fitted PCA results."""
    series = [
        ("Covariance", covariance_result.explained, COLORS["cov"]),
        ("Correlation", correlation_result.explained, COLORS["corr"]),
    ]
    save_line_plot(
        "07_scree_variance.png",
        "Baseline PCA: explained variance by component",
        "Explained variance (%)",
        series,
        out_dir=out_dir,
    )
    save_line_plot(
        "07_cumulative_variance.png",
        "Baseline PCA: cumulative explained variance",
        "Cumulative explained variance (%)",
        series,
        cumulative=True,
        out_dir=out_dir,
    )
    save_pc1_plot(
        covariance_result.weights.iloc[:, :3],
        correlation_result.weights.iloc[:, :3],
        out_dir=out_dir,
    )
