"""Charts and console output for variance."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plotting import save_figure, style_axis
from utils.variance import GROUP_COLUMNS

GROUP_LABELS = [
    "Benchmark SPY/XLF",
    "Residual PC1",
    "Residuals PC2-PC3",
    "Residuals PC4-PC12",
]
GROUP_COLORS = ["#2f6690", "#d99a2b", "#e07a5f", "#9aa58b"]


def plot_stock_decomposition(
    ledger: pd.DataFrame, complete_panel: pd.DataFrame, *, out_dir: Path
) -> None:
    """Save a 100% stacked bar chart for the twelve stocks."""
    figure, axis = plt.subplots(figsize=(11, 7))
    left = np.zeros(len(ledger))
    for column, label, color in zip(GROUP_COLUMNS, GROUP_LABELS, GROUP_COLORS):
        values = ledger[column].to_numpy()
        axis.barh(
            ledger.index,
            values,
            left=left,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        left += values
    start = complete_panel.index[0].strftime("%Y-%m-%d")
    end = complete_panel.index[-1].strftime("%Y-%m-%d")
    axis.set_title(
        "Variance decomposition by stock", loc="left", color="#252525", pad=18
    )
    axis.text(
        0,
        1.015,
        f"Raw variance share | complete panel {start}-{end} | n={len(complete_panel):,}",
        transform=axis.transAxes,
        color="#666666",
        fontsize=9,
    )
    axis.set_xlabel("Percentage of raw variance")
    axis.set_xlim(0, 100)
    axis.set_xticks([0, 25, 50, 75, 100])
    style_axis(axis, grid_axis="x", grid_color="#d9d9d9", grid_linewidth=0.7)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    figure.tight_layout()
    save_figure(figure, out_dir / "10_variance_decomposition_by_stock.png")


def plot_global_decomposition(summary: pd.DataFrame, *, out_dir: Path) -> None:
    """Save the same decomposition aggregated across the full universe."""
    figure, axis = plt.subplots(figsize=(10, 4.0))
    left = 0.0
    for label, color in zip(GROUP_LABELS, GROUP_COLORS):
        value = summary.loc[label, "share_of_raw_pct"]
        axis.barh(
            ["CORE universe"],
            [value],
            left=left,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        if value >= 7:
            axis.text(
                left + value / 2,
                0,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="#252525",
            )
        left += value
    axis.set_title(
        "Aggregate variance decomposition", loc="left", color="#252525", pad=18
    )
    axis.text(
        0,
        1.08,
        "Denominator: sum of raw variances across the twelve stocks",
        transform=axis.transAxes,
        color="#666666",
        fontsize=9,
    )
    axis.set_xlim(0, 100)
    axis.set_xlabel("Percentage of raw variance")
    axis.set_xticks([0, 25, 50, 75, 100])
    style_axis(axis, grid_axis="x", grid_color="#d9d9d9", grid_linewidth=0.7)
    figure.legend(loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=2, frameon=False)
    figure.subplots_adjust(left=0.13, right=0.98, top=0.78, bottom=0.34)
    save_figure(figure, out_dir / "10_variance_decomposition_global.png")


def print_summary(
    complete_panel, global_summary, ledger, results, *, out_dir: Path
) -> None:
    print("=== VARIANCE DECOMPOSITION ===")
    print(f"Complete panel: {complete_panel.shape}")
    print(f"Period: {complete_panel.index[0]} -> {complete_panel.index[-1]}")
    print("\nRaw variance share by stock (%):")
    print(ledger[GROUP_COLUMNS + ["total_pct"]].round(2).to_string())
    print("\nAggregate decomposition (% of total raw variance):")
    print(global_summary[["share_of_raw_pct"]].round(2).to_string())
    print(
        "\nMaximum error in raw = benchmark + residual identity:",
        f"{ledger['variance_identity_error'].max():.3e}",
    )
    residual_component_error = (
        (results["component_variance"].sum(axis=1) - results["residual_variance"])
        .abs()
        .max()
    )
    print(
        "Maximum error in residual component sum = residual variance:",
        f"{residual_component_error:.3e}",
    )
    print(f"\nOutputs saved to: {out_dir}")
