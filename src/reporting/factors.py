"""Charts and console output for factors."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from config import CORE_UNIVERSE
from utils.factor import (
    ELASTIC_NET_L2,
    L1_GRID,
    N_COMPONENTS,
    SELECTED_L1,
    ZERO_TOLERANCE,
)
from utils.plotting import save_figure, style_axis

STOCKS = list(CORE_UNIVERSE)
GRID_COLOR = "#D9DEE5"
METHOD_COLORS = {"Lasso": "#2F6690", "Elastic Net": "#D99A2B"}


def plot_loading_heatmaps(loadings: pd.DataFrame, *, out_dir: Path) -> None:
    """Compare standard, rotated, and sparse loading structures."""
    transformations = ["raw", "benchmark_residual"]
    methods = ["standard_pca", "varimax", "elastic_net_spca"]
    method_titles = {
        "standard_pca": "Standard correlation PCA",
        "varimax": "Varimax",
        "elastic_net_spca": "Elastic-Net Sparse PCA",
    }
    panels = []
    maximum = 0.0
    for transformation in transformations:
        for method in methods:
            subset = loadings[
                (loadings["transformation"] == transformation)
                & (loadings["method"] == method)
            ]
            pivot = subset.pivot(
                index="stock", columns="component_number", values="loading"
            ).reindex(STOCKS)
            panels.append((transformation, method, pivot))
            maximum = max(maximum, np.nanmax(np.abs(pivot.to_numpy())))
    figure, axes = plt.subplots(
        len(transformations), len(methods), figsize=(14, 9), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    image = None
    for axis, (transformation, method, pivot) in zip(axes.ravel(), panels):
        image = axis.imshow(pivot.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")
        axis.set_title(
            f"{transformation.replace('_', ' ').title()}\n{method_titles[method]}"
        )
        axis.set_xticks(range(N_COMPONENTS))
        axis.set_xticklabels([f"F{i}" for i in range(1, N_COMPONENTS + 1)])
        axis.set_yticks(range(len(STOCKS)))
        axis.set_yticklabels(STOCKS)
        for row in range(len(STOCKS)):
            for column in range(N_COMPONENTS):
                value = pivot.iloc[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#252525",
                )
        axis.tick_params(axis="both", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        fraction=0.025,
        pad=0.02,
        label="Loading",
    )
    figure.suptitle(
        "Internal factor composition: standard, rotated, and sparse views",
        x=0.06,
        ha="left",
        y=0.985,
        fontsize=14,
    )
    figure.text(
        0.06,
        0.945,
        "Loadings are shown after correlation scaling; signs are oriented for readability.",
        color="#666666",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.07, right=0.9, bottom=0.08, top=0.83, wspace=0.26, hspace=0.3
    )
    save_figure(figure, out_dir / "12_factor_loading_heatmaps.png", tight_bbox=True)


def plot_sparse_path(path: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot the sparsity/reconstruction trade-off for both penalties."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for axis, transformation in zip(axes, ["raw", "benchmark_residual"]):
        subset = path[path["transformation"] == transformation]
        values_for_axis = []
        for method in ("Lasso", "Elastic Net"):
            method_subset = (
                subset[subset["method"] == method]
                .groupby("l1_penalty", as_index=False)
                .agg(
                    nonzero_count=("nonzero_count", "mean"),
                    reconstruction_pct=("reconstruction_pct_first_3", "first"),
                )
                .sort_values("l1_penalty")
            )
            axis.plot(
                method_subset["l1_penalty"],
                method_subset["reconstruction_pct"],
                marker="o",
                linewidth=1.5,
                color=METHOD_COLORS[method],
                label=method,
            )
            values_for_axis.extend(method_subset["reconstruction_pct"].tolist())
        axis.set_title(transformation.replace("_", " ").title())
        axis.set_xlabel("L1 penalty")
        axis.set_xticks(L1_GRID)
        lower = min(values_for_axis) - 0.03
        upper = max(values_for_axis) + 0.03
        axis.set_ylim(lower, upper)
        axis.set_ylabel("Reconstruction variance retained (%)")
        style_axis(axis, grid_color=GRID_COLOR)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Sparse-factor trade-off across penalty choices",
        x=0.08,
        ha="left",
        y=0.98,
        fontsize=14,
    )
    figure.text(
        0.08,
        0.91,
        "Higher L1 creates more zero weights; Elastic Net adds an L2 grouping term.",
        color="#666666",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.74, wspace=0.12)
    save_figure(figure, out_dir / "12_sparse_penalty_path.png", tight_bbox=True)


def print_summary(fitted, panel, *, out_dir: Path) -> None:
    print("=== INTERNAL FACTOR ISOLATION ===")
    print(f"Complete panel: {panel.shape}")
    print(f"Period: {panel.index[0]} -> {panel.index[-1]}")
    print("Primary method: correlation PCA")
    print(
        f"Selected Sparse PCA penalties: L1={SELECTED_L1:.2f}, L2={ELASTIC_NET_L2:.2f}"
    )
    for transformation, result in fitted.items():
        pca_result = result["pca"]
        lasso = result["lasso_spca"]
        elastic_net = result["elastic_net_spca"]
        print(f"\n--- {transformation.replace('_', ' ').title()} ---")
        print(
            f"Standard PCA first 3 components: {100 * pca_result.explained[:N_COMPONENTS].sum():.4f}%"
        )
        print(
            f"Varimax first 3-component subspace: {100 * pca_result.explained[:N_COMPONENTS].sum():.4f}%"
        )
        print(
            f"Lasso non-zero weights: {int((lasso.weights.abs() > ZERO_TOLERANCE).sum().sum())} | reconstruction: {lasso.reconstruction_pct:.4f}%"
        )
        print(
            f"Elastic-Net non-zero weights: {int((elastic_net.weights.abs() > ZERO_TOLERANCE).sum().sum())} | reconstruction: {elastic_net.reconstruction_pct:.4f}%"
        )
        print("Elastic-Net score correlation:")
        print(elastic_net.score_correlation.round(3).to_string())
    print(
        "\nSparse PCA note: reconstruction percentages are reported instead of ordinary PCA explained-variance shares because sparse factors are penalized."
    )
    print(f"\nOutputs saved to: {out_dir}")
