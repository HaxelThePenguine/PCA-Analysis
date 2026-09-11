"""Charts and console output for local factors."""

from __future__ import annotations

from pathlib import Path

import matplotlib

from utils.plotting import save_figure, style_axis

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from utils.local_factor import (
    BOOTSTRAP_REPLICATIONS,
    N_COMPONENTS,
    PRIMARY_STARTS,
    STOCKS,
)

GRID_COLOR = "#D9DEE5"
COLORS = {"raw": "#2F6690", "benchmark_residual": "#C26A2E"}


def plot_loading_comparison(loadings: pd.DataFrame, *, out_dir: Path) -> None:
    """Compare principal-component and L1-rotated structural loadings."""
    panels = []
    maximum = 0.0
    for transformation in ("raw", "benchmark_residual"):
        for method in ("principal_component_basis", "l1_rotation"):
            subset = loadings[
                (loadings["transformation"] == transformation)
                & (loadings["method"] == method)
            ]
            pivot = subset.pivot(
                index="stock", columns="factor_number", values="loading"
            ).reindex(STOCKS)
            panels.append((transformation, method, pivot))
            maximum = max(maximum, float(np.abs(pivot.to_numpy()).max()))
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    norm_scale = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    image = None
    for axis, (transformation, method, pivot) in zip(axes.ravel(), panels):
        image = axis.imshow(
            pivot.to_numpy(), cmap="RdBu_r", norm=norm_scale, aspect="auto"
        )
        axis.set_title(
            f"{transformation.replace('_', ' ').title()}\n{('PC basis' if method == 'principal_component_basis' else 'L1 rotation')}"
        )
        axis.set_xticks(
            range(N_COMPONENTS), [f"F{i}" for i in range(1, N_COMPONENTS + 1)]
        )
        axis.set_yticks(range(len(STOCKS)), STOCKS)
        axis.tick_params(axis="both", length=0)
        for row in range(len(STOCKS)):
            for column in range(N_COMPONENTS):
                axis.text(
                    column,
                    row,
                    f"{pivot.iloc[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#252525",
                )
        for spine in axis.spines.values():
            spine.set_visible(False)
    colorbar_axis = figure.add_axes([0.86, 0.16, 0.022, 0.68])
    figure.colorbar(image, cax=colorbar_axis, label="Structural loading")
    figure.suptitle(
        "L1 rotation reveals local loading directions", x=0.06, ha="left", y=0.99
    )
    figure.text(
        0.06,
        0.95,
        "Columns have norm sqrt(n); L1 rotation changes coordinates, not the selected PCA subspace.",
        fontsize=9,
        color="#666666",
    )
    figure.subplots_adjust(
        left=0.08, right=0.82, bottom=0.07, top=0.87, wspace=0.22, hspace=0.28
    )
    save_figure(figure, out_dir / "13_l1_loading_comparison.png", tight_bbox=True)


def plot_bootstrap_support(probabilities: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot session-bootstrap support probabilities for residual local factors."""
    pivot = probabilities.pivot(
        index="stock", columns="factor", values="active_probability"
    ).reindex(STOCKS)
    figure, axis = plt.subplots(figsize=(7.2, 7.0))
    image = axis.imshow(
        pivot.to_numpy(), cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto"
    )
    axis.set_xticks(range(N_COMPONENTS), pivot.columns)
    axis.set_yticks(range(len(STOCKS)), STOCKS)
    axis.tick_params(axis="both", length=0)
    for row in range(len(STOCKS)):
        for column in range(N_COMPONENTS):
            probability = pivot.iloc[row, column]
            axis.text(
                column,
                row,
                f"{probability:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if probability > 0.58 else "#252525",
            )
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.colorbar(
        image, ax=axis, fraction=0.045, pad=0.03, label="Active-loading probability"
    )
    figure.suptitle(
        "Residual local-factor support stability", x=0.08, ha="left", y=0.98
    )
    figure.text(
        0.08,
        0.93,
        f"{BOOTSTRAP_REPLICATIONS} whole-session bootstrap replications; factors aligned by cosine similarity.",
        fontsize=9,
        color="#666666",
    )
    figure.subplots_adjust(left=0.13, right=0.88, bottom=0.08, top=0.87)
    save_figure(
        figure, out_dir / "13_bootstrap_support_probability.png", tight_bbox=True
    )


def plot_factor_diagnostics(
    factor_counts: pd.DataFrame, stability: pd.DataFrame, *, out_dir: Path
) -> None:
    """Plot eigenvalue-ratio context and bootstrap factor-direction stability."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for transformation, color in COLORS.items():
        subset = factor_counts[factor_counts["transformation"] == transformation]
        axes[0].plot(
            subset["component"],
            subset["explained_pct"],
            marker="o",
            color=color,
            label=transformation.replace("_", " ").title(),
        )
    axes[0].axvline(N_COMPONENTS, color="#777777", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Factor-count context")
    axes[0].legend(frameon=False)
    style_axis(axes[0])
    groups = [
        stability.loc[stability["factor"] == factor, "cosine_similarity"].to_numpy()
        for factor in [f"LF{i}" for i in range(1, N_COMPONENTS + 1)]
    ]
    box = axes[1].boxplot(groups, patch_artist=True, tick_labels=["LF1", "LF2", "LF3"])
    for patch in box["boxes"]:
        patch.set_facecolor("#9EC5E5")
        patch.set_edgecolor("#2F6690")
    axes[1].axhline(0.9, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("Absolute loading cosine")
    axes[1].set_title("Session-bootstrap direction stability")
    style_axis(axes[1])
    figure.suptitle("Identification diagnostics", x=0.06, ha="left", y=0.98)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.83, wspace=0.25)
    save_figure(figure, out_dir / "13_identification_diagnostics.png", tight_bbox=True)


def print_summary(fits, k_sensitivity, panel, stability, *, out_dir: Path) -> None:
    print("=== L1 LOCAL FACTOR IDENTIFICATION ===")
    print(f"Complete panel: {panel.shape}")
    print(f"Period: {panel.index[0]} -> {panel.index[-1]}")
    print(
        f"Primary K={N_COMPONENTS}; starts={PRIMARY_STARTS}; session bootstraps={BOOTSTRAP_REPLICATIONS}"
    )
    for transformation, result in fits.items():
        test = result.local_factor_test
        print(f"\n--- {transformation.replace('_', ' ').title()} ---")
        print(
            f"Local factor detected: {test.has_local_factors} | small counts={test.small_counts.tolist()} | critical count>{test.gamma_n}"
        )
        print(f"L1 norms: {result.l1_norms.round(4).tolist()}")
        print(
            f"PCA-subspace reconstruction: {result.reconstruction_pct:.4f}% | max equivalence error={result.reconstruction_max_abs_error:.3e}"
        )
        print("Rotated structural loadings:")
        print(result.rotated_loadings.round(3).to_string())
    summary = stability.groupby("factor").agg(
        median_cosine=("cosine_similarity", "median"),
        fifth_pct_cosine=("cosine_similarity", lambda values: values.quantile(0.05)),
        median_jaccard=("support_jaccard", "median"),
        local_detection_rate=("has_local_factor", "mean"),
    )
    print("\n--- Residual session-bootstrap stability ---")
    print(summary.round(3).to_string())
    print("\nK sensitivity:")
    print(k_sensitivity.round(4).to_string(index=False))
    print(f"\nOutputs saved to: {out_dir}")
