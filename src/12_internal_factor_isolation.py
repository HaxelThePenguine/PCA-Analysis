"""Compare PCA, Varimax, and Elastic-Net Sparse PCA factor structures."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from benchmark_utils import ResidualizationResult, residualize_against_benchmarks
from config import (
    BENCHMARKS,
    CORE_UNIVERSE,
    REPORTS_DIR,
    RETURN_MATRIX_CLEAN_FILE,
    ensure_project_directories,
)
from data_utils import load_panel
from pca_utils import (
    PCAResult,
    fit_elastic_net_sparse_pca,
    fit_pca,
    fit_varimax,
)
from plotting_utils import save_figure, style_axis


OUT_DIR = REPORTS_DIR / "internal_factor_isolation"
STOCKS = list(CORE_UNIVERSE)
N_COMPONENTS = 3
SELECTED_L1 = 0.10
ELASTIC_NET_L2 = 0.10
L1_GRID = (0.02, 0.05, 0.10, 0.15, 0.20)
ZERO_TOLERANCE = 1e-8
GRID_COLOR = "#D9DEE5"
METHOD_COLORS = {
    "Lasso": "#2F6690",
    "Elastic Net": "#D99A2B",
}


def load_complete_panel() -> pd.DataFrame:
    """Load the complete CORE plus benchmark panel used by this study."""

    required = STOCKS + list(BENCHMARKS)
    return load_panel(
        RETURN_MATRIX_CLEAN_FILE,
        required,
        context="Complete benchmark panel",
        drop_incomplete=True,
    )


def build_factor_panels(
    panel: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], ResidualizationResult]:
    """Return raw and benchmark-residualized stock panels."""

    residualization = residualize_against_benchmarks(panel, stocks=STOCKS)
    return {
        "raw": panel.loc[:, STOCKS],
        "benchmark_residual": residualization["residual_returns"],
    }, residualization


def loading_rows(
    transformation: str,
    method: str,
    weights: pd.DataFrame,
    loadings: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert a factor loading matrix into a tidy table."""

    rows = []
    for component_number, component in enumerate(weights.columns, start=1):
        for stock in weights.index:
            rows.append(
                {
                    "transformation": transformation,
                    "method": method,
                    "component": component,
                    "component_number": component_number,
                    "stock": stock,
                    "weight": weights.loc[stock, component],
                    "loading": loadings.loc[stock, component],
                    "absolute_loading": abs(loadings.loc[stock, component]),
                    "is_nonzero": abs(weights.loc[stock, component])
                    > ZERO_TOLERANCE,
                }
            )
    return rows


def method_metrics(
    transformation: str,
    method: str,
    component_names: list[str],
    weights: pd.DataFrame,
    reconstruction_pct: float,
    score_correlation: pd.DataFrame,
    explained: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Summarize interpretability and reconstruction diagnostics."""

    rows = []
    max_score_correlation = np.nan
    if score_correlation.shape[0] > 1:
        upper = score_correlation.to_numpy()[np.triu_indices(
            score_correlation.shape[0], k=1
        )]
        if len(upper):
            max_score_correlation = np.max(np.abs(upper))

    for component_number, component in enumerate(component_names):
        explained_pct = np.nan
        cumulative_pct = np.nan
        if explained is not None:
            explained_pct = 100 * explained[component_number]
            cumulative_pct = 100 * explained[: component_number + 1].sum()
        rows.append(
            {
                "transformation": transformation,
                "method": method,
                "component": component,
                "component_number": component_number + 1,
                "explained_pct": explained_pct,
                "cumulative_pct": cumulative_pct,
                "reconstruction_pct_first_3": reconstruction_pct,
                "nonzero_count": int(
                    (weights.iloc[:, component_number].abs() > ZERO_TOLERANCE).sum()
                ),
                "max_abs_factor_score_correlation": max_score_correlation,
            }
        )
    return rows


def fit_transformation(
    transformation: str,
    data: pd.DataFrame,
) -> dict[str, Any]:
    """Fit all requested factor views for one stock transformation."""

    pca_result = fit_pca(data, method="correlation")
    varimax_result = fit_varimax(pca_result, n_components=N_COMPONENTS)
    lasso_result = fit_elastic_net_sparse_pca(
        pca_result.analysis_data,
        n_components=N_COMPONENTS,
        l1_penalty=SELECTED_L1,
        l2_penalty=0.0,
        initial_pca=pca_result,
    )
    elastic_net_result = fit_elastic_net_sparse_pca(
        pca_result.analysis_data,
        n_components=N_COMPONENTS,
        l1_penalty=SELECTED_L1,
        l2_penalty=ELASTIC_NET_L2,
        initial_pca=pca_result,
    )

    loading_rows_result = []
    loading_rows_result.extend(
        loading_rows(
            transformation,
            "standard_pca",
            pca_result.weights.iloc[:, :N_COMPONENTS],
            pca_result.loadings.iloc[:, :N_COMPONENTS],
        )
    )
    loading_rows_result.extend(
        loading_rows(
            transformation,
            "varimax",
            varimax_result.rotated_weights,
            varimax_result.rotated_loadings,
        )
    )
    loading_rows_result.extend(
        loading_rows(
            transformation,
            "lasso_spca",
            lasso_result.weights,
            lasso_result.score_loadings,
        )
    )
    loading_rows_result.extend(
        loading_rows(
            transformation,
            "elastic_net_spca",
            elastic_net_result.weights,
            elastic_net_result.score_loadings,
        )
    )

    metric_rows = []
    metric_rows.extend(
        method_metrics(
            transformation,
            "standard_pca",
            [f"PC{i}" for i in range(1, N_COMPONENTS + 1)],
            pca_result.weights.iloc[:, :N_COMPONENTS],
            100 * pca_result.explained[:N_COMPONENTS].sum(),
            pca_result.scores.iloc[:, :N_COMPONENTS].corr(),
            explained=pca_result.explained,
        )
    )
    metric_rows.extend(
        method_metrics(
            transformation,
            "varimax",
            list(varimax_result.rotated_weights.columns),
            varimax_result.rotated_weights,
            100 * pca_result.explained[:N_COMPONENTS].sum(),
            varimax_result.scores.corr(),
        )
    )
    for method, sparse_result in (
        ("lasso_spca", lasso_result),
        ("elastic_net_spca", elastic_net_result),
    ):
        metric_rows.extend(
            method_metrics(
                transformation,
                method,
                list(sparse_result.weights.columns),
                sparse_result.weights,
                sparse_result.reconstruction_pct,
                sparse_result.score_correlation,
            )
        )

    score_tables = {
        "standard_pca": pca_result.scores.iloc[:, :N_COMPONENTS],
        "varimax": varimax_result.scores,
        "lasso_spca": lasso_result.scores,
        "elastic_net_spca": elastic_net_result.scores,
    }
    score_frames = []
    for method, scores in score_tables.items():
        frame = scores.copy()
        frame.columns = [f"{transformation}_{method}_{column}" for column in scores]
        score_frames.append(frame)

    return {
        "pca": pca_result,
        "varimax": varimax_result,
        "lasso_spca": lasso_result,
        "elastic_net_spca": elastic_net_result,
        "loading_rows": loading_rows_result,
        "metric_rows": metric_rows,
        "score_frame": pd.concat(score_frames, axis=1),
    }


def fit_sparse_penalty_path(
    transformation: str,
    data: pd.DataFrame,
    pca_result: PCAResult,
) -> list[dict[str, Any]]:
    """Evaluate Lasso and Elastic-Net sparsity/reconstruction trade-offs."""

    rows = []
    for method, l2_penalty in (
        ("Lasso", 0.0),
        ("Elastic Net", ELASTIC_NET_L2),
    ):
        for l1_penalty in L1_GRID:
            result = fit_elastic_net_sparse_pca(
                pca_result.analysis_data,
                n_components=N_COMPONENTS,
                l1_penalty=l1_penalty,
                l2_penalty=l2_penalty,
                initial_pca=pca_result,
            )
            for component_number, component in enumerate(result.weights.columns):
                rows.append(
                    {
                        "transformation": transformation,
                        "method": method,
                        "l1_penalty": l1_penalty,
                        "l2_penalty": l2_penalty,
                        "component": component,
                        "nonzero_count": int(
                            (
                                result.weights.iloc[:, component_number].abs()
                                > ZERO_TOLERANCE
                            ).sum()
                        ),
                        "reconstruction_pct_first_3": result.reconstruction_pct,
                        "converged": result.converged,
                        "iterations": result.iterations,
                    }
                )
    return rows


def plot_loading_heatmaps(loadings: pd.DataFrame) -> None:
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
                index="stock",
                columns="component_number",
                values="loading",
            ).reindex(STOCKS)
            panels.append((transformation, method, pivot))
            maximum = max(maximum, np.nanmax(np.abs(pivot.to_numpy())))

    figure, axes = plt.subplots(
        len(transformations),
        len(methods),
        figsize=(14, 9),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    image = None
    for axis, (transformation, method, pivot) in zip(axes.ravel(), panels):
        image = axis.imshow(
            pivot.to_numpy(),
            cmap="RdBu_r",
            norm=norm,
            aspect="auto",
        )
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
        left=0.07,
        right=0.90,
        bottom=0.08,
        top=0.83,
        wspace=0.26,
        hspace=0.30,
    )
    save_figure(
        figure,
        OUT_DIR / "12_factor_loading_heatmaps.png",
        tight_bbox=True,
    )


def plot_sparse_path(path: pd.DataFrame) -> None:
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
                    reconstruction_pct=(
                        "reconstruction_pct_first_3",
                        "first",
                    ),
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
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.16,
        top=0.74,
        wspace=0.12,
    )
    save_figure(
        figure,
        OUT_DIR / "12_sparse_penalty_path.png",
        tight_bbox=True,
    )


def main() -> None:
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_complete_panel()
    factor_panels, residualization = build_factor_panels(panel)
    all_loading_rows = []
    all_metric_rows = []
    all_path_rows = []
    score_frames = []
    rotation_tables = []
    fitted = {}

    for transformation, data in factor_panels.items():
        result = fit_transformation(transformation, data)
        fitted[transformation] = result
        all_loading_rows.extend(result["loading_rows"])
        all_metric_rows.extend(result["metric_rows"])
        all_path_rows.extend(
            fit_sparse_penalty_path(
                transformation,
                data,
                result["pca"],
            )
        )
        score_frames.append(result["score_frame"])

        rotation = result["varimax"].rotation.copy()
        rotation.insert(0, "transformation", transformation)
        rotation_tables.append(rotation.reset_index(names="source_component"))

    loadings = pd.DataFrame(all_loading_rows)
    metrics = pd.DataFrame(all_metric_rows)
    sparse_path = pd.DataFrame(all_path_rows)
    scores = pd.concat(score_frames, axis=1)
    rotations = pd.concat(rotation_tables, ignore_index=True)

    loadings.to_csv(OUT_DIR / "12_factor_loadings.csv", index=False)
    metrics.to_csv(OUT_DIR / "12_factor_metrics.csv", index=False)
    sparse_path.to_csv(OUT_DIR / "12_sparse_penalty_path.csv", index=False)
    rotations.to_csv(OUT_DIR / "12_varimax_rotation.csv", index=False)
    scores.to_parquet(OUT_DIR / "12_factor_scores.parquet")
    residualization["coefficients"].to_csv(
        OUT_DIR / "12_benchmark_coefficients.csv"
    )
    residualization["diagnostics"].to_csv(
        OUT_DIR / "12_benchmark_diagnostics.csv"
    )

    plot_loading_heatmaps(loadings)
    plot_sparse_path(sparse_path)

    print("=== INTERNAL FACTOR ISOLATION ===")
    print(f"Complete panel: {panel.shape}")
    print(f"Period: {panel.index[0]} -> {panel.index[-1]}")
    print("Primary method: correlation PCA")
    print(
        "Selected Sparse PCA penalties: "
        f"L1={SELECTED_L1:.2f}, L2={ELASTIC_NET_L2:.2f}"
    )
    for transformation, result in fitted.items():
        pca_result = result["pca"]
        lasso = result["lasso_spca"]
        elastic_net = result["elastic_net_spca"]
        print(f"\n--- {transformation.replace('_', ' ').title()} ---")
        print(
            "Standard PCA first 3 components: "
            f"{100 * pca_result.explained[:N_COMPONENTS].sum():.4f}%"
        )
        print(
            "Varimax first 3-component subspace: "
            f"{100 * pca_result.explained[:N_COMPONENTS].sum():.4f}%"
        )
        print(
            "Lasso non-zero weights: "
            f"{int((lasso.weights.abs() > ZERO_TOLERANCE).sum().sum())}"
            f" | reconstruction: {lasso.reconstruction_pct:.4f}%"
        )
        print(
            "Elastic-Net non-zero weights: "
            f"{int((elastic_net.weights.abs() > ZERO_TOLERANCE).sum().sum())}"
            f" | reconstruction: {elastic_net.reconstruction_pct:.4f}%"
        )
        print("Elastic-Net score correlation:")
        print(elastic_net.score_correlation.round(3).to_string())

    print(
        "\nSparse PCA note: reconstruction percentages are reported instead of "
        "ordinary PCA explained-variance shares because sparse factors are penalized."
    )
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
