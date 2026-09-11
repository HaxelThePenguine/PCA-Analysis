"""PCA, Varimax, and sparse-factor comparisons with penalty-path diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.pca import PCAResult, fit_elastic_net_sparse_pca, fit_pca, fit_varimax

N_COMPONENTS = 3


SELECTED_L1 = 0.10


ELASTIC_NET_L2 = 0.10


L1_GRID = (0.02, 0.05, 0.10, 0.15, 0.20)
ZERO_TOLERANCE = 1e-8


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
                    "is_nonzero": abs(weights.loc[stock, component]) > ZERO_TOLERANCE,
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
        upper = score_correlation.to_numpy()[
            np.triu_indices(score_correlation.shape[0], k=1)
        ]
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

    views = [
        (
            "standard_pca",
            pca_result.weights.iloc[:, :N_COMPONENTS],
            pca_result.loadings.iloc[:, :N_COMPONENTS],
            pca_result.scores.iloc[:, :N_COMPONENTS],
            100 * pca_result.explained[:N_COMPONENTS].sum(),
            pca_result.explained,
        ),
        (
            "varimax",
            varimax_result.rotated_weights,
            varimax_result.rotated_loadings,
            varimax_result.scores,
            100 * pca_result.explained[:N_COMPONENTS].sum(),
            None,
        ),
        (
            "lasso_spca",
            lasso_result.weights,
            lasso_result.score_loadings,
            lasso_result.scores,
            lasso_result.reconstruction_pct,
            None,
        ),
        (
            "elastic_net_spca",
            elastic_net_result.weights,
            elastic_net_result.score_loadings,
            elastic_net_result.scores,
            elastic_net_result.reconstruction_pct,
            None,
        ),
    ]
    loading_rows_result, metric_rows, score_frames = [], [], []
    for method, weights, loadings, scores, retained, explained in views:
        loading_rows_result.extend(
            loading_rows(transformation, method, weights, loadings)
        )
        metric_rows.extend(
            method_metrics(
                transformation,
                method,
                list(weights.columns),
                weights,
                retained,
                scores.corr(),
                explained,
            )
        )
        score_frames.append(scores.add_prefix(f"{transformation}_{method}_"))

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
