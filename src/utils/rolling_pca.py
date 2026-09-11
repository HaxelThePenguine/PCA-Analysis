"""Session-based rolling PCA and within-window benchmark diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import BENCHMARKS, CORE_UNIVERSE
from utils.benchmark import residualize_against_benchmarks
from utils.pca import fit_pca
from utils.rolling import rolling_starts, session_index, session_window_positions

WINDOWS = (20, 60)
STEP_SESSIONS = 5
PCA_METHODS = ("correlation", "covariance")
STOCKS = list(CORE_UNIVERSE)


def residualize_window(
    window_panel: pd.DataFrame,
    stocks=CORE_UNIVERSE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove the SPY/XLF exposure estimated inside one rolling window."""

    benchmark_result = residualize_against_benchmarks(
        window_panel,
        stocks=stocks,
    )
    residuals = benchmark_result["residual_returns"]
    diagnostics = benchmark_result["diagnostics"][
        ["r_squared", "variance_removed_pct", "residual_std"]
    ].join(benchmark_result["coefficients"])
    diagnostics["max_abs_residual_benchmark_corr"] = 0.0
    for benchmark in BENCHMARKS:
        diagnostics["max_abs_residual_benchmark_corr"] = np.maximum(
            diagnostics["max_abs_residual_benchmark_corr"],
            residuals.corrwith(window_panel[benchmark]).abs(),
        )
    return residuals, diagnostics


def collect_rolling_results(
    complete_panel: pd.DataFrame,
    normalized_returns: pd.DataFrame,
    *,
    stocks=CORE_UNIVERSE,
    window_sizes=WINDOWS,
    step_sessions: int = 5,
    methods=PCA_METHODS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run both PCA variants for each transformation and rolling window."""

    sessions, codes = session_index(complete_panel.index)
    stocks = tuple(stocks)
    if len(stocks) < 3:
        raise ValueError("Rolling PC1-PC3 diagnostics require at least three stocks.")
    if not complete_panel.index.equals(normalized_returns.index):
        raise ValueError("Raw and normalized panels must have identical row indices.")
    if list(normalized_returns.columns) != list(stocks):
        raise ValueError("Normalized columns must match the requested stock order.")
    metrics = []
    loadings = []
    benchmark_diagnostics = []
    previous_pc1 = {}

    for window_size in window_sizes:
        for start in rolling_starts(len(sessions), window_size, step_sessions):
            end = start + window_size - 1
            positions = session_window_positions(codes, start, window_size)
            window_panel = complete_panel.iloc[positions]
            window_raw = window_panel[list(stocks)]
            window_normalized = normalized_returns.iloc[positions]
            window_residuals, diagnostics = residualize_window(window_panel, stocks)
            window_start = str(sessions[start])
            window_end = str(sessions[end])

            for stock, values in diagnostics.iterrows():
                benchmark_diagnostics.append(
                    {
                        "window_size_sessions": window_size,
                        "window_start": window_start,
                        "window_end": window_end,
                        "stock": stock,
                        **values.to_dict(),
                    }
                )

            transformed_data = {
                "raw": window_raw,
                "intraday_normalized": window_normalized,
                "spy_xlf_residual": window_residuals,
            }
            benchmark_removed = diagnostics["variance_removed_pct"].mean()
            max_residual_corr = diagnostics["max_abs_residual_benchmark_corr"].max()

            for transformation, data in transformed_data.items():
                for method in methods:
                    pca_result = fit_pca(data, method=method)
                    values = pca_result.eigenvalues
                    vectors = pca_result.eigenvectors
                    explained = pca_result.explained
                    effective_dimension = 1.0 / np.square(explained).sum()
                    key = (window_size, transformation, method)
                    similarity = np.nan
                    if key in previous_pc1:
                        similarity = abs(np.dot(previous_pc1[key], vectors[:, 0]))
                    previous_pc1[key] = vectors[:, 0]
                    benchmark_removed_value = (
                        benchmark_removed
                        if transformation == "spy_xlf_residual"
                        else np.nan
                    )
                    residual_corr_value = (
                        max_residual_corr
                        if transformation == "spy_xlf_residual"
                        else np.nan
                    )

                    metrics.append(
                        {
                            "window_size_sessions": window_size,
                            "window_start": window_start,
                            "window_end": window_end,
                            "n_sessions": window_size,
                            "n_observations": len(window_panel),
                            "transformation": transformation,
                            "pca_method": method,
                            "pc1_eigenvalue": values[0],
                            "pc2_eigenvalue": values[1],
                            "pc3_eigenvalue": values[2],
                            "pc1_explained_pct": 100 * explained[0],
                            "pc2_explained_pct": 100 * explained[1],
                            "pc3_explained_pct": 100 * explained[2],
                            "first3_cumulative_pct": 100 * explained[:3].sum(),
                            "effective_dimension": effective_dimension,
                            "pc1_similarity_to_previous": similarity,
                            "mean_benchmark_variance_removed_pct": benchmark_removed_value,
                            "max_abs_residual_benchmark_corr": residual_corr_value,
                        }
                    )

                    for stock_position, stock in enumerate(stocks):
                        loadings.append(
                            {
                                "window_size_sessions": window_size,
                                "window_start": window_start,
                                "window_end": window_end,
                                "transformation": transformation,
                                "pca_method": method,
                                "stock": stock,
                                "pc1_loading": vectors[stock_position, 0],
                                "pc2_loading": vectors[stock_position, 1],
                                "pc3_loading": vectors[stock_position, 2],
                            }
                        )

    return (
        pd.DataFrame(metrics),
        pd.DataFrame(loadings),
        pd.DataFrame(benchmark_diagnostics),
    )
