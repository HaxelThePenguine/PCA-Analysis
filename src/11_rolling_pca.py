"""Descriptive rolling covariance and correlation PCA on the CORE panel."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from benchmark_utils import residualize_against_benchmarks
from config import (
    BENCHMARKS,
    CORE_UNIVERSE,
    REPORTS_DIR,
    RETURN_MATRIX_CLEAN_FILE,
    ensure_project_directories,
)
from data_utils import load_panel, normalize_intraday_volatility
from pca_utils import fit_pca
from plotting_utils import save_figure


OUT_DIR = REPORTS_DIR / "rolling_pca"
WINDOWS = (20, 60)
STEP_SESSIONS = 5
TRANSFORMATIONS = (
    "raw",
    "intraday_normalized",
    "spy_xlf_residual",
)
PCA_METHODS = ("correlation", "covariance")
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
STOCKS = list(CORE_UNIVERSE)


def build_intraday_normalized_returns(
    stock_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize each ticker by its full-sample minute-of-day volatility profile."""

    return normalize_intraday_volatility(stock_returns).normalized_returns


def residualize_window(
    window_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove the SPY/XLF exposure estimated inside one rolling window."""

    benchmark_result = residualize_against_benchmarks(
        window_panel,
        stocks=STOCKS,
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


def session_index(index: pd.DatetimeIndex) -> tuple[pd.Index, np.ndarray]:
    """Return unique trading sessions and an integer session code per row."""

    dates = np.asarray(index.date)
    sessions, codes = np.unique(dates, return_inverse=True)
    return pd.Index(sessions, name="session_date"), codes


def rolling_starts(n_sessions: int, window_size: int) -> list[int]:
    """Return regular starts and always include the final trailing window."""

    last_start = n_sessions - window_size
    if last_start < 0:
        return []
    starts = list(range(0, last_start + 1, STEP_SESSIONS))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def collect_rolling_results(
    complete_panel: pd.DataFrame,
    normalized_returns: pd.DataFrame,
    sessions: pd.Index,
    codes: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run both PCA variants for each transformation and rolling window."""

    metrics = []
    loadings = []
    benchmark_diagnostics = []
    previous_pc1 = {}

    for window_size in WINDOWS:
        for start in rolling_starts(len(sessions), window_size):
            end = start + window_size - 1
            positions = np.flatnonzero((codes >= start) & (codes <= end))
            window_panel = complete_panel.iloc[positions]
            window_raw = window_panel[STOCKS]
            window_normalized = normalized_returns.iloc[positions]
            window_residuals, diagnostics = residualize_window(window_panel)
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
            max_residual_corr = diagnostics[
                "max_abs_residual_benchmark_corr"
            ].max()

            for transformation, data in transformed_data.items():
                for method in PCA_METHODS:
                    pca_result = fit_pca(data, method=method)
                    values = pca_result.eigenvalues
                    vectors = pca_result.eigenvectors
                    explained = pca_result.explained
                    effective_dimension = 1.0 / np.square(explained).sum()
                    key = (window_size, transformation, method)
                    similarity = np.nan
                    if key in previous_pc1:
                        similarity = abs(
                            np.dot(previous_pc1[key], vectors[:, 0])
                        )
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

                    for stock_position, stock in enumerate(STOCKS):
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


def style_axis(axis: Axes) -> None:
    """Apply the project style plus concise date formatting."""

    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axis.xaxis.get_major_locator())
    )


def plot_correlation_metrics(metrics: pd.DataFrame) -> None:
    """Plot the main rolling correlation-PCA diagnostics."""

    correlation = metrics[metrics["pca_method"] == "correlation"]
    plot_specs = [
        ("pc1_explained_pct", "PC1 explained variance (%)"),
        ("first3_cumulative_pct", "First 3 PCs explained variance (%)"),
        ("effective_dimension", "Effective cross-sectional dimension"),
    ]
    figure, axes = plt.subplots(
        len(WINDOWS),
        len(plot_specs),
        figsize=(15, 8),
        sharex="col",
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
                    pd.to_datetime(series["window_end"]),
                    series[metric],
                    color=TRANSFORM_COLORS[transformation],
                    linewidth=1.4,
                    label=TRANSFORM_LABELS[transformation],
                )
            axis.set_title(f"{window_size}-session window")
            axis.set_ylabel(ylabel)
            style_axis(axis)
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
        "Rolling correlation PCA diagnostics",
        x=0.06,
        ha="left",
        y=1.04,
        fontsize=14,
    )
    figure.text(
        0.06,
        1.005,
        "Descriptive estimates; windows overlap and are not independent.",
        color="#666666",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(
        figure,
        OUT_DIR / "11_rolling_correlation_metrics.png",
        tight_bbox=True,
    )


def plot_loading_stability(metrics: pd.DataFrame) -> None:
    """Plot sign-invariant PC1 similarity to the previous rolling window."""

    correlation = metrics[metrics["pca_method"] == "correlation"]
    similarities = correlation["pc1_similarity_to_previous"].dropna()
    lower_limit = max(
        0.0,
        np.floor((similarities.min() - 0.005) * 100) / 100,
    )
    figure, axes = plt.subplots(
        len(WINDOWS),
        1,
        figsize=(11, 6),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for axis, window_size in zip(axes, WINDOWS):
        subset = correlation[
            correlation["window_size_sessions"] == window_size
        ].sort_values("window_end")
        for transformation in TRANSFORMATIONS:
            series = subset[subset["transformation"] == transformation]
            axis.plot(
                pd.to_datetime(series["window_end"]),
                series["pc1_similarity_to_previous"],
                color=TRANSFORM_COLORS[transformation],
                linewidth=1.4,
                label=TRANSFORM_LABELS[transformation],
            )
        axis.set_title(f"{window_size}-session window")
        axis.set_ylabel("Absolute loading similarity")
        axis.set_ylim(lower_limit, 1.005)
        style_axis(axis)
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
        "Rolling PC1 loading stability",
        x=0.08,
        ha="left",
        y=1.14,
        fontsize=14,
    )
    figure.text(
        0.08,
        1.09,
        "Absolute dot product; 1 = identical. First window has no comparison.",
        color="#666666",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(
        figure,
        OUT_DIR / "11_pc1_loading_stability.png",
        tight_bbox=True,
    )


def plot_benchmark_variance_removed(metrics: pd.DataFrame) -> None:
    """Plot the average rolling variance share removed by SPY and XLF."""

    residual = metrics[
        (metrics["pca_method"] == "correlation")
        & (metrics["transformation"] == "spy_xlf_residual")
    ]
    figure, axis = plt.subplots(figsize=(11, 4.8))
    for window_size, color in zip(WINDOWS, ("#2F6690", "#D99A2B")):
        subset = residual[
            residual["window_size_sessions"] == window_size
        ].sort_values("window_end")
        axis.plot(
            pd.to_datetime(subset["window_end"]),
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
    style_axis(axis)
    figure.tight_layout()
    save_figure(
        figure,
        OUT_DIR / "11_benchmark_variance_removed.png",
        tight_bbox=True,
    )


def main() -> None:
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    required = STOCKS + list(BENCHMARKS)
    panel = load_panel(
        RETURN_MATRIX_CLEAN_FILE,
        required,
        context="Complete benchmark panel",
        drop_incomplete=True,
    )

    sessions, codes = session_index(panel.index)
    if len(sessions) < max(WINDOWS):
        raise ValueError("The panel does not contain enough sessions for rolling PCA.")

    normalized_returns = build_intraday_normalized_returns(panel[STOCKS])
    metrics, loadings, benchmark_diagnostics = collect_rolling_results(
        panel,
        normalized_returns,
        sessions,
        codes,
    )

    metrics.to_csv(OUT_DIR / "11_rolling_pca_metrics.csv", index=False)
    loadings.to_csv(OUT_DIR / "11_rolling_pca_loadings.csv", index=False)
    benchmark_diagnostics.to_csv(
        OUT_DIR / "11_rolling_benchmark_diagnostics.csv",
        index=False,
    )
    plot_correlation_metrics(metrics)
    plot_loading_stability(metrics)
    plot_benchmark_variance_removed(metrics)

    primary = metrics[metrics["pca_method"] == "correlation"]
    summary = (
        primary.groupby(["window_size_sessions", "transformation"])[
            [
                "pc1_explained_pct",
                "first3_cumulative_pct",
                "effective_dimension",
            ]
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
            f"{size} sessions ({len(rolling_starts(len(sessions), size))} windows)"
            for size in WINDOWS
        )
    )
    print("\n=== CORRELATION PCA ROLLING SUMMARY ===")
    print(summary.to_string())
    print(
        "\nBootstrap confidence bands are intentionally not implemented yet; "
        "the current outputs are descriptive rolling estimates."
    )
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
