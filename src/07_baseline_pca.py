"""Baseline covariance/correlation PCA on the CORE return panel."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CORE_UNIVERSE,
    REPORTS_DIR,
    RETURN_CORE_FILE,
    ensure_project_directories,
)
from data_utils import load_panel, save_csv_tables
from pca_utils import fit_pca, format_pca_summary
from plotting_utils import save_figure, style_axis


OUT_DIR = REPORTS_DIR / "pca_baseline"
COLORS = {"cov": "#2F6B9A", "corr": "#C4933F", "grid": "#D9DEE5"}


def save_line_plot(
    filename: str,
    title: str,
    ylabel: str,
    series: Sequence[tuple[str, np.ndarray, str]],
    *,
    cumulative: bool = False,
) -> None:
    components = np.arange(1, len(series[0][1]) + 1)
    fig, axis = plt.subplots(figsize=(8, 5))

    for label, values, color in series:
        y = values.cumsum() * 100 if cumulative else values * 100
        axis.plot(components, y, marker="o", color=color, label=label)

    if cumulative:
        axis.axhline(
            80,
            color="#7B8794",
            linestyle="--",
            linewidth=0.9,
            label="80% threshold",
        )
        axis.set_ylim(0, 105)

    axis.set_title(title)
    axis.set_xlabel("Number of components" if cumulative else "Principal component")
    axis.set_ylabel(ylabel)
    axis.set_xticks(components)
    axis.legend(frameon=False)
    style_axis(axis, grid_color=COLORS["grid"])
    fig.tight_layout()
    save_figure(fig, OUT_DIR / filename, tight_bbox=True)


def save_pc1_plot(
    cov_loadings: pd.DataFrame,
    corr_loadings: pd.DataFrame,
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
    save_figure(fig, OUT_DIR / "07_pc1_loadings.png", tight_bbox=True)


def main() -> None:
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    returns = load_panel(
        RETURN_CORE_FILE,
        CORE_UNIVERSE,
        context="CORE panel",
    )

    means = returns.mean()
    stds = returns.std(ddof=1)
    X = returns.subtract(means, axis="columns")

    covariance = (X.T @ X) / (len(X) - 1)
    covariance_diff = (covariance - returns.cov()).abs().to_numpy().max()
    covariance_result = fit_pca(returns, method="covariance")
    cov_values = covariance_result.eigenvalues
    cov_vectors = covariance_result.eigenvectors
    cov_explained = covariance_result.explained
    cov_summary = covariance_result.summary
    cov_loadings = covariance_result.weights.iloc[:, :3]

    scores = covariance_result.scores.to_numpy()
    score_error = np.max(
        np.abs(cov_values - pd.DataFrame(scores).var(ddof=1).to_numpy())
    )
    reconstruction = scores[:, :3] @ cov_vectors[:, :3].T
    residuals = X.to_numpy() - reconstruction
    reconstruction_pct = 100 * (
        1 - np.sum(residuals**2) / np.sum(X.to_numpy() ** 2)
    )
    residual_df = pd.DataFrame(residuals, index=X.index, columns=X.columns)
    ticker_summary = pd.DataFrame(
        {
            "original_variance": X.var(ddof=1),
            "residual_variance": residual_df.var(ddof=1),
        }
    )
    ticker_summary["explained_pct"] = 100 * (
        1
        - ticker_summary["residual_variance"]
        / ticker_summary["original_variance"]
    )

    correlation_result = fit_pca(returns, method="correlation")
    correlation = correlation_result.matrix
    correlation_diff = (correlation - returns.corr()).abs().to_numpy().max()
    corr_values = correlation_result.eigenvalues
    corr_explained = correlation_result.explained
    corr_summary = correlation_result.summary
    corr_loadings = correlation_result.weights.iloc[:, :3]

    tables = {
        "07_return_stats.csv": pd.DataFrame({"mean": means, "std": stds}),
        "07_covariance_matrix.csv": covariance,
        "07_correlation_matrix.csv": correlation,
        "07_covariance_summary.csv": cov_summary,
        "07_correlation_summary.csv": corr_summary,
        "07_covariance_loadings_pc1_pc3.csv": cov_loadings,
        "07_correlation_loadings_pc1_pc3.csv": corr_loadings,
        "07_covariance_technical_loadings_pc1_pc3.csv": covariance_result.loadings.iloc[:, :3],
        "07_correlation_technical_loadings_pc1_pc3.csv": correlation_result.loadings.iloc[:, :3],
        "07_ticker_reconstruction.csv": ticker_summary,
    }
    save_csv_tables(tables, OUT_DIR)

    series = [
        ("Covariance", cov_explained, COLORS["cov"]),
        ("Correlation", corr_explained, COLORS["corr"]),
    ]
    save_line_plot(
        "07_scree_variance.png",
        "Baseline PCA: explained variance by component",
        "Explained variance (%)",
        series,
    )
    save_line_plot(
        "07_cumulative_variance.png",
        "Baseline PCA: cumulative explained variance",
        "Cumulative explained variance (%)",
        series,
        cumulative=True,
    )
    save_pc1_plot(cov_loadings, corr_loadings)

    print("=== BASELINE PCA CORE ===")
    print(f"Panel: {returns.shape} | NaN: 0")
    print(f"Date: {returns.index.min()} -> {returns.index.max()}")
    print(f"Covariance check: {covariance_diff:.3e}")
    print(f"Score/eigenvalue check: {score_error:.3e}")
    print(f"First 3 components: {reconstruction_pct:.4f}% explained")
    print(f"Correlation check: {correlation_diff:.3e}")
    print(f"Correlation eigenvalue sum: {corr_values.sum():.12f}")

    print("\n=== COVARIANCE PCA ===")
    print(format_pca_summary(cov_summary))
    print("\nPC1-PC3 loadings:")
    print(cov_loadings.round(4).to_string())

    print("\n=== CORRELATION PCA ===")
    print(format_pca_summary(corr_summary))
    print("\nPC1-PC3 loadings:")
    print(corr_loadings.round(4).to_string())

    print("\n=== EXPLAINED VARIANCE BY STOCK, FIRST 3 PCS ===")
    print(ticker_summary[["explained_pct"]].round(2).to_string())
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
