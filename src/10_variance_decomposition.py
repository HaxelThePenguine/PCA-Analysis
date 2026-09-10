"""Decompose stock variance into SPY/XLF and residual PCA components."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
from pca_utils import fit_pca


OUT_DIR = REPORTS_DIR / "variance_decomposition"
GROUP_COLUMNS = [
    "benchmark_pct",
    "residual_PC1_pct",
    "residual_PC2_to_PC3_pct",
    "residual_PC4_to_PC12_pct",
]
GROUP_LABELS = [
    "Benchmark SPY/XLF",
    "Residual PC1",
    "Residuals PC2-PC3",
    "Residuals PC4-PC12",
]
GROUP_COLORS = ["#2f6690", "#d99a2b", "#e07a5f", "#9aa58b"]


def build_variance_ledger(complete_panel, stocks):
    """Fit benchmarks and split each stock's variance into additive parts."""

    benchmark_result = residualize_against_benchmarks(
        complete_panel,
        stocks=stocks,
    )
    fitted_returns = benchmark_result["fitted_returns"]
    residual_returns = benchmark_result["residual_returns"]

    raw_variance = complete_panel[stocks].var(ddof=1)
    benchmark_variance = fitted_returns.var(ddof=1)
    residual_variance = residual_returns.var(ddof=1)

    residual_pca = fit_pca(residual_returns, method="covariance")
    residual_covariance = residual_pca.matrix
    values = residual_pca.eigenvalues
    vectors = residual_pca.eigenvectors

    # vectors[i, j] is the eigenvector weight of stock i on residual PC j.
    # Therefore lambda_j * weight_ij^2 is PC j's contribution to stock i's
    # residual variance.
    component_variance = (vectors**2) * values[None, :]
    component_labels = [
        f"residual_PC{component + 1}_variance"
        for component in range(component_variance.shape[1])
    ]
    component_variance = pd.DataFrame(
        component_variance,
        index=stocks,
        columns=component_labels,
    )
    component_pct = component_variance.divide(
        raw_variance,
        axis="index",
    ) * 100
    component_pct.columns = [
        column.replace("_variance", "_pct")
        for column in component_pct.columns
    ]

    ledger = pd.DataFrame(
        {
            "raw_variance": raw_variance,
            "benchmark_variance": benchmark_variance,
            "residual_variance": residual_variance,
            "benchmark_pct": 100 * benchmark_variance / raw_variance,
            "residual_pct": 100 * residual_variance / raw_variance,
        }
    )
    ledger = ledger.join(component_pct)
    ledger["residual_PC2_to_PC3_pct"] = ledger[
        ["residual_PC2_pct", "residual_PC3_pct"]
    ].sum(axis=1)
    ledger["residual_PC4_to_PC12_pct"] = ledger[
        [f"residual_PC{component}_pct" for component in range(4, 13)]
    ].sum(axis=1)
    ledger["total_pct"] = ledger[GROUP_COLUMNS].sum(axis=1)
    ledger["variance_identity_error"] = (
        ledger["raw_variance"]
        - ledger["benchmark_variance"]
        - ledger["residual_variance"]
    ).abs()

    return {
        "ledger": ledger,
        "component_variance": component_variance,
        "component_pct": component_pct,
        "raw_variance": raw_variance,
        "benchmark_variance": benchmark_variance,
        "residual_variance": residual_variance,
        "residual_eigenvalues": values,
        "residual_covariance": residual_covariance,
    }


def build_global_summary(results):
    """Aggregate the decomposition using total variance as the denominator."""

    raw_total = results["raw_variance"].sum()
    benchmark_total = results["benchmark_variance"].sum()
    residual_total = results["residual_variance"].sum()
    eigenvalues = results["residual_eigenvalues"]

    components = {
        "Benchmark SPY/XLF": benchmark_total,
        "Residual PC1": eigenvalues[0],
        "Residuals PC2-PC3": eigenvalues[1:3].sum(),
        "Residuals PC4-PC12": eigenvalues[3:].sum(),
    }
    summary = pd.DataFrame(
        {
            "variance": components,
        }
    )
    summary["share_of_raw_pct"] = 100 * summary["variance"] / raw_total
    summary["share_of_residual_pct"] = np.nan
    summary.loc["Residual PC1":, "share_of_residual_pct"] = (
        100 * summary.loc["Residual PC1":, "variance"] / residual_total
    )
    summary.loc["Benchmark SPY/XLF", "share_of_residual_pct"] = 0.0
    return summary


def plot_stock_decomposition(ledger, complete_panel):
    """Save a 100% stacked bar chart for the twelve stocks."""

    figure, axis = plt.subplots(figsize=(11, 7))
    left = np.zeros(len(ledger))
    for column, label, color in zip(
        GROUP_COLUMNS,
        GROUP_LABELS,
        GROUP_COLORS,
    ):
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
        "Variance decomposition by stock",
        loc="left",
        color="#252525",
        pad=18,
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
    axis.grid(axis="x", color="#d9d9d9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(OUT_DIR / "10_variance_decomposition_by_stock.png", dpi=180)
    plt.close(figure)


def plot_global_decomposition(summary):
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
        "Aggregate variance decomposition",
        loc="left",
        color="#252525",
        pad=18,
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
    axis.grid(axis="x", color="#d9d9d9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        frameon=False,
    )
    figure.subplots_adjust(left=0.13, right=0.98, top=0.78, bottom=0.34)
    figure.savefig(OUT_DIR / "10_variance_decomposition_global.png", dpi=180)
    plt.close(figure)


def main():
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_returns = pd.read_parquet(RETURN_MATRIX_CLEAN_FILE)
    stocks = list(CORE_UNIVERSE)
    required = stocks + list(BENCHMARKS)
    missing = [symbol for symbol in required if symbol not in all_returns.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    panel = all_returns.loc[:, required]
    complete_panel = panel.dropna(how="any")
    if complete_panel.index.has_duplicates:
        raise ValueError("The benchmark panel contains duplicate index values.")
    if not complete_panel.index.is_monotonic_increasing:
        raise ValueError("The benchmark panel index is not sorted.")

    results = build_variance_ledger(complete_panel, stocks)
    ledger = results["ledger"]
    global_summary = build_global_summary(results)

    ledger.to_csv(OUT_DIR / "10_variance_ledger.csv")
    results["component_variance"].to_csv(
        OUT_DIR / "10_residual_pc_contributions_variance.csv"
    )
    results["component_pct"].to_csv(
        OUT_DIR / "10_residual_pc_contributions_pct_of_raw.csv"
    )
    global_summary.to_csv(OUT_DIR / "10_global_variance_decomposition.csv")
    results["residual_covariance"].to_csv(
        OUT_DIR / "10_residual_covariance_matrix.csv"
    )

    plot_stock_decomposition(ledger, complete_panel)
    plot_global_decomposition(global_summary)

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
    print(
        "Maximum error in residual component sum = residual variance:",
        f"{(results['component_variance'].sum(axis=1) - results['residual_variance']).abs().max():.3e}",
    )
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
