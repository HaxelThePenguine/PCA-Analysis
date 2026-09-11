"""Reusable routines for variance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from utils.benchmark import residualize_against_benchmarks
from utils.pca import fit_pca

GROUP_COLUMNS = [
    "benchmark_pct",
    "residual_PC1_pct",
    "residual_PC2_to_PC3_pct",
    "residual_PC4_to_PC12_pct",
]


def build_variance_ledger(
    complete_panel: pd.DataFrame,
    stocks: Sequence[str],
) -> dict[str, Any]:
    """Fit benchmarks and split each stock's variance into additive parts."""

    benchmark_result = residualize_against_benchmarks(
        complete_panel,
        stocks=stocks,
    )
    fitted_returns = benchmark_result["fitted_returns"]
    residual_returns = benchmark_result["residual_returns"]

    stock_columns = list(stocks)
    raw_variance = complete_panel[stock_columns].var(ddof=1)
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
        index=stock_columns,
        columns=component_labels,
    )
    component_pct = (
        component_variance.divide(
            raw_variance,
            axis="index",
        )
        * 100
    )
    component_pct.columns = [
        column.replace("_variance", "_pct") for column in component_pct.columns
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


def build_global_summary(results: dict[str, Any]) -> pd.DataFrame:
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
