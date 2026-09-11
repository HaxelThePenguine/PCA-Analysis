"""Reusable benchmark projection and residualization utilities."""

from __future__ import annotations

from typing import Sequence, TypedDict

import numpy as np
import pandas as pd

from config import BENCHMARKS
from utils.data import require_columns


class ResidualizationResult(TypedDict):
    """Named outputs from a multivariate benchmark projection."""

    coefficients: pd.DataFrame
    fitted_returns: pd.DataFrame
    residual_returns: pd.DataFrame
    diagnostics: pd.DataFrame


def residualize_against_benchmarks(
    panel: pd.DataFrame,
    stocks: Sequence[str],
    benchmarks: Sequence[str] = BENCHMARKS,
) -> ResidualizationResult:
    """Project stock returns on benchmark returns and return residual data."""

    stock_names = list(stocks)
    benchmark_names = list(benchmarks)
    required = require_columns(
        panel,
        stock_names + benchmark_names,
        context="Benchmark panel",
    )
    if panel.loc[:, required].isna().any().any():
        raise ValueError("Benchmark residualization requires a complete panel.")

    benchmark_values = panel.loc[:, benchmark_names].to_numpy(dtype=float)
    stock_values = panel.loc[:, stock_names].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(panel)), benchmark_values])
    coefficients = np.linalg.lstsq(design, stock_values, rcond=None)[0]
    fitted_values = design @ coefficients
    residual_values = stock_values - fitted_values

    coefficient_columns = ["alpha"] + [f"beta_{name}" for name in benchmark_names]
    coefficient_frame = pd.DataFrame(
        coefficients.T,
        index=stock_names,
        columns=coefficient_columns,
    )
    fitted_returns = pd.DataFrame(
        fitted_values,
        index=panel.index,
        columns=stock_names,
    )
    residual_returns = pd.DataFrame(
        residual_values,
        index=panel.index,
        columns=stock_names,
    )

    raw_std = panel.loc[:, stock_names].std()
    residual_std = residual_returns.std()
    diagnostics = pd.DataFrame(
        {
            "raw_std": raw_std,
            "residual_std": residual_std,
            "residual_mean": residual_returns.mean(),
        }
    )
    for benchmark in benchmark_names:
        diagnostics[f"corr_resid_{benchmark}"] = residual_returns.corrwith(
            panel[benchmark]
        )
    diagnostics["r_squared"] = 1 - (residual_std / raw_std) ** 2
    diagnostics["variance_removed_pct"] = diagnostics["r_squared"] * 100

    return {
        "coefficients": coefficient_frame,
        "fitted_returns": fitted_returns,
        "residual_returns": residual_returns,
        "diagnostics": diagnostics,
    }


def build_factor_panels(
    panel: pd.DataFrame, stocks: Sequence[str]
) -> tuple[dict[str, pd.DataFrame], ResidualizationResult]:
    """Return raw/residual panels and the benchmark fit used to construct them."""
    result = residualize_against_benchmarks(panel, stocks=stocks)
    return {
        "raw": panel.loc[:, list(stocks)],
        "benchmark_residual": result["residual_returns"],
    }, result
