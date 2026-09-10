"""Reusable benchmark projection and residualization utilities."""

import numpy as np
import pandas as pd

from config import BENCHMARKS


def residualize_against_benchmarks(
    panel,
    stocks,
    benchmarks=BENCHMARKS,
):
    """Project stock returns on benchmark returns and return residual data."""

    stock_names = list(stocks)
    benchmark_names = list(benchmarks)
    required = stock_names + benchmark_names
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
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
