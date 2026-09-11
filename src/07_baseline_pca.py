"""Baseline covariance/correlation PCA on the CORE return panel."""

from __future__ import annotations

import pandas as pd

from config import (
    CORE_UNIVERSE,
    REPORTS_DIR,
    RETURN_CORE_FILE,
    ensure_project_directories,
)
from reporting.baseline import make_plots, print_summary
from utils.data import load_panel, save_csv_tables
from utils.pca import fit_pca, pca_diagnostics, reconstruction_by_stock

OUT_DIR = REPORTS_DIR / "pca_baseline"


def main() -> None:
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    returns = load_panel(RETURN_CORE_FILE, CORE_UNIVERSE, context="CORE panel")
    covariance_result = fit_pca(returns, method="covariance")
    correlation_result = fit_pca(returns, method="correlation")
    covariance_diff, score_error = pca_diagnostics(covariance_result)
    correlation_diff, _ = pca_diagnostics(correlation_result)
    ticker_summary = reconstruction_by_stock(covariance_result)
    tables = {
        "07_return_stats.csv": pd.DataFrame(
            {"mean": returns.mean(), "std": returns.std(ddof=1)}
        ),
        "07_covariance_matrix.csv": covariance_result.matrix,
        "07_correlation_matrix.csv": correlation_result.matrix,
        "07_covariance_summary.csv": covariance_result.summary,
        "07_correlation_summary.csv": correlation_result.summary,
        "07_covariance_loadings_pc1_pc3.csv": covariance_result.weights.iloc[:, :3],
        "07_correlation_loadings_pc1_pc3.csv": correlation_result.weights.iloc[:, :3],
        "07_covariance_technical_loadings_pc1_pc3.csv": covariance_result.loadings.iloc[
            :, :3
        ],
        "07_correlation_technical_loadings_pc1_pc3.csv": correlation_result.loadings.iloc[
            :, :3
        ],
        "07_ticker_reconstruction.csv": ticker_summary,
    }
    save_csv_tables(tables, OUT_DIR)
    make_plots(covariance_result, correlation_result, out_dir=OUT_DIR)
    print_summary(
        returns,
        covariance_result,
        correlation_result,
        ticker_summary,
        covariance_diff,
        correlation_diff,
        score_error,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
