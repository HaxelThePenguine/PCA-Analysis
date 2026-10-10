"""Intraday-volatility normalization and PCA robustness check."""

from __future__ import annotations

import pandas as pd

from config import (
    CORE_UNIVERSE,
    REPORTS_DIR,
    RETURN_CORE_FILE,
    ensure_project_directories,
)
from reporting.intraday import print_summary, save_profile_plot
from utils.data import load_panel, normalize_intraday_volatility, save_csv_tables
from utils.pca import fit_pca, pca_diagnostics

OUT_DIR = REPORTS_DIR / "pca_intraday"


def main() -> None:
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    returns = load_panel(RETURN_CORE_FILE, CORE_UNIVERSE, context="CORE panel")
    normalization = normalize_intraday_volatility(returns)
    normalized_returns = normalization.normalized_returns
    profile = normalization.summary()
    normalization_error = normalization.unit_volatility_error()
    pca_result = fit_pca(normalized_returns, method="covariance")
    covariance_diff, score_error = pca_diagnostics(pca_result)
    tables = {
        "08_intraday_volatility_profile.csv": profile,
        "08_normalized_return_stats.csv": pd.DataFrame(
            {"mean": normalized_returns.mean(), "std": normalized_returns.std(ddof=1)}
        ),
        "08_normalized_covariance_matrix.csv": pca_result.matrix,
        "08_normalized_pca_summary.csv": pca_result.summary,
        "08_normalized_pca_loadings_pc1_pc3.csv": pca_result.weights.iloc[:, :3],
        "08_normalized_technical_loadings_pc1_pc3.csv": pca_result.loadings.iloc[:, :3],
    }
    save_csv_tables(tables, OUT_DIR)
    save_profile_plot(profile, out_dir=OUT_DIR)
    print_summary(
        returns,
        normalization,
        pca_result,
        profile,
        normalization_error,
        covariance_diff,
        score_error,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
