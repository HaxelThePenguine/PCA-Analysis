"""SPY/XLF residualization and PCA of the remaining CORE structure."""

from __future__ import annotations

from config import (
    BENCHMARKS,
    CORE_UNIVERSE,
    REPORTS_DIR,
    RETURN_MATRIX_CLEAN_FILE,
    ensure_project_directories,
)
from reporting.residuals import print_summary
from utils.benchmark import residualize_against_benchmarks
from utils.data import load_panel, save_csv_tables
from utils.pca import fit_pca, pca_diagnostics

OUT_DIR = REPORTS_DIR / "pca_residuals"


def main() -> None:
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stocks = list(CORE_UNIVERSE)
    required = stocks + list(BENCHMARKS)
    panel = load_panel(
        RETURN_MATRIX_CLEAN_FILE,
        required,
        context="Benchmark panel",
        require_complete=False,
    )
    complete_panel = panel.dropna(how="any")
    benchmark_result = residualize_against_benchmarks(complete_panel, stocks=stocks)
    betas = benchmark_result["coefficients"]
    residual_returns = benchmark_result["residual_returns"]
    diagnostics = benchmark_result["diagnostics"]
    residual_pca = fit_pca(residual_returns, method="covariance")
    covariance_diff, score_error = pca_diagnostics(residual_pca)
    tables = {
        "09_betas_spy_xlf.csv": betas,
        "09_residual_diagnostics.csv": diagnostics,
        "09_benchmark_correlation.csv": complete_panel.loc[:, list(BENCHMARKS)].corr(),
        "09_residual_covariance_matrix.csv": residual_pca.matrix,
        "09_residual_pca_summary.csv": residual_pca.summary,
        "09_residual_pca_loadings_pc1_pc3.csv": residual_pca.weights.iloc[:, :3],
        "09_residual_technical_loadings_pc1_pc3.csv": residual_pca.loadings.iloc[:, :3],
    }
    save_csv_tables(tables, OUT_DIR)
    print_summary(
        panel,
        complete_panel,
        benchmark_result,
        residual_pca,
        covariance_diff,
        score_error,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
