"""SPY/XLF residualization and PCA of the remaining CORE structure."""

from __future__ import annotations

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
from data_utils import load_panel, save_csv_tables
from pca_utils import fit_pca, format_pca_summary


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

    benchmark_result = residualize_against_benchmarks(
        complete_panel,
        stocks=stocks,
    )
    betas = benchmark_result["coefficients"]
    residual_returns = benchmark_result["residual_returns"]
    diagnostics = benchmark_result["diagnostics"]

    residual_pca = fit_pca(residual_returns, method="covariance")
    residual_covariance = residual_pca.matrix
    covariance_diff = (
        residual_covariance - residual_returns.subtract(
            residual_returns.mean(),
            axis="columns",
        ).cov()
    ).abs().to_numpy().max()
    values = residual_pca.eigenvalues
    explained = residual_pca.explained
    summary = residual_pca.summary
    loadings = residual_pca.weights.iloc[:, :3]

    scores = residual_pca.scores.to_numpy()
    score_error = np.max(
        np.abs(
            values
            - pd.DataFrame(scores).var(ddof=1).to_numpy()
        )
    )

    tables = {
        "09_betas_spy_xlf.csv": betas,
        "09_residual_diagnostics.csv": diagnostics,
        "09_benchmark_correlation.csv": complete_panel.loc[:, list(BENCHMARKS)].corr(),
        "09_residual_covariance_matrix.csv": residual_covariance,
        "09_residual_pca_summary.csv": summary,
        "09_residual_pca_loadings_pc1_pc3.csv": loadings,
        "09_residual_technical_loadings_pc1_pc3.csv": residual_pca.loadings.iloc[:, :3],
    }
    save_csv_tables(tables, OUT_DIR)

    print("=== SPY/XLF RESIDUALIZATION ===")
    print(f"Initial panel: {panel.shape}")
    print(f"Complete panel: {complete_panel.shape}")
    print(f"Observations retained: {100 * len(complete_panel) / len(panel):.2f}%")
    maximum_residual_correlation = (
        diagnostics[["corr_resid_SPY", "corr_resid_XLF"]]
        .abs()
        .to_numpy()
        .max()
    )
    print(
        "Maximum residual-SPY/XLF correlation: "
        f"{maximum_residual_correlation:.3e}"
    )
    print(f"Covariance check: {covariance_diff:.3e}")
    print(f"Score/eigenvalue check: {score_error:.3e}")
    print(f"Residual PC1: {explained[0] * 100:.4f}%")
    print(f"First 3 residual PCs: {explained[:3].sum() * 100:.4f}%")

    print("\n=== BETAS ===")
    print(betas.round(4).to_string())
    print("\n=== RESIDUAL DIAGNOSTICS ===")
    print(diagnostics.round(4).to_string())
    print("\n=== RESIDUAL PCA ===")
    print(format_pca_summary(summary))
    print("\nPC1-PC3 loadings:")
    print(loadings.round(4).to_string())
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
