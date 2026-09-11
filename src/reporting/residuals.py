"""Charts and console output for residuals."""

from __future__ import annotations

from pathlib import Path

from utils.pca import format_pca_summary


def print_summary(
    panel,
    complete_panel,
    benchmark_result,
    residual_pca,
    covariance_diff,
    score_error,
    *,
    out_dir: Path,
) -> None:
    print("=== SPY/XLF RESIDUALIZATION ===")
    print(f"Initial panel: {panel.shape}")
    print(f"Complete panel: {complete_panel.shape}")
    print(f"Observations retained: {100 * len(complete_panel) / len(panel):.2f}%")
    maximum_residual_correlation = (
        benchmark_result["diagnostics"][["corr_resid_SPY", "corr_resid_XLF"]]
        .abs()
        .to_numpy()
        .max()
    )
    print(f"Maximum residual-SPY/XLF correlation: {maximum_residual_correlation:.3e}")
    print(f"Covariance check: {covariance_diff:.3e}")
    print(f"Score/eigenvalue check: {score_error:.3e}")
    print(f"Residual PC1: {residual_pca.explained[0] * 100:.4f}%")
    print(f"First 3 residual PCs: {residual_pca.explained[:3].sum() * 100:.4f}%")
    print("\n=== BETAS ===")
    print(benchmark_result["coefficients"].round(4).to_string())
    print("\n=== RESIDUAL DIAGNOSTICS ===")
    print(benchmark_result["diagnostics"].round(4).to_string())
    print("\n=== RESIDUAL PCA ===")
    print(format_pca_summary(residual_pca.summary))
    print("\nPC1-PC3 loadings:")
    print(residual_pca.weights.iloc[:, :3].round(4).to_string())
    print(f"\nOutputs saved to: {out_dir}")
