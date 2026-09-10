"""SPY/XLF residualization and PCA of the remaining CORE structure."""

import numpy as np
import pandas as pd

from config import (
    CORE_UNIVERSE,
    REPORTS_DIR,
    RETURN_MATRIX_CLEAN_FILE,
    ensure_project_directories,
)


OUT_DIR = REPORTS_DIR / "pca_residuals"
BENCHMARKS = ["SPY", "XLF"]


def run_pca(matrix):
    values, vectors = np.linalg.eigh(matrix.to_numpy())
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    # The sign is arbitrary; orient the largest coefficient positively.
    for component in range(vectors.shape[1]):
        pivot = np.argmax(np.abs(vectors[:, component]))
        if vectors[pivot, component] < 0:
            vectors[:, component] *= -1

    explained = values / values.sum()
    labels = [f"PC{i}" for i in range(1, len(values) + 1)]
    summary = pd.DataFrame(
        {
            "eigenvalue": values,
            "explained_pct": explained * 100,
            "cumulative_pct": explained.cumsum() * 100,
        },
        index=labels,
    )
    loadings = pd.DataFrame(
        vectors[:, :3],
        index=matrix.columns,
        columns=["PC1", "PC2", "PC3"],
    )
    return values, vectors, explained, summary, loadings


def format_summary(summary):
    table = summary.copy()
    table["eigenvalue"] = table["eigenvalue"].map(lambda x: f"{x:.8e}")
    for column in ("explained_pct", "cumulative_pct"):
        table[column] = table[column].map(lambda x: f"{x:.4f}")
    return table.to_string()


def main():
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_returns = pd.read_parquet(RETURN_MATRIX_CLEAN_FILE)
    stocks = list(CORE_UNIVERSE)
    required = stocks + BENCHMARKS
    missing = [symbol for symbol in required if symbol not in all_returns.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    panel = all_returns.loc[:, required]
    complete_panel = panel.dropna(how="any")
    if complete_panel.index.has_duplicates:
        raise ValueError("The benchmark panel contains duplicate index values.")
    if not complete_panel.index.is_monotonic_increasing:
        raise ValueError("The benchmark panel index is not sorted.")

    benchmark_returns = complete_panel[BENCHMARKS].to_numpy()
    design = np.column_stack(
        [np.ones(len(benchmark_returns)), benchmark_returns]
    )
    stock_returns = complete_panel[stocks].to_numpy()

    # One OLS fit per stock, written as a single least-squares system.
    coefficients = np.linalg.lstsq(
        design,
        stock_returns,
        rcond=None,
    )[0]
    betas = pd.DataFrame(
        coefficients.T,
        index=stocks,
        columns=["alpha", "beta_SPY", "beta_XLF"],
    )

    fitted_returns = design @ coefficients
    residual_returns = pd.DataFrame(
        stock_returns - fitted_returns,
        index=complete_panel.index,
        columns=stocks,
    )
    raw_std = complete_panel[stocks].std()
    residual_std = residual_returns.std()
    diagnostics = pd.DataFrame(
        {
            "raw_std": raw_std,
            "residual_std": residual_std,
            "residual_mean": residual_returns.mean(),
            "corr_resid_SPY": residual_returns.corrwith(
                complete_panel["SPY"]
            ),
            "corr_resid_XLF": residual_returns.corrwith(
                complete_panel["XLF"]
            ),
        }
    )
    diagnostics["r_squared"] = 1 - (residual_std / raw_std) ** 2
    diagnostics["variance_removed_pct"] = diagnostics["r_squared"] * 100

    centered_residuals = residual_returns.subtract(
        residual_returns.mean(),
        axis="columns",
    )
    residual_covariance = (
        centered_residuals.T @ centered_residuals
    ) / (len(centered_residuals) - 1)
    covariance_diff = (
        residual_covariance - centered_residuals.cov()
    ).abs().to_numpy().max()
    values, vectors, explained, summary, loadings = run_pca(
        residual_covariance
    )

    scores = centered_residuals.to_numpy() @ vectors
    score_error = np.max(
        np.abs(
            values
            - pd.DataFrame(scores).var(ddof=1).to_numpy()
        )
    )

    tables = {
        "09_betas_spy_xlf.csv": betas,
        "09_residual_diagnostics.csv": diagnostics,
        "09_benchmark_correlation.csv": complete_panel[BENCHMARKS].corr(),
        "09_residual_covariance_matrix.csv": residual_covariance,
        "09_residual_pca_summary.csv": summary,
        "09_residual_pca_loadings_pc1_pc3.csv": loadings,
    }
    for filename, table in tables.items():
        table.to_csv(OUT_DIR / filename)

    print("=== SPY/XLF RESIDUALIZATION ===")
    print(f"Initial panel: {panel.shape}")
    print(f"Complete panel: {complete_panel.shape}")
    print(f"Observations retained: {100 * len(complete_panel) / len(panel):.2f}%")
    print(f"Maximum residual-SPY/XLF correlation: {diagnostics[['corr_resid_SPY', 'corr_resid_XLF']].abs().to_numpy().max():.3e}")
    print(f"Covariance check: {covariance_diff:.3e}")
    print(f"Score/eigenvalue check: {score_error:.3e}")
    print(f"Residual PC1: {explained[0] * 100:.4f}%")
    print(f"First 3 residual PCs: {explained[:3].sum() * 100:.4f}%")

    print("\n=== BETAS ===")
    print(betas.round(4).to_string())
    print("\n=== RESIDUAL DIAGNOSTICS ===")
    print(diagnostics.round(4).to_string())
    print("\n=== RESIDUAL PCA ===")
    print(format_summary(summary))
    print("\nPC1-PC3 loadings:")
    print(loadings.round(4).to_string())
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
