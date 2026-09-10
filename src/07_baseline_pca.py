"""Baseline covariance/correlation PCA on the CORE return panel."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CORE_UNIVERSE,
    REPORTS_DIR,
    RETURN_CORE_FILE,
    ensure_project_directories,
)


OUT_DIR = REPORTS_DIR / "pca_baseline"
COLORS = {"cov": "#2F6B9A", "corr": "#C4933F", "grid": "#D9DEE5"}


def pca(matrix):
    """Return sorted eigenvalues, eigenvectors, variance shares and tables."""

    values, vectors = np.linalg.eigh(matrix.to_numpy())
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    # Eigenvector signs are arbitrary; orient the largest coefficient positively.
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


def style_axis(axis, grid_axis="y"):
    axis.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_line_plot(filename, title, ylabel, series, cumulative=False):
    components = np.arange(1, len(series[0][1]) + 1)
    fig, axis = plt.subplots(figsize=(8, 5))

    for label, values, color in series:
        y = values.cumsum() * 100 if cumulative else values * 100
        axis.plot(components, y, marker="o", color=color, label=label)

    if cumulative:
        axis.axhline(
            80,
            color="#7B8794",
            linestyle="--",
            linewidth=0.9,
            label="Soglia 80%",
        )
        axis.set_ylim(0, 105)

    axis.set_title(title)
    axis.set_xlabel(
        "Numero di componenti" if cumulative else "Componente principale"
    )
    axis.set_ylabel(ylabel)
    axis.set_xticks(components)
    axis.legend(frameon=False)
    style_axis(axis)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pc1_plot(cov_loadings, corr_loadings):
    order = corr_loadings["PC1"].sort_values().index
    positions = np.arange(len(order))
    width = 0.38

    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(
        positions - width / 2,
        cov_loadings.loc[order, "PC1"],
        height=width,
        color=COLORS["cov"],
        label="Covariance",
    )
    axis.barh(
        positions + width / 2,
        corr_loadings.loc[order, "PC1"],
        height=width,
        color=COLORS["corr"],
        label="Correlation",
    )
    axis.axvline(0, color="#1F2933", linewidth=0.8)
    axis.set_title("Baseline PCA: PC1 loadings")
    axis.set_xlabel("Eigenvector loading")
    axis.set_yticks(positions)
    axis.set_yticklabels(order)
    axis.legend(frameon=False)
    style_axis(axis, grid_axis="x")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_pc1_loadings.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    returns = pd.read_parquet(RETURN_CORE_FILE)
    missing = [symbol for symbol in CORE_UNIVERSE if symbol not in returns.columns]
    if missing:
        raise ValueError(f"Incomplete CORE panel; missing symbols: {missing}")
    returns = returns.loc[:, list(CORE_UNIVERSE)]

    if returns.isna().any().any():
        raise ValueError("The CORE panel contains NaN values.")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("The CORE panel index is invalid.")

    means = returns.mean()
    stds = returns.std(ddof=1)
    X = returns.subtract(means, axis="columns")

    covariance = (X.T @ X) / (len(X) - 1)
    covariance_diff = (covariance - returns.cov()).abs().to_numpy().max()
    cov_values, cov_vectors, cov_explained, cov_summary, cov_loadings = pca(
        covariance
    )

    scores = X.to_numpy() @ cov_vectors
    score_error = np.max(
        np.abs(cov_values - pd.DataFrame(scores).var(ddof=1).to_numpy())
    )
    reconstruction = scores[:, :3] @ cov_vectors[:, :3].T
    residuals = X.to_numpy() - reconstruction
    reconstruction_pct = 100 * (
        1 - np.sum(residuals**2) / np.sum(X.to_numpy() ** 2)
    )
    residual_df = pd.DataFrame(residuals, index=X.index, columns=X.columns)
    ticker_summary = pd.DataFrame(
        {
            "original_variance": X.var(ddof=1),
            "residual_variance": residual_df.var(ddof=1),
        }
    )
    ticker_summary["explained_pct"] = 100 * (
        1
        - ticker_summary["residual_variance"]
        / ticker_summary["original_variance"]
    )

    Z = X.divide(stds, axis="columns")
    correlation = Z.cov()
    correlation_diff = (correlation - returns.corr()).abs().to_numpy().max()
    corr_values, corr_vectors, corr_explained, corr_summary, corr_loadings = pca(
        correlation
    )

    tables = {
        "07_return_stats.csv": pd.DataFrame({"mean": means, "std": stds}),
        "07_covariance_matrix.csv": covariance,
        "07_correlation_matrix.csv": correlation,
        "07_covariance_summary.csv": cov_summary,
        "07_correlation_summary.csv": corr_summary,
        "07_covariance_loadings_pc1_pc3.csv": cov_loadings,
        "07_correlation_loadings_pc1_pc3.csv": corr_loadings,
        "07_ticker_reconstruction.csv": ticker_summary,
    }
    for filename, table in tables.items():
        table.to_csv(OUT_DIR / filename)

    series = [
        ("Covariance", cov_explained, COLORS["cov"]),
        ("Correlation", corr_explained, COLORS["corr"]),
    ]
    save_line_plot(
        "07_scree_variance.png",
        "Baseline PCA: explained variance by component",
        "Explained variance (%)",
        series,
    )
    save_line_plot(
        "07_cumulative_variance.png",
        "Baseline PCA: cumulative explained variance",
        "Cumulative explained variance (%)",
        series,
        cumulative=True,
    )
    save_pc1_plot(cov_loadings, corr_loadings)

    print("=== BASELINE PCA CORE ===")
    print(f"Panel: {returns.shape} | NaN: 0")
    print(f"Date: {returns.index.min()} -> {returns.index.max()}")
    print(f"Covariance check: {covariance_diff:.3e}")
    print(f"Score/eigenvalue check: {score_error:.3e}")
    print(f"First 3 components: {reconstruction_pct:.4f}% explained")
    print(f"Correlation check: {correlation_diff:.3e}")
    print(f"Correlation eigenvalue sum: {corr_values.sum():.12f}")

    print("\n=== COVARIANCE PCA ===")
    print(format_summary(cov_summary))
    print("\nPC1-PC3 loadings:")
    print(cov_loadings.round(4).to_string())

    print("\n=== CORRELATION PCA ===")
    print(format_summary(corr_summary))
    print("\nPC1-PC3 loadings:")
    print(corr_loadings.round(4).to_string())

    print("\n=== EXPLAINED VARIANCE BY STOCK, FIRST 3 PCS ===")
    print(ticker_summary[["explained_pct"]].round(2).to_string())
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
