"""Intraday-volatility normalization and PCA robustness check."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CORE_UNIVERSE, REPORTS_DIR, RETURN_CORE_FILE, ensure_project_directories
from pca_utils import fit_pca, format_pca_summary


OUT_DIR = REPORTS_DIR / "pca_intraday"
BLUE = "#2F6B9A"
GRID = "#D9DEE5"


def save_profile_plot(profile):
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(profile.index, profile["mean_volatility"], color=BLUE, linewidth=1.6)
    axis.set_title("Average intraday volatility of the CORE panel")
    axis.set_xlabel("Minutes from market open")
    axis.set_ylabel("Return standard deviation")
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_intraday_volatility_profile.png", dpi=180)
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

    minute_from_open = pd.Series(
        returns.index.hour * 60 + returns.index.minute - (9 * 60 + 30),
        index=returns.index,
        name="minute_from_open",
    )
    volatility_profile = returns.groupby(minute_from_open).std(ddof=1)
    observations = minute_from_open.value_counts().sort_index()
    profile = pd.DataFrame(
        {
            "mean_volatility": volatility_profile.mean(axis=1),
            "min_volatility": volatility_profile.min(axis=1),
            "max_volatility": volatility_profile.max(axis=1),
            "observations": observations,
        }
    )
    profile.index.name = "minute_from_open"

    # Each row is divided by the volatility of that ticker at that minute.
    scale = volatility_profile.loc[minute_from_open.to_numpy()].copy()
    scale.index = returns.index
    if scale.isna().any().any() or (scale <= 0).any().any():
        raise ValueError("Invalid volatility profile.")

    normalized_returns = returns.divide(scale)
    if not np.isfinite(normalized_returns.to_numpy()).all():
        raise ValueError("Normalized returns contain non-finite values.")

    normalized_profile = normalized_returns.groupby(minute_from_open).std(ddof=1)
    normalization_error = np.max(np.abs(normalized_profile.to_numpy() - 1))

    centered = normalized_returns.subtract(
        normalized_returns.mean(),
        axis="columns",
    )
    pca_result = fit_pca(normalized_returns, method="covariance")
    covariance = pca_result.matrix
    covariance_diff = (covariance - centered.cov()).abs().to_numpy().max()
    values = pca_result.eigenvalues
    vectors = pca_result.eigenvectors
    explained = pca_result.explained
    summary = pca_result.summary
    loadings = pca_result.weights.iloc[:, :3]

    scores = pca_result.scores.to_numpy()
    score_error = np.max(
        np.abs(values - pd.DataFrame(scores).var(ddof=1).to_numpy())
    )

    tables = {
        "08_intraday_volatility_profile.csv": profile,
        "08_normalized_return_stats.csv": pd.DataFrame(
            {
                "mean": normalized_returns.mean(),
                "std": normalized_returns.std(ddof=1),
            }
        ),
        "08_normalized_covariance_matrix.csv": covariance,
        "08_normalized_pca_summary.csv": summary,
        "08_normalized_pca_loadings_pc1_pc3.csv": loadings,
        "08_normalized_technical_loadings_pc1_pc3.csv": pca_result.loadings.iloc[:, :3],
    }
    for filename, table in tables.items():
        table.to_csv(OUT_DIR / filename)
    save_profile_plot(profile)

    print("=== INTRADAY NORMALIZATION ===")
    print(f"Panel: {returns.shape} | NaN: 0")
    print(f"Profile: minute {profile.index.min()} -> {profile.index.max()}")
    print(f"Observations per minute: {observations.min()} -> {observations.max()}")
    print(f"Normalized volatility check: {normalization_error:.3e}")
    print(f"Covariance check: {covariance_diff:.3e}")
    print(f"Score/eigenvalue check: {score_error:.3e}")
    print(f"Eigenvalue sum: {values.sum():.8f}")
    print(f"First 3 components: {explained[:3].sum() * 100:.4f}% explained")

    print("\n=== INTRADAY PROFILE, START/END ===")
    print(pd.concat([profile.head(3), profile.tail(3)]).round(6).to_string())
    print("\n=== NORMALIZED PCA ===")
    print(format_pca_summary(summary))
    print("\nPC1-PC3 loadings:")
    print(loadings.round(4).to_string())
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
