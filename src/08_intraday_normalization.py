"""Intraday-volatility normalization and PCA robustness check."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CORE_UNIVERSE, REPORTS_DIR, RETURN_CORE_FILE, ensure_project_directories
from data_utils import load_panel, normalize_intraday_volatility, save_csv_tables
from pca_utils import fit_pca, format_pca_summary
from plotting_utils import save_figure, style_axis


OUT_DIR = REPORTS_DIR / "pca_intraday"
BLUE = "#2F6B9A"
GRID = "#D9DEE5"


def save_profile_plot(profile: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(profile.index, profile["mean_volatility"], color=BLUE, linewidth=1.6)
    axis.set_title("Average intraday volatility of the CORE panel")
    axis.set_xlabel("Minutes from market open")
    axis.set_ylabel("Return standard deviation")
    style_axis(axis, grid_color=GRID)
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "08_intraday_volatility_profile.png")


def main() -> None:
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    returns = load_panel(
        RETURN_CORE_FILE,
        CORE_UNIVERSE,
        context="CORE panel",
    )
    normalization = normalize_intraday_volatility(returns)
    normalized_returns = normalization.normalized_returns
    volatility_profile = normalization.volatility_by_symbol
    observations = normalization.observations_by_minute
    profile = pd.DataFrame(
        {
            "mean_volatility": volatility_profile.mean(axis=1),
            "min_volatility": volatility_profile.min(axis=1),
            "max_volatility": volatility_profile.max(axis=1),
            "observations": observations,
        }
    )
    profile.index.name = "minute_from_open"

    normalized_profile = normalized_returns.groupby(
        normalized_returns.index.hour * 60
        + normalized_returns.index.minute
        - (9 * 60 + 30)
    ).std(ddof=1)
    normalization_error = np.max(np.abs(normalized_profile.to_numpy() - 1))

    centered = normalized_returns.subtract(
        normalized_returns.mean(),
        axis="columns",
    )
    pca_result = fit_pca(normalized_returns, method="covariance")
    covariance = pca_result.matrix
    covariance_diff = (covariance - centered.cov()).abs().to_numpy().max()
    values = pca_result.eigenvalues
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
    save_csv_tables(tables, OUT_DIR)
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
