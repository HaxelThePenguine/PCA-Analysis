"""Charts and console output for intraday."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils.pca import format_pca_summary
from utils.plotting import save_figure, style_axis

BLUE = "#2F6B9A"
GRID = "#D9DEE5"


def save_profile_plot(profile: pd.DataFrame, *, out_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(profile.index, profile["mean_volatility"], color=BLUE, linewidth=1.6)
    axis.set_title("Average intraday volatility of the CORE panel")
    axis.set_xlabel("Minutes from market open")
    axis.set_ylabel("Return standard deviation")
    style_axis(axis, grid_color=GRID)
    fig.tight_layout()
    save_figure(fig, out_dir / "08_intraday_volatility_profile.png")


def print_summary(
    returns,
    normalization,
    pca_result,
    profile,
    normalization_error,
    covariance_diff,
    score_error,
    *,
    out_dir: Path,
) -> None:
    print("=== INTRADAY NORMALIZATION ===")
    print(f"Panel: {returns.shape} | NaN: 0")
    print(f"Profile: minute {profile.index.min()} -> {profile.index.max()}")
    print(
        f"Observations per minute: {normalization.observations_by_minute.min()} -> {normalization.observations_by_minute.max()}"
    )
    print(f"Normalized volatility check: {normalization_error:.3e}")
    print(f"Covariance check: {covariance_diff:.3e}")
    print(f"Score/eigenvalue check: {score_error:.3e}")
    print(f"Eigenvalue sum: {pca_result.eigenvalues.sum():.8f}")
    print(f"First 3 components: {pca_result.explained[:3].sum() * 100:.4f}% explained")
    print("\n=== INTRADAY PROFILE, START/END ===")
    print(pd.concat([profile.head(3), profile.tail(3)]).round(6).to_string())
    print("\n=== NORMALIZED PCA ===")
    print(format_pca_summary(pca_result.summary))
    print("\nPC1-PC3 loadings:")
    print(pca_result.weights.iloc[:, :3].round(4).to_string())
    print(f"\nOutputs saved to: {out_dir}")
