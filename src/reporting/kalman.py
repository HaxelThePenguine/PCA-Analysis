"""Charts and console output for kalman."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plotting import save_figure, style_axis


def plot_loadings(loadings: pd.DataFrame, *, out_dir: Path) -> None:
    pivot = loadings.pivot_table(
        index="window_end", columns=["factor", "stock"], values="loading"
    ).sort_index()
    dates = pd.to_datetime(pivot.index)
    figure, axes = plt.subplots(
        1, loadings["factor"].nunique(), figsize=(13, 4), sharex=True
    )
    for number, axis in enumerate(np.atleast_1d(axes), start=1):
        factor = f"LF{number}"
        columns = [c for c in pivot.columns if c[0] == factor]
        top = pivot[columns].abs().median().nlargest(4).index
        for _, stock in top:
            axis.plot(dates, pivot[factor, stock], label=stock, linewidth=1.2)
        axis.set_title(factor)
        axis.set_ylabel("Loading")
        axis.legend(frameon=False, fontsize=8, ncol=2)
        style_axis(axis, format_dates=True)
    figure.suptitle("13-bis: dynamic L1 loadings", x=0.06, ha="left")
    figure.tight_layout()
    save_figure(figure, out_dir / "13_bis_loading_evolution.png", tight_bbox=True)


def print_summary(
    candidates, panel, stress, summary, windows, *, out_dir: Path
) -> None:
    dynamic = summary.iloc[2]
    recurring = candidates[candidates["recurring_candidate"]]
    answer = f"Kalman dynamic factors: median filtered holdout residual variance {dynamic['holdout_residual_variance_median']:.6g}, one-step {dynamic['one_step_residual_variance_median']:.6g}; loading cosine {dynamic['cosine_median']:.3f}, support Jaccard {dynamic['support_jaccard_median']:.3f}. Recurring candidates: {len(recurring)}; group labels remain data-dependent."
    (out_dir / "13_bis_summary.txt").write_text(answer + "\n", encoding="utf-8")
    print("=== 13-BIS KALMAN DYNAMIC FACTORS + L1 ===")
    print(f"Session residual panel: {panel.shape}; windows: {len(windows)}")
    print(summary.round(5).to_string(index=False))
    print("\nRecurring candidates:")
    print(recurring.to_string(index=False) if len(recurring) else "none above rule")
    print("\nRegional stress:")
    print(stress.round(4).to_string(index=False))
    print(f"\n{answer}\nOutputs saved to: {out_dir}")
