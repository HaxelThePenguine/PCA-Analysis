"""Charts and console output for regimes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm

from utils.dynamic_local_factor_regimes import (
    FACTORS,
    PRIMARY_STARTS,
    SENSITIVITY_STARTS,
    STEP_SESSIONS,
    STOCKS,
    WINDOWS,
)
from utils.plotting import save_figure, style_axis

CRISIS_SHADE_START = pd.Timestamp("2023-03-01")
CRISIS_SHADE_END = pd.Timestamp("2023-06-01")
COLORS = {"LF1": "#2F6690", "LF2": "#D99A2B", "LF3": "#C26A2E"}


def _shade_crisis(axis: Axes, dates: pd.Series | pd.DatetimeIndex) -> None:
    """Highlight March-May 2023 without assigning a causal interpretation."""
    parsed = pd.to_datetime(dates)
    if len(parsed) == 0:
        return
    minimum = parsed.min()
    maximum = parsed.max()
    if maximum < CRISIS_SHADE_START or minimum > CRISIS_SHADE_END:
        return
    axis.axvspan(
        CRISIS_SHADE_START,
        CRISIS_SHADE_END,
        color="#E07A5F",
        alpha=0.12,
        linewidth=0,
        label="Mar–May 2023",
    )


def _shade_index_crisis(axis: Axes, dates: pd.DatetimeIndex) -> None:
    """Highlight the crisis interval on an image plot with integer x positions."""
    if len(dates) == 0:
        return
    mask = (dates >= CRISIS_SHADE_START) & (dates <= CRISIS_SHADE_END)
    if mask.any():
        positions = np.flatnonzero(mask)
        axis.axvspan(
            max(-0.5, positions[0] - 0.5),
            positions[-1] + 0.5,
            color="#E07A5F",
            alpha=0.12,
            linewidth=0,
        )


def _format_image_dates(axis: Axes, dates: pd.DatetimeIndex) -> None:
    """Use a small, readable set of dates on rolling heatmaps."""
    if len(dates) == 0:
        return
    count = min(7, len(dates))
    positions = np.linspace(0, len(dates) - 1, count, dtype=int)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [dates[position].strftime("%Y-%m-%d") for position in positions],
        rotation=30,
        ha="right",
    )


def plot_loading_heatmaps(loadings: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot full-sample-aligned structural loadings through time."""
    data = loadings[loadings["alignment_method"] == "ex_post_full_sample_alignment"]
    maximum = float(np.nanmax(np.abs(data["structural_loading"])))
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    figure, axes = plt.subplots(
        len(WINDOWS), len(FACTORS), figsize=(18, 10), squeeze=False, sharey=True
    )
    image = None
    for row, window_size in enumerate(WINDOWS):
        for column, factor in enumerate(FACTORS):
            axis = axes[row, column]
            subset = data[
                (data["window_sessions"] == window_size) & (data["factor"] == factor)
            ]
            pivot = subset.pivot(
                index="stock", columns="window_end", values="structural_loading"
            ).reindex(STOCKS)
            dates = pd.DatetimeIndex(pd.to_datetime(pivot.columns))
            image = axis.imshow(
                pivot.to_numpy(dtype=float),
                cmap="RdBu_r",
                norm=norm,
                aspect="auto",
                interpolation="nearest",
            )
            _shade_index_crisis(axis, dates)
            _format_image_dates(axis, dates)
            axis.set_title(f"{window_size}-session: {factor}")
            if column == 0:
                axis.set_yticks(range(len(STOCKS)))
                axis.set_yticklabels(STOCKS)
            else:
                axis.set_yticks(range(len(STOCKS)))
                axis.set_yticklabels([])
            axis.tick_params(axis="both", length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
    colorbar_axis = figure.add_axes([0.925, 0.16, 0.015, 0.68])
    figure.colorbar(image, cax=colorbar_axis, label="Structural loading")
    figure.suptitle(
        "Rolling L1-rotation structural loadings",
        x=0.06,
        ha="left",
        y=0.98,
        fontsize=14,
    )
    figure.text(
        0.06,
        0.945,
        "Ex-post full-sample alignment is descriptive only; the shaded interval is March–May 2023.",
        color="#666666",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.05, right=0.9, bottom=0.1, top=0.86, wspace=0.14, hspace=0.28
    )
    save_figure(figure, out_dir / "14_rolling_l1_loading_heatmaps.png", tight_bbox=True)


def plot_support_heatmaps(support: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot full-sample-aligned active support through time."""
    data = support[support["alignment_method"] == "ex_post_full_sample_alignment"]
    figure, axes = plt.subplots(
        len(WINDOWS), len(FACTORS), figsize=(18, 10), squeeze=False, sharey=True
    )
    image = None
    for row, window_size in enumerate(WINDOWS):
        for column, factor in enumerate(FACTORS):
            axis = axes[row, column]
            subset = data[
                (data["window_sessions"] == window_size) & (data["factor"] == factor)
            ]
            pivot = subset.pivot(
                index="stock", columns="window_end", values="is_active"
            ).reindex(STOCKS)
            dates = pd.DatetimeIndex(pd.to_datetime(pivot.columns))
            image = axis.imshow(
                pivot.astype(float).to_numpy(),
                cmap="Blues",
                vmin=0.0,
                vmax=1.0,
                aspect="auto",
                interpolation="nearest",
            )
            _shade_index_crisis(axis, dates)
            _format_image_dates(axis, dates)
            axis.set_title(f"{window_size}-session: {factor}")
            axis.set_yticks(range(len(STOCKS)))
            axis.set_yticklabels(STOCKS if column == 0 else [])
            axis.tick_params(axis="both", length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
    colorbar_axis = figure.add_axes([0.925, 0.16, 0.015, 0.68])
    figure.colorbar(image, cax=colorbar_axis, label="Active support (1 = active)")
    figure.suptitle(
        "Rolling L1-rotation support membership", x=0.06, ha="left", y=0.98, fontsize=14
    )
    figure.text(
        0.06,
        0.945,
        "Support uses |structural loading| ≥ h_n; labels are aligned ex post for readability.",
        color="#666666",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.05, right=0.9, bottom=0.1, top=0.86, wspace=0.14, hspace=0.28
    )
    save_figure(
        figure, out_dir / "14_rolling_support_membership_heatmaps.png", tight_bbox=True
    )


def plot_stability(stability: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot past-only cosine and support-Jaccard diagnostics."""
    data = stability[stability["alignment_method"] == "past_only"]
    figure, axes = plt.subplots(
        len(WINDOWS), 2, figsize=(15, 8), squeeze=False, sharex="col", sharey="col"
    )
    metrics = [
        ("cosine_similarity_anchor", "Cosine to past-only anchor", 0.8),
        ("support_jaccard_anchor", "Support Jaccard to past-only anchor", 0.5),
    ]
    for row, window_size in enumerate(WINDOWS):
        subset = data[data["window_sessions"] == window_size]
        for column, (metric, ylabel, threshold) in enumerate(metrics):
            axis = axes[row, column]
            for factor in FACTORS:
                factor_data = subset[subset["factor"] == factor].sort_values(
                    "window_end"
                )
                axis.plot(
                    pd.to_datetime(factor_data["window_end"]),
                    factor_data[metric],
                    color=COLORS[factor],
                    linewidth=1.4,
                    label=factor,
                )
            axis.axhline(threshold, color="#777777", linestyle="--", linewidth=1.0)
            _shade_crisis(axis, pd.to_datetime(subset["window_end"]))
            axis.set_ylim(-0.02, 1.05)
            axis.set_title(f"{window_size}-session window")
            axis.set_ylabel(ylabel)
            style_axis(axis, format_dates=True)
            if row == len(WINDOWS) - 1:
                axis.set_xlabel("Window end")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Past-only factor stability diagnostics", x=0.06, ha="left", y=1.04, fontsize=14
    )
    figure.text(
        0.06,
        1.005,
        "Dashed lines are descriptive instability thresholds, not structural-break tests.",
        color="#666666",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(figure, out_dir / "14_rolling_factor_stability.png", tight_bbox=True)


def plot_small_loading_counts(stability: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot small-loading counts against the local-factor critical count."""
    data = stability[stability["alignment_method"] == "past_only"]
    figure, axes = plt.subplots(len(WINDOWS), 1, figsize=(12, 7), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, window_size in zip(axes, WINDOWS):
        subset = data[data["window_sessions"] == window_size]
        for factor in FACTORS:
            factor_data = subset[subset["factor"] == factor].sort_values("window_end")
            axis.plot(
                pd.to_datetime(factor_data["window_end"]),
                factor_data["n_small_loadings"],
                color=COLORS[factor],
                linewidth=1.4,
                label=factor,
            )
        gamma = float(subset["gamma_n"].iloc[0])
        axis.axhline(
            gamma, color="#777777", linestyle="--", linewidth=1.0, label="gamma_n"
        )
        _shade_crisis(axis, pd.to_datetime(subset["window_end"]))
        axis.set_ylim(0, len(STOCKS) + 0.2)
        axis.set_ylabel("Small loadings")
        axis.set_title(f"{window_size}-session window")
        style_axis(axis, format_dates=True)
    axes[-1].set_xlabel("Window end")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Local-factor diagnostic through time", x=0.08, ha="left", y=1.05, fontsize=14
    )
    figure.text(
        0.08,
        1.015,
        "A factor is local under the reference rule only when its small-loading count exceeds gamma_n.",
        color="#666666",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(figure, out_dir / "14_small_loading_counts.png", tight_bbox=True)


def plot_variance_diagnostics(diagnostics: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot explained variance and rolling SPY/XLF variance removal."""
    data = diagnostics[diagnostics["alignment_method"] == "past_only"]
    metrics = [
        ("pc1_explained_pct", "PC1 explained variance (%)"),
        ("cumulative_explained_pct_first3", "First-three explained variance (%)"),
        ("benchmark_variance_removed_pct", "Mean SPY/XLF variance removed (%)"),
    ]
    figure, axes = plt.subplots(
        len(WINDOWS), len(metrics), figsize=(17, 8), sharex="col"
    )
    axes = np.atleast_2d(axes)
    for row, window_size in enumerate(WINDOWS):
        subset = data[data["window_sessions"] == window_size].sort_values("window_end")
        for column, (metric, ylabel) in enumerate(metrics):
            axis = axes[row, column]
            axis.plot(
                pd.to_datetime(subset["window_end"]),
                subset[metric],
                color="#2F6690"
                if metric != "benchmark_variance_removed_pct"
                else "#C26A2E",
                linewidth=1.5,
            )
            _shade_crisis(axis, pd.to_datetime(subset["window_end"]))
            axis.set_title(f"{window_size}-session window")
            axis.set_ylabel(ylabel)
            style_axis(axis, format_dates=True)
            if row == len(WINDOWS) - 1:
                axis.set_xlabel("Window end")
    figure.suptitle(
        "Rolling variance and benchmark diagnostics",
        x=0.06,
        ha="left",
        y=1.02,
        fontsize=14,
    )
    figure.text(
        0.06,
        0.985,
        "All quantities are re-estimated within the trailing window; shaded interval is March–May 2023.",
        color="#666666",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(
        figure, out_dir / "14_variance_benchmark_diagnostics.png", tight_bbox=True
    )


def _grouped_bar(
    axis: Axes, frame: pd.DataFrame, value: str, ylabel: str, title: str
) -> None:
    """Draw a compact factor-by-regime grouped bar chart."""
    regimes = list(frame["regime"].drop_duplicates())
    positions = np.arange(len(regimes))
    width = 0.24
    for index, factor in enumerate(FACTORS):
        values = []
        for regime in regimes:
            subset = frame[(frame["regime"] == regime) & (frame["factor"] == factor)]
            values.append(float(subset[value].iloc[0]) if len(subset) else np.nan)
        axis.bar(
            positions + (index - 1) * width,
            values,
            width=width,
            color=COLORS[factor],
            label=factor,
        )
    axis.set_xticks(positions, [regime.replace("_", "\n") for regime in regimes])
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    style_axis(axis)


def plot_regime_comparison(regime_summary: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot the explicit pre-crisis, crisis, post-crisis, and recent comparison."""
    data = regime_summary[regime_summary["alignment_method"] == "past_only"]
    selected = data[data["window_sessions"] == 60]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    _grouped_bar(
        axes[0],
        selected,
        "mean_cosine_similarity_anchor",
        "Mean cosine similarity",
        "60-session regime comparison",
    )
    _grouped_bar(
        axes[1],
        selected,
        "mean_support_jaccard_anchor",
        "Mean support Jaccard",
        "60-session regime comparison",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Factor stability by market regime", x=0.06, ha="left", y=1.08, fontsize=14
    )
    figure.text(
        0.06,
        1.03,
        "Regime is assigned by the majority of sessions in a window; mixed windows are retained and flagged.",
        color="#666666",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.9])
    save_figure(figure, out_dir / "14_regime_comparison.png", tight_bbox=True)


def plot_window_comparison(stability: pd.DataFrame, *, out_dir: Path) -> None:
    """Compare 60- and 120-session stability summaries."""
    data = stability[stability["alignment_method"] == "past_only"]
    summary = data.groupby(["window_sessions", "factor"], as_index=False).agg(
        mean_cosine=("cosine_similarity_anchor", "mean"),
        mean_jaccard=("support_jaccard_anchor", "mean"),
        local_rate=("is_local_factor", "mean"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True)
    metrics = [
        ("mean_cosine", "Mean past-only cosine", (0.0, 1.02)),
        ("mean_jaccard", "Mean past-only Jaccard", (0.0, 1.02)),
        ("local_rate", "Local-factor rate", (0.0, 1.02)),
    ]
    positions = np.arange(len(FACTORS))
    width = 0.34
    for axis, (metric, ylabel, limits) in zip(axes, metrics):
        for index, window_size in enumerate(WINDOWS):
            values = [
                float(
                    summary[
                        (summary["window_sessions"] == window_size)
                        & (summary["factor"] == factor)
                    ][metric].iloc[0]
                )
                for factor in FACTORS
            ]
            axis.bar(
                positions + (index - 0.5) * width,
                values,
                width=width,
                label=f"{window_size} sessions",
                color="#2F6690" if window_size == 60 else "#D99A2B",
            )
        axis.set_xticks(positions, FACTORS)
        axis.set_ylim(*limits)
        axis.set_ylabel(ylabel)
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "60-session versus 120-session factor diagnostics",
        x=0.06,
        ha="left",
        y=1.08,
        fontsize=14,
    )
    figure.text(
        0.06,
        1.03,
        "The 60-session window is primary; 120 sessions is the persistence robustness check.",
        color="#666666",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.9])
    save_figure(figure, out_dir / "14_window_size_comparison.png", tight_bbox=True)


def make_plots(
    loadings: pd.DataFrame,
    support: pd.DataFrame,
    stability: pd.DataFrame,
    diagnostics: pd.DataFrame,
    regime_summary: pd.DataFrame,
    *,
    out_dir: Path,
) -> None:
    """Create all requested diagnostic figures."""
    plot_loading_heatmaps(loadings, out_dir=out_dir)
    plot_support_heatmaps(support, out_dir=out_dir)
    plot_stability(stability, out_dir=out_dir)
    plot_small_loading_counts(stability, out_dir=out_dir)
    plot_variance_diagnostics(diagnostics, out_dir=out_dir)
    plot_regime_comparison(regime_summary, out_dir=out_dir)
    plot_window_comparison(stability, out_dir=out_dir)


def print_summary(results: dict[str, Any], *, out_dir: Path) -> None:
    """Print a concise real-data summary for reproducibility logs."""
    stability = results["stability"]
    diagnostics = results["diagnostics"]
    sensitivity = results["sensitivity"]
    primary = stability[stability["alignment_method"] == "past_only"]
    print("=== DYNAMIC L1 LOCAL FACTOR REGIMES ===")
    print(f"Windows: {WINDOWS}; session step: {STEP_SESSIONS}")
    print(f"Primary starts: {PRIMARY_STARTS}; sensitivity starts: {SENSITIVITY_STARTS}")
    print(f"Primary rows: {len(primary)} factor-window rows")
    print(
        f"Sensitivity windows: {(sensitivity['window_id'].nunique() if not sensitivity.empty else 0)}"
    )
    summary = primary.groupby(["window_sessions", "factor"], as_index=False).agg(
        mean_cosine=("cosine_similarity_anchor", "mean"),
        p05_cosine=("cosine_similarity_anchor", lambda values: values.quantile(0.05)),
        mean_jaccard=("support_jaccard_anchor", "mean"),
        local_rate=("is_local_factor", "mean"),
        n_regime_candidates=("regime_candidate", "sum"),
    )
    print(summary.round(4).to_string(index=False))
    print("\nRegime summary (60-session, past-only):")
    print(
        results["regime_summary"][
            (results["regime_summary"]["window_sessions"] == 60)
            & (results["regime_summary"]["alignment_method"] == "past_only")
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nWindow diagnostics:")
    print(diagnostics.head().round(4).to_string(index=False))
    print(f"\nOutputs saved to: {out_dir}")
