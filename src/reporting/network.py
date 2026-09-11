"""Publication-style figures and a compact narrative for stage 15."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plotting import save_figure, style_axis


COLORS = {
    "own_har": "#777777",
    "network_har_l1_1se": "#2F6690",
    "network_har_l1_min_loss": "#D99A2B",
    "raw": "#777777",
    "benchmark_residual": "#C26A2E",
    "factor_adjusted": "#2F6690",
}


def _empty_figure(path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.text(0.5, 0.5, "No valid observations", ha="center", va="center")
    axis.set_title(title)
    axis.set_axis_off()
    save_figure(figure, path, tight_bbox=True)


def plot_cumulative_qlike(forecasts: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot cumulative network-minus-own QLIKE by forecast date."""

    own = forecasts[forecasts["model"] == "own_har"][
        ["spec_name", "forecast_origin", "target_date", "stock", "qlike"]
    ].rename(columns={"qlike": "own_qlike"})
    network = forecasts[forecasts["model"] == "network_har_l1_1se"][
        ["spec_name", "forecast_origin", "target_date", "stock", "qlike"]
    ].rename(columns={"qlike": "network_qlike"})
    joined = own.merge(
        network,
        on=["spec_name", "forecast_origin", "target_date", "stock"],
        how="inner",
    )
    if joined.empty:
        _empty_figure(out_dir / "15_cumulative_qlike_difference.png", "Cumulative QLIKE difference")
        return
    joined["difference"] = joined["network_qlike"] - joined["own_qlike"]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for spec_name, group in joined.groupby("spec_name"):
        daily = group.groupby("target_date", as_index=False)["difference"].mean().sort_values("target_date")
        axis.plot(
            pd.to_datetime(daily["target_date"]),
            daily["difference"].cumsum(),
            linewidth=1.5,
            label=spec_name,
        )
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.set_ylabel("Cumulative network HAR QLIKE − own HAR QLIKE")
    axis.set_xlabel("Forecast target date")
    axis.set_title("Cumulative QLIKE difference")
    axis.legend(frameon=False)
    style_axis(axis, format_dates=True)
    figure.tight_layout()
    save_figure(figure, out_dir / "15_cumulative_qlike_difference.png", tight_bbox=True)


def plot_stock_qlike_improvement(forecasts: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot stock-level QLIKE improvement with a simple 95% sampling interval."""

    own = forecasts[forecasts["model"] == "own_har"][
        ["spec_name", "forecast_origin", "target_date", "stock", "qlike"]
    ].rename(columns={"qlike": "own_qlike"})
    network = forecasts[forecasts["model"] == "network_har_l1_1se"][
        ["spec_name", "forecast_origin", "target_date", "stock", "qlike"]
    ].rename(columns={"qlike": "network_qlike"})
    joined = own.merge(network, on=["spec_name", "forecast_origin", "target_date", "stock"], how="inner")
    if joined.empty:
        _empty_figure(out_dir / "15_stock_qlike_improvement.png", "Stock-level QLIKE improvement")
        return
    joined["improvement"] = joined["own_qlike"] - joined["network_qlike"]
    summary = joined.groupby(["spec_name", "stock"], as_index=False).agg(
        mean_improvement=("improvement", "mean"),
        sd_improvement=("improvement", "std"),
        n=("improvement", "size"),
    )
    summary["se"] = summary["sd_improvement"].fillna(0.0) / np.sqrt(summary["n"])
    specs = list(summary["spec_name"].drop_duplicates())
    stocks = sorted(summary["stock"].unique())
    positions = np.arange(len(stocks))
    width = 0.8 / max(len(specs), 1)
    figure, axis = plt.subplots(figsize=(12, 5.5))
    for number, spec_name in enumerate(specs):
        data = summary[summary["spec_name"] == spec_name].set_index("stock").reindex(stocks)
        offset = (number - (len(specs) - 1) / 2.0) * width
        axis.bar(
            positions + offset,
            data["mean_improvement"],
            width=width,
            yerr=1.96 * data["se"],
            capsize=2,
            color=COLORS["network_har_l1_1se"] if number == 0 else COLORS["network_har_l1_min_loss"],
            label=spec_name,
        )
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.set_xticks(positions, stocks)
    axis.set_ylabel("Mean QLIKE improvement (own − network)")
    axis.set_title("Stock-level network HAR QLIKE improvement")
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, out_dir / "15_stock_qlike_improvement.png", tight_bbox=True)


def plot_stable_network_heatmap(edge_stability: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot selection probabilities for the primary directed network."""

    data = edge_stability[
        (edge_stability["model"] == "network_har_l1_1se")
        & edge_stability["spec_name"].astype(str).str.contains("120_expanding")
    ]
    if data.empty:
        _empty_figure(out_dir / "15_stable_directed_network_heatmap.png", "Stable directed network")
        return
    stocks = sorted(set(data["source"]) | set(data["target"]))
    matrix = data.pivot_table(index="source", columns="target", values="selection_probability", aggfunc="mean").reindex(index=stocks, columns=stocks).fillna(0.0)
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal")
    axis.set_xticks(range(len(stocks)), stocks, rotation=45, ha="right")
    axis.set_yticks(range(len(stocks)), stocks)
    axis.set_xlabel("Target stock")
    axis.set_ylabel("Source stock")
    axis.set_title("Directed cross-HAR edge selection probability")
    figure.colorbar(image, ax=axis, label="Selection probability")
    figure.tight_layout()
    save_figure(figure, out_dir / "15_stable_directed_network_heatmap.png", tight_bbox=True)


def plot_edge_persistence(edge_history: pd.DataFrame, edge_stability: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot representative primary edge-selection paths through time."""

    history = edge_history[
        (edge_history["model"] == "network_har_l1_1se")
        & edge_history["spec_name"].astype(str).str.contains("120_expanding")
    ]
    stable = edge_stability[
        (edge_stability["model"] == "network_har_l1_1se")
        & edge_stability["spec_name"].astype(str).str.contains("120_expanding")
    ]
    if history.empty or stable.empty:
        _empty_figure(out_dir / "15_edge_selection_persistence.png", "Edge-selection persistence")
        return
    ranked = stable.sort_values(
        ["selection_probability", "median_absolute_strength"],
        ascending=False,
    )
    non_degenerate = ranked[ranked["selection_probability"] < 1.0 - 1e-12]
    selected_stability = non_degenerate.head(6) if len(non_degenerate) >= 6 else ranked.head(6)
    top_edges = selected_stability["edge_id"].tolist()
    probability = selected_stability.set_index("edge_id")["selection_probability"]
    origin = (
        history[history["edge_id"].isin(top_edges)]
        .groupby(["forecast_origin", "edge_id"], as_index=False)["selected"]
        .max()
    )
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for edge in top_edges:
        data = origin[origin["edge_id"] == edge].sort_values("forecast_origin")
        axis.step(
            pd.to_datetime(data["forecast_origin"]),
            data["selected"].astype(float),
            where="post",
            linewidth=1.2,
            label=f"{edge} (p={probability.loc[edge]:.2f})",
        )
    axis.set_ylim(-0.05, 1.05)
    axis.set_ylabel("Selected in refit")
    axis.set_xlabel("Forecast origin")
    title = "Representative directed-edge selection paths"
    if len(non_degenerate) >= 6:
        title += " (primary 120-session expanding specification)"
    else:
        title += " (most persistent primary edges)"
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    style_axis(axis, format_dates=True)
    figure.tight_layout()
    save_figure(figure, out_dir / "15_edge_selection_persistence.png", tight_bbox=True)


def plot_group_connectivity(group_table: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot descriptive within-group selection probabilities across controls."""

    if group_table.empty:
        _empty_figure(out_dir / "15_group_connectivity.png", "Group connectivity")
        return
    selected = group_table[
        (group_table["source_group"] == group_table["target_group"])
        & group_table["spec_name"].astype(str).str.startswith("descriptive_")
    ].copy()
    if selected.empty:
        _empty_figure(out_dir / "15_group_connectivity.png", "Group connectivity")
        return
    selected["group"] = selected["source_group"]
    pivot = selected.pivot_table(index="group", columns="spec_name", values="mean_selection_probability", aggfunc="mean").fillna(0.0)
    figure, axis = plt.subplots(figsize=(11, 5.5))
    order = [
        "descriptive_raw_120",
        "descriptive_raw_60",
        "descriptive_benchmark_residual_120",
        "descriptive_benchmark_residual_60",
        "descriptive_factor_adjusted_120",
        "descriptive_factor_adjusted_60",
    ]
    columns = [column for column in order if column in pivot.columns]
    pivot = pivot.reindex(columns=columns)
    positions = np.arange(len(pivot.index))
    width = 0.8 / max(len(columns), 1)
    control_colors = {
        "raw": COLORS["raw"],
        "benchmark_residual": COLORS["benchmark_residual"],
        "factor_adjusted": COLORS["factor_adjusted"],
    }
    for number, column in enumerate(columns):
        stem = str(column).removeprefix("descriptive_")
        control, window = stem.rsplit("_", 1)
        label = f"{control.replace('_', ' ').title()} / {window} sessions"
        offset = (number - (len(columns) - 1) / 2.0) * width
        axis.bar(
            positions + offset,
            pivot[column].to_numpy(dtype=float),
            width=width,
            color=control_colors[control],
            hatch="//" if window == "60" else "",
            label=label,
        )
    axis.set_xticks(positions, pivot.index)
    axis.set_ylabel("Mean within-group edge selection probability")
    axis.set_xlabel("Economic group")
    axis.set_title("Within-group directed connectivity before and after factor adjustment")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, out_dir / "15_group_connectivity.png", tight_bbox=True)


def plot_variance_removed(factor_diagnostics: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot rolling benchmark and K=3 factor variance removal."""

    if factor_diagnostics.empty:
        _empty_figure(out_dir / "15_variance_removed.png", "Variance removed")
        return
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for window, group in factor_diagnostics.groupby("window_sessions"):
        data = group.sort_values("factor_origin")
        axis.plot(pd.to_datetime(data["factor_origin"]), data["mean_benchmark_variance_removed_pct"], linewidth=1.2, label=f"Benchmark, {window} sessions")
        axis.plot(pd.to_datetime(data["factor_origin"]), data["factor_variance_removed_pct"], linewidth=1.2, linestyle="--", label=f"K=3 factor, {window} sessions")
    axis.set_ylabel("Training variance removed (%)")
    axis.set_xlabel("Factor-model origin")
    axis.set_title("Variance removed by benchmarks and the retained K=3 subspace")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    style_axis(axis, format_dates=True)
    figure.tight_layout()
    save_figure(figure, out_dir / "15_variance_removed.png", tight_bbox=True)


def plot_factor_loading_evolution(factor_loadings: pd.DataFrame, *, out_dir: Path) -> None:
    """Plot the most important primary factor loadings through time."""

    data = factor_loadings[
        factor_loadings["spec_name"].astype(str).eq("factor_adjusted_120")
    ].copy()
    if data.empty:
        _empty_figure(out_dir / "15_factor_loading_evolution.png", "Factor loading evolution")
        return
    data["training_end"] = pd.to_datetime(data["training_end"])
    importance = (
        data.assign(abs_loading=data["loading"].abs())
        .groupby(["factor", "stock"], as_index=False)["abs_loading"]
        .mean()
    )
    factors = sorted(data["factor"].unique())
    figure, axes = plt.subplots(len(factors), 1, figsize=(11, 8), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, factor in zip(axes, factors):
        top_stocks = importance[importance["factor"] == factor].nlargest(4, "abs_loading")["stock"].tolist()
        for stock in top_stocks:
            path = data[(data["factor"] == factor) & (data["stock"] == stock)].sort_values("training_end")
            axis.plot(path["training_end"], path["loading"], linewidth=1.25, label=stock)
        axis.axhline(0.0, color="#444444", linewidth=0.7)
        axis.set_ylabel(factor)
        axis.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Training-window end")
    figure.suptitle("Evolution of the most important K=3 structural loadings", y=0.995)
    style_axis(axes[-1], format_dates=True)
    figure.tight_layout()
    save_figure(figure, out_dir / "15_factor_loading_evolution.png", tight_bbox=True)


def make_figures(
    forecasts: pd.DataFrame,
    edge_history: pd.DataFrame,
    edge_stability: pd.DataFrame,
    group_connectivity: pd.DataFrame,
    factor_diagnostics: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    *,
    out_dir: Path,
) -> None:
    """Render the compact stage-15 figure set."""

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_cumulative_qlike(forecasts, out_dir=out_dir)
    plot_stock_qlike_improvement(forecasts, out_dir=out_dir)
    plot_stable_network_heatmap(edge_stability, out_dir=out_dir)
    plot_edge_persistence(edge_history, edge_stability, out_dir=out_dir)
    plot_group_connectivity(group_connectivity, out_dir=out_dir)
    plot_variance_removed(factor_diagnostics, out_dir=out_dir)
    plot_factor_loading_evolution(factor_loadings, out_dir=out_dir)


def print_summary(
    forecast_summary: pd.DataFrame,
    hac_tests: pd.DataFrame,
    edge_stability: pd.DataFrame,
    *,
    out_dir: Path,
) -> str:
    """Write and print a concise evidence-led stage-15 summary."""

    primary = forecast_summary[
        (forecast_summary["model"].isin(["own_har", "network_har_l1_1se"]))
        & forecast_summary["spec_name"].astype(str).str.contains("120_expanding")
    ]
    test = hac_tests[
        (hac_tests["level"] == "pooled")
        & hac_tests["spec_name"].astype(str).str.contains("120_expanding")
    ]
    if not primary.empty:
        own = primary[primary["model"] == "own_har"]["qlike"].mean()
        network = primary[primary["model"] == "network_har_l1_1se"]["qlike"].mean()
        improvement = 100.0 * (own - network) / own if own else np.nan
        qlike_sentence = f"For the primary 120-session expanding specification, mean pooled QLIKE is {network:.6g} for the one-standard-error network HAR and {own:.6g} for own HAR, a relative improvement of {improvement:.3f} percent."
    else:
        qlike_sentence = "The primary forecast comparison did not produce a valid paired result."
    if not test.empty:
        pooled = test.iloc[0]
        test_sentence = f"The pooled HAC loss differential is {pooled['mean_difference']:.6g} with HAC standard error {pooled['hac_se']:.6g} and nominal p-value {pooled['p_value']:.6g}; negative values favor the network model."
    else:
        test_sentence = "No pooled HAC comparison was available."
    stable = edge_stability[edge_stability["stable_edge"]].sort_values("selection_probability", ascending=False)
    if not stable.empty:
        strongest = ", ".join(
            f"{row.edge_id} ({row.selection_probability:.2f})" for row in stable.head(5).itertuples()
        )
        edge_sentence = f"The most persistent descriptive primary edges are {strongest}. Selection probability is a stability descriptor, not a p-value."
    else:
        edge_sentence = "No directed edge met the descriptive 0.70 persistence rule in the primary network."
    summary = (
        "Stage 15 studies one-session-ahead predictability of factor-adjusted bank realized variance. "
        "The benchmark projection and rolling correlation-PCA/L1 transformation are estimated only from preceding sessions. "
        "Removing all three rotated directions is algebraically identical to removing the complete rolling K=3 PCA subspace; L1 rotation supplies labels and does not change the residual projector.\n\n"
        + qlike_sentence
        + " "
        + test_sentence
        + " "
        + edge_sentence
        + "\n\n"
        "These results are pseudo-out-of-sample because the historical 2023–2026 sample was already inspected during factor discovery and because the lasso is repeatedly selected within that sample. "
        "The directed network is a conditional Granger-predictive description under the stated information set, not structural causality, contagion, alpha, or evidence of deployable trading profitability. "
        "The March–May 2023 stress interval is not used as an outer forecast comparison when the factor-estimation burn-in removes it from the valid forecast sample."
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "15_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print("=== FACTOR-ADJUSTED RESIDUAL NETWORK ===")
    print(summary)
    print(f"Outputs saved to: {out_dir}")
    return summary
