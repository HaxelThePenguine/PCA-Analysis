"""Stage 16 forecast results, uncertainty, and network diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from reporting.markdown import markdown_table

def _report_table(frame: pd.DataFrame) -> str:
    return markdown_table(
        frame, max_rows=25, float_digits=6,
        empty="No observations were available.", missing="unavailable",
    )


def write_results_markdown(
    path: Path,
    *,
    manifest: Mapping[str, object],
    model_summary: pd.DataFrame,
    stock_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    hac: pd.DataFrame,
    bootstrap: pd.DataFrame,
    edge_stability: pd.DataFrame,
    forecast_status: str,
) -> None:
    """Save Stage 16 model results and evaluation notes."""

    status = str(manifest.get("status", forecast_status))
    lines = [
        "# Stage 16 OOS Results",
        "",
        f"Protocol: `{manifest.get('protocol_version', 'unknown')}`.",
        f"Run mode: `{manifest.get('mode', 'unknown')}`.  Status: **{status}**.",
        f"Freeze timestamp: `{manifest.get('freeze_timestamp_utc', 'unknown')}`.  Last data examined: `{manifest.get('last_data_examined', 'unknown')}`.",
        "",
        "## Evaluation sample",
        "",
        "The historical walk-forward sample is pseudo-OOS because it informed model development. Prospective forecasts are saved at issuance and scored once their outcomes are observed.",
        "",
        f"Forecast status: `{forecast_status}`. Prospective evaluation requires 252 exchange sessions after the freeze. Missing future outcomes remain pending.",
        "",
        "## Forecast comparisons",
        "",
        _report_table(comparison),
        "",
        "The primary comparison averages Network-HAR minus Own-HAR QLIKE across stocks within each date. Negative differences favor Network HAR. Loss levels across factor windows are not directly comparable because the residual targets differ.",
        "",
        "### Model summaries",
        "",
        _report_table(model_summary),
        "",
        "### Per-stock summaries",
        "",
        _report_table(stock_summary),
        "",
        "## Uncertainty",
        "",
        "Bartlett-HAC tests use date-level loss differences. The moving-block bootstrap resamples the same dates for all stocks. Its percentile interval describes the observed loss series; it does not include all uncertainty from factor estimation, tuning, or model discovery.",
        "",
        _report_table(hac),
        "",
        _report_table(bootstrap),
        "",
        "Per-stock tests are secondary, with Benjamini–Hochberg and Holm adjustments where available. The one-SE penalty rule uses three chronological folds to choose a model; it is not a confidence interval.",
        "",
        "## Network diagnostics",
        "",
        _report_table(edge_stability),
        "",
        "Edge persistence and bootstrap selection rates measure stability, separately from forecast gains. The edge bootstrap holds factor features and the observed penalty fixed. Directed edges describe conditional predictability and do not establish causality or trading profitability.",
        "",
        "## Target and method",
        "",
        "The target is factor-adjusted realized variance from complete, non-overlapping five-minute bins. HAR uses log variance; QLIKE uses raw positive variance. Forecasts use information through session d's close to predict session d+1.",
        "",
        "Daily, weekly, and monthly predictors follow Corsi (2009), using means of log variance. Patton (2011) motivates QLIKE for noisy volatility proxies. The nested, expanding, penalized models require caution when interpreting predictive tests; see Giacomini and White (2006).",
        "",
        "References: [Corsi (2009)](https://academic.oup.com/jfec/article-abstract/7/2/174/856522), [Patton (2011)](https://doi.org/10.1016/j.jeconom.2010.03.034), and [Giacomini and White (2006)](https://doi.org/10.1111/j.1468-0262.2006.00718.x).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figures(
    scored: pd.DataFrame,
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    edge_stability: pd.DataFrame,
    *,
    out_dir: Path,
) -> None:
    """Save forecast-loss and network figures if plotting is available."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if scored is not None and not scored.empty:
        value = scored[scored["score_status"] == "scored"].copy()
        daily = (
            value[value["model"].isin(["own_har", "network_har_l1_1se"])]
            .groupby(["spec_name", "target_date", "model"], as_index=False)["qlike"]
            .mean()
            .pivot(index=["spec_name", "target_date"], columns="model", values="qlike")
            .reset_index()
        )
        if {"own_har", "network_har_l1_1se"}.issubset(daily.columns):
            daily["loss_difference"] = daily["network_har_l1_1se"] - daily["own_har"]
        differential = daily.dropna(subset=["loss_difference"]).loc[
            :, ["spec_name", "target_date", "loss_difference"]
        ]
        if not differential.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            for spec_name, group in differential.groupby("spec_name"):
                group = group.sort_values("target_date")
                ax.plot(group["target_date"], group["loss_difference"].cumsum(), label=str(spec_name))
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title("Cumulative Network HAR minus own-HAR QLIKE")
            ax.set_ylabel("Cumulative QLIKE differential")
            ax.legend(frameon=False)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(out_dir / "oos_cumulative_qlike_difference.png", dpi=150)
            plt.close(fig)

    if comparison is not None and not comparison.empty and {"spec_name", "network_qlike_minus_own"}.issubset(comparison.columns):
        fig, ax = plt.subplots(figsize=(9, 4))
        ordered = comparison.sort_values("network_qlike_minus_own")
        ax.bar(ordered["spec_name"].astype(str), ordered["network_qlike_minus_own"])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("Date-aggregated QLIKE differential by specification")
        ax.set_ylabel("Network minus own HAR")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(out_dir / "oos_qlike_comparison.png", dpi=150)
        plt.close(fig)

    if bootstrap is not None and not bootstrap.empty:
        value = bootstrap.copy()
        if {"spec_name", "observed_mean_difference", "bootstrap_ci_low", "bootstrap_ci_high"}.issubset(value.columns):
            fig, ax = plt.subplots(figsize=(9, 4))
            x = np.arange(len(value))
            means = value["observed_mean_difference"].to_numpy(dtype=float)
            low = means - value["bootstrap_ci_low"].to_numpy(dtype=float)
            high = value["bootstrap_ci_high"].to_numpy(dtype=float) - means
            ax.errorbar(x, means, yerr=np.vstack([low, high]), fmt="o")
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xticks(x, value["spec_name"].astype(str), rotation=25, ha="right")
            ax.set_title("Moving-block bootstrap intervals")
            ax.set_ylabel("Mean QLIKE differential")
            fig.tight_layout()
            fig.savefig(out_dir / "oos_bootstrap_intervals.png", dpi=150)
            plt.close(fig)
