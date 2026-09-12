"""Tabular inference and presentation for the Stage 17 Factor-HAR experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from reporting.markdown import markdown_table
from utils.network_har import hac_mean_test


def summarize_models(scored: pd.DataFrame) -> pd.DataFrame:
    """Summarize matched, scoreable forecasts by specification and model."""

    eligible = scored[scored["comparison_eligible"] & scored["qlike"].notna()]
    if eligible.empty:
        return pd.DataFrame()
    return (
        eligible.groupby(["spec_name", "model"], as_index=False)
        .agg(
            n=("qlike", "size"),
            mean_qlike=("qlike", "mean"),
            median_qlike=("qlike", "median"),
            log_mse=("log_mse", "mean"),
            log_mae=("log_mae", "mean"),
        )
        .sort_values(["spec_name", "mean_qlike"])
    )


def build_hac_comparisons(
    scored: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    """Test date-level mean QLIKE differences with Bartlett-HAC errors."""

    eligible = scored[scored["comparison_eligible"] & scored["qlike"].notna()]
    keys = ["spec_name", "stock", "forecast_origin", "target_date"]
    wide = eligible.pivot(index=keys, columns="model", values="qlike")
    rows: list[dict[str, object]] = []
    for model_a, model_b in pairs:
        if model_a not in wide or model_b not in wide:
            continue
        difference = (
            (wide[model_a] - wide[model_b])
            .dropna()
            .rename("difference")
            .reset_index()
        )
        for spec_name, group in difference.groupby("spec_name"):
            date_difference = group.groupby("target_date")["difference"].mean()
            for lag in (5, 20):
                rows.append(
                    {
                        "spec_name": spec_name,
                        "model_a": model_a,
                        "model_b": model_b,
                        "hac_lag": lag,
                        **hac_mean_test(date_difference, max_lag=lag),
                    }
                )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["holm_p_value_within_spec_lag"] = np.nan
    for _, indices in result.groupby(["spec_name", "hac_lag"]).groups.items():
        ordered = result.loc[indices, "p_value"].sort_values()
        adjusted = np.maximum.accumulate(
            ordered.to_numpy(dtype=float) * np.arange(len(ordered), 0, -1)
        )
        result.loc[ordered.index, "holm_p_value_within_spec_lag"] = np.clip(
            adjusted, 0.0, 1.0
        )
    return result


def write_results_markdown(
    path: Path,
    *,
    manifest: Mapping[str, object],
    model_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    hac: pd.DataFrame,
) -> None:
    """Write the self-contained Stage 17 run report."""

    lines = [
        "# Stage 17 OOS Factor-HAR Results",
        "",
        f"Status: `{manifest.get('status', 'unknown')}`.",
        "",
        "The target is next-session realized variance after removing only SPY and XLF. "
        "Every reported comparison uses identical forecast keys. Negative mean QLIKE "
        "differences favor model A over model B. Historical audit results are pseudo-OOS "
        "development evidence and are not an untouched holdout.",
        "",
    ]
    if model_summary.empty:
        lines.append("No outcomes are currently available for a matched evaluation.")
    else:
        lines.extend(["## Model summary", "", markdown_table(model_summary), ""])
        lines.extend(["## Predeclared comparisons", "", markdown_table(comparisons), ""])
        lines.extend(["## Date-clustered HAC tests", "", markdown_table(hac), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
