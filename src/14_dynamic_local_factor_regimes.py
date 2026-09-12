"""Rolling stability and regime diagnostics for the L1 local-factor stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from config import (
    BENCHMARKS,
    REPORTS_DIR,
    RETURN_MATRIX_CLEAN_FILE,
    ensure_project_directories,
)
from reporting.regimes import make_plots, print_summary
from utils.data import load_panel, validate_panel
from utils.dynamic_local_factor_regimes import (
    FULL_SAMPLE_RANDOM_STATE,
    FULL_SAMPLE_STARTS,
    PRIMARY_STARTS,
    RANDOM_STATE,
    SENSITIVITY_STARTS,
    STEP_SESSIONS,
    STOCKS,
    WINDOWS,
    _annotate_persistence,
    _merge_sensitivity_annotations,
    build_regime_summary,
    build_unstable_windows,
    build_window_definitions,
    build_window_diagnostics,
    fit_full_sample_reference,
    run_rolling_fits,
    run_start_count_sensitivity,
)

OUT_DIR = REPORTS_DIR / "dynamic_local_factor_regimes"
OUTPUT_FILES = {
    "14_rolling_l1_loadings.csv": "loadings",
    "14_rolling_score_loadings.csv": "score_loadings",
    "14_rolling_factor_stability.csv": "stability",
    "14_rolling_support_membership.csv": "support",
    "14_rolling_window_diagnostics.csv": "diagnostics",
    "14_regime_summary.csv": "regime_summary",
    "14_unstable_windows.csv": "unstable_windows",
    "14_start_count_sensitivity.csv": "sensitivity",
    "14_rolling_score_correlations.csv": "score_correlations",
}


def write_outputs(results: dict[str, Any], *, out_dir: Path) -> None:
    """Write all machine-readable stage-14 artifacts to the ignored report folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, key in OUTPUT_FILES.items():
        results[key].to_csv(out_dir / filename, index=False)


def run_analysis(
    panel: pd.DataFrame | None = None,
    *,
    window_sizes: Iterable[int] = WINDOWS,
    step_sessions: int = STEP_SESSIONS,
    primary_starts: int = PRIMARY_STARTS,
    sensitivity_starts: int = SENSITIVITY_STARTS,
    full_sample_starts: int = FULL_SAMPLE_STARTS,
    base_seed: int = RANDOM_STATE,
    full_sample_seed: int = FULL_SAMPLE_RANDOM_STATE,
    make_figures: bool = True,
    write_files: bool = True,
) -> dict[str, Any]:
    """Run the dynamic local-factor regime analysis and return all result tables."""
    if panel is None:
        required = STOCKS + list(BENCHMARKS)
        panel = load_panel(
            RETURN_MATRIX_CLEAN_FILE,
            required,
            context="Dynamic local-factor panel",
            drop_incomplete=True,
        )
    else:
        panel = panel.copy()
        validate_panel(
            panel, context="Dynamic local-factor panel", require_complete=True
        )
        missing = [
            column for column in STOCKS + list(BENCHMARKS) if column not in panel
        ]
        if missing:
            raise ValueError(
                f"Dynamic local-factor panel is missing columns: {missing}"
            )
        panel = panel.loc[:, STOCKS + list(BENCHMARKS)]
    window_sizes = tuple((int(size) for size in window_sizes))
    if set(window_sizes) != set(WINDOWS):
        raise ValueError(f"The production specification must use windows {WINDOWS}.")
    definitions = build_window_definitions(panel, window_sizes, step_sessions)
    definitions_by_id = {definition.window_id: definition for definition in definitions}
    if not definitions:
        raise ValueError("The panel does not contain enough sessions for stage 14.")
    full_reference = fit_full_sample_reference(
        panel, n_starts=full_sample_starts, random_seed=full_sample_seed
    )
    fits, loadings, score_loadings, support, extra = run_rolling_fits(
        panel,
        definitions,
        primary_starts=primary_starts,
        full_sample_reference=full_reference,
        base_seed=base_seed,
    )
    stability = _annotate_persistence(extra["stability"])
    primary_candidates = set(
        stability.loc[
            (stability["alignment_method"] == "past_only")
            & (stability["is_quarterly_window"] | stability["instability_flag"]),
            "window_id",
        ]
    )
    sensitivity = run_start_count_sensitivity(
        panel,
        definitions_by_id,
        fits,
        extra["contexts"],
        primary_candidates,
        sensitivity_starts=sensitivity_starts,
        base_seed=base_seed,
    )
    stability = _merge_sensitivity_annotations(stability, sensitivity)
    diagnostics = build_window_diagnostics(stability)
    diagnostics = diagnostics.sort_values(
        ["window_sessions", "window_end", "alignment_method"]
    ).reset_index(drop=True)
    regime_summary = build_regime_summary(stability)
    unstable_windows = build_unstable_windows(stability)
    for frame in (
        loadings,
        score_loadings,
        support,
        stability,
        regime_summary,
        unstable_windows,
    ):
        if not frame.empty:
            sort_columns = [
                column
                for column in (
                    "window_sessions",
                    "window_end",
                    "alignment_method",
                    "regime",
                    "factor",
                )
                if column in frame.columns
            ]
            frame.sort_values(sort_columns, inplace=True)
            frame.reset_index(drop=True, inplace=True)
    extra["diagnostics"] = diagnostics
    extra["stability"] = stability
    extra["regime_summary"] = regime_summary
    extra["unstable_windows"] = unstable_windows
    extra["sensitivity"] = sensitivity
    extra["loadings"] = loadings
    extra["score_loadings"] = score_loadings
    extra["support"] = support
    extra["fits"] = fits
    extra["full_sample_reference"] = full_reference
    if write_files:
        ensure_project_directories()
        write_outputs(extra, out_dir=OUT_DIR)
    if make_figures:
        ensure_project_directories()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        make_plots(
            loadings, support, stability, diagnostics, regime_summary, out_dir=OUT_DIR
        )
    return extra


def main() -> None:
    """Run stage 14 on the configured real-data panel."""
    results = run_analysis()
    print_summary(results, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
