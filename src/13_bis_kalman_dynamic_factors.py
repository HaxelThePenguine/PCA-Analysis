"""13-bis: session Kalman factors followed by the existing L1 rotation."""

from __future__ import annotations

import pandas as pd

from config import (
    REPORTS_DIR,
    ensure_project_directories,
)
from reporting.kalman import plot_loadings, print_summary
from utils.benchmark import residualize_against_benchmarks
from utils.data import load_benchmark_panel, validate_panel
from utils.kalman import (
    STOCKS,
    build_comparison_summary,
    build_stress_summary,
    candidate_summary,
    collect_k_sensitivity,
    collect_window_results,
    make_windows,
)
from utils.l1_rotation import fit_l1_rotation
from utils.pca import fit_pca

PRIMARY_K = 3
SENSITIVITY_K = (2, 4, 5)
PRIMARY_STARTS = 80
SENSITIVITY_STARTS = 30
SENSITIVITY_EVERY = 5
SEED = 13213
SMALL_TOLERANCE = 1e-08
CRISIS_START = pd.Timestamp("2023-03-01")
CRISIS_END = pd.Timestamp("2023-05-31")
OUT_DIR = REPORTS_DIR / "kalman_dynamic_local_factors"


def load_session_residuals() -> pd.DataFrame:
    """Residualize the intraday panel, then aggregate residual returns by session."""
    panel = load_benchmark_panel()
    residuals = residualize_against_benchmarks(panel, stocks=STOCKS)["residual_returns"]
    sessions = residuals.groupby(residuals.index.normalize(), sort=True).sum()
    sessions.index.name = "session"
    validate_panel(sessions, context="Session residual panel", require_complete=True)
    return sessions


def main() -> None:
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_session_residuals()
    windows = make_windows(panel)
    static_pca = fit_pca(panel, method="correlation")
    static_result = fit_l1_rotation(
        static_pca, n_components=PRIMARY_K, n_starts=PRIMARY_STARTS, random_state=SEED
    )
    static_loadings = static_result.rotated_loadings.to_numpy(dtype=float)
    loadings, metrics, groups, comparisons = collect_window_results(
        panel, windows, static_loadings, k=PRIMARY_K, starts=PRIMARY_STARTS, seed=SEED
    )
    candidates = candidate_summary(groups)
    sensitivity = collect_k_sensitivity(
        panel,
        windows,
        factor_counts=SENSITIVITY_K,
        every=SENSITIVITY_EVERY,
        starts=SENSITIVITY_STARTS,
        seed=SEED,
    )
    stress = build_stress_summary(groups, CRISIS_START, CRISIS_END)
    summary = build_comparison_summary(
        static_pca, static_loadings, metrics, comparisons
    )
    static_reference = pd.DataFrame(
        [
            {"factor": f"LF{i + 1}", "stock": stock, "loading": static_loadings[row, i]}
            for i in range(PRIMARY_K)
            for row, stock in enumerate(STOCKS)
        ]
    )
    outputs = {
        "13_bis_dynamic_loadings.csv": loadings,
        "13_bis_dynamic_factor_metrics.csv": metrics,
        "13_bis_static_vs_kalman_windows.csv": comparisons,
        "13_bis_comparison_summary.csv": summary,
        "13_bis_k_sensitivity.csv": sensitivity,
        "13_bis_structural_group_metrics.csv": groups,
        "13_bis_structural_candidates.csv": candidates,
        "13_bis_regional_stress_comparison.csv": stress,
        "13_bis_static_reference_loadings.csv": static_reference,
    }
    for filename, table in outputs.items():
        table.to_csv(OUT_DIR / filename, index=False)
    plot_loadings(loadings, out_dir=OUT_DIR)
    print_summary(candidates, panel, stress, summary, windows, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
