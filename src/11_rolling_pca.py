"""Descriptive rolling covariance and correlation PCA on the CORE panel."""

from __future__ import annotations

from config import (
    BENCHMARKS,
    REPORTS_DIR,
    RETURN_MATRIX_CLEAN_FILE,
    ensure_project_directories,
)
from reporting.rolling_pca import (
    plot_benchmark_variance_removed,
    plot_correlation_metrics,
    plot_loading_stability,
    print_summary,
)
from utils.data import load_panel, normalize_intraday_volatility
from utils.rolling import session_index
from utils.rolling_pca import (
    STEP_SESSIONS,
    STOCKS,
    WINDOWS,
    collect_rolling_results,
)

OUT_DIR = REPORTS_DIR / "rolling_pca"


def main() -> None:
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    required = STOCKS + list(BENCHMARKS)
    panel = load_panel(
        RETURN_MATRIX_CLEAN_FILE,
        required,
        context="Complete benchmark panel",
        drop_incomplete=True,
    )
    sessions, _ = session_index(panel.index)
    if len(sessions) < max(WINDOWS):
        raise ValueError("The panel does not contain enough sessions for rolling PCA.")
    normalized_returns = normalize_intraday_volatility(panel[STOCKS]).normalized_returns
    metrics, loadings, benchmark_diagnostics = collect_rolling_results(
        panel,
        normalized_returns,
        stocks=STOCKS,
        window_sizes=WINDOWS,
        step_sessions=STEP_SESSIONS,
    )
    metrics.to_csv(OUT_DIR / "11_rolling_pca_metrics.csv", index=False)
    loadings.to_csv(OUT_DIR / "11_rolling_pca_loadings.csv", index=False)
    benchmark_diagnostics.to_csv(
        OUT_DIR / "11_rolling_benchmark_diagnostics.csv", index=False
    )
    plot_correlation_metrics(metrics, out_dir=OUT_DIR)
    plot_loading_stability(metrics, out_dir=OUT_DIR)
    plot_benchmark_variance_removed(metrics, out_dir=OUT_DIR)
    print_summary(metrics, panel, sessions, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
