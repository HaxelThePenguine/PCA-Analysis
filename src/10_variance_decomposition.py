"""Decompose stock variance into SPY/XLF and residual PCA components."""

from __future__ import annotations

from config import (
    BENCHMARKS,
    CORE_UNIVERSE,
    REPORTS_DIR,
    RETURN_MATRIX_CLEAN_FILE,
    ensure_project_directories,
)
from reporting.variance import (
    plot_global_decomposition,
    plot_stock_decomposition,
    print_summary,
)
from utils.data import load_panel, save_csv_tables
from utils.variance import build_global_summary, build_variance_ledger

OUT_DIR = REPORTS_DIR / "variance_decomposition"


def main() -> None:
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stocks = list(CORE_UNIVERSE)
    required = stocks + list(BENCHMARKS)
    complete_panel = load_panel(
        RETURN_MATRIX_CLEAN_FILE,
        required,
        context="Benchmark panel",
        drop_incomplete=True,
    )
    results = build_variance_ledger(complete_panel, stocks)
    ledger = results["ledger"]
    global_summary = build_global_summary(results)
    save_csv_tables(
        {
            "10_variance_ledger.csv": ledger,
            "10_residual_pc_contributions_variance.csv": results["component_variance"],
            "10_residual_pc_contributions_pct_of_raw.csv": results["component_pct"],
            "10_global_variance_decomposition.csv": global_summary,
            "10_residual_covariance_matrix.csv": results["residual_covariance"],
        },
        OUT_DIR,
    )
    plot_stock_decomposition(ledger, complete_panel, out_dir=OUT_DIR)
    plot_global_decomposition(global_summary, out_dir=OUT_DIR)
    print_summary(complete_panel, global_summary, ledger, results, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
