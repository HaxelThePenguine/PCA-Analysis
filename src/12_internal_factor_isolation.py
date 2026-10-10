"""Compare PCA, Varimax, and Elastic-Net Sparse PCA factor structures."""

from __future__ import annotations

import pandas as pd

from config import (
    CORE_UNIVERSE,
    REPORTS_DIR,
    ensure_project_directories,
)
from reporting.factors import (
    plot_loading_heatmaps,
    plot_sparse_path,
    print_summary,
)
from utils.benchmark import (
    build_factor_panels,
)
from utils.data import load_benchmark_panel
from utils.factor import fit_sparse_penalty_path, fit_transformation

OUT_DIR = REPORTS_DIR / "internal_factor_isolation"
STOCKS = list(CORE_UNIVERSE)


def main() -> None:
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_benchmark_panel()
    factor_panels, residualization = build_factor_panels(panel, STOCKS)
    all_loading_rows = []
    all_metric_rows = []
    all_path_rows = []
    score_frames = []
    rotation_tables = []
    fitted = {}
    for transformation, data in factor_panels.items():
        result = fit_transformation(transformation, data)
        fitted[transformation] = result
        all_loading_rows.extend(result["loading_rows"])
        all_metric_rows.extend(result["metric_rows"])
        all_path_rows.extend(fit_sparse_penalty_path(transformation, result["pca"]))
        score_frames.append(result["score_frame"])
        rotation = result["varimax"].rotation.copy()
        rotation.insert(0, "transformation", transformation)
        rotation_tables.append(rotation.reset_index(names="source_component"))
    loadings = pd.DataFrame(all_loading_rows)
    metrics = pd.DataFrame(all_metric_rows)
    sparse_path = pd.DataFrame(all_path_rows)
    scores = pd.concat(score_frames, axis=1)
    rotations = pd.concat(rotation_tables, ignore_index=True)
    loadings.to_csv(OUT_DIR / "12_factor_loadings.csv", index=False)
    metrics.to_csv(OUT_DIR / "12_factor_metrics.csv", index=False)
    sparse_path.to_csv(OUT_DIR / "12_sparse_penalty_path.csv", index=False)
    rotations.to_csv(OUT_DIR / "12_varimax_rotation.csv", index=False)
    scores.to_parquet(OUT_DIR / "12_factor_scores.parquet")
    residualization["coefficients"].to_csv(OUT_DIR / "12_benchmark_coefficients.csv")
    residualization["diagnostics"].to_csv(OUT_DIR / "12_benchmark_diagnostics.csv")
    plot_loading_heatmaps(loadings, out_dir=OUT_DIR)
    plot_sparse_path(sparse_path, out_dir=OUT_DIR)
    print_summary(fitted, panel, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
