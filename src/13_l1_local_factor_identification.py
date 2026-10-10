"""Identify local banking factors with the L1-rotation criterion."""

from __future__ import annotations

import pandas as pd

from config import (
    REPORTS_DIR,
    ensure_project_directories,
)
from reporting.local_factors import (
    plot_bootstrap_support,
    plot_factor_diagnostics,
    plot_loading_comparison,
    print_summary,
)
from utils.benchmark import build_factor_panels
from utils.data import load_benchmark_panel
from utils.l1_rotation import L1RotationResult, fit_l1_rotation
from utils.local_factor import (
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_STARTS,
    N_COMPONENTS,
    PRIMARY_STARTS,
    RANDOM_STATE,
    STOCKS,
    bootstrap_local_factors,
    diagnostic_rows,
    factor_count_rows,
    k_sensitivity_rows,
    loading_rows,
)
from utils.pca import PCAResult, fit_pca

OUT_DIR = REPORTS_DIR / "local_factor_identification"


def main() -> None:
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_benchmark_panel()
    factor_panels, _ = build_factor_panels(panel, STOCKS)
    fits: dict[str, L1RotationResult] = {}
    pca_fits: dict[str, PCAResult] = {}
    loading_records = []
    diagnostic_records = []
    factor_count_records = []
    score_frames = []
    score_loading_frames = []
    score_correlation_frames = []
    rotation_frames = []
    for offset, (transformation, data) in enumerate(factor_panels.items()):
        pca = fit_pca(data, method="correlation")
        result = fit_l1_rotation(
            pca,
            n_components=N_COMPONENTS,
            n_starts=PRIMARY_STARTS,
            random_state=RANDOM_STATE + offset,
        )
        pca_fits[transformation] = pca
        fits[transformation] = result
        loading_records.extend(loading_rows(transformation, result))
        diagnostic_records.extend(diagnostic_rows(transformation, result))
        factor_count_records.extend(factor_count_rows(transformation, pca))
        scores = result.scores.add_prefix(f"{transformation}_")
        score_frames.append(scores)
        score_loadings = result.score_loadings.copy()
        score_loadings.insert(0, "transformation", transformation)
        score_loading_frames.append(score_loadings.reset_index(names="stock"))
        score_correlations = (
            result.score_correlation.rename_axis("factor")
            .reset_index()
            .melt(id_vars="factor", var_name="other_factor", value_name="correlation")
        )
        score_correlations.insert(0, "transformation", transformation)
        score_correlation_frames.append(score_correlations)
        rotation = result.rotation.copy()
        rotation.insert(0, "transformation", transformation)
        rotation_frames.append(rotation.reset_index(names="principal_component"))
    probabilities, stability = bootstrap_local_factors(
        factor_panels["benchmark_residual"],
        fits["benchmark_residual"],
        replications=BOOTSTRAP_REPLICATIONS,
        n_starts=BOOTSTRAP_STARTS,
        random_state=RANDOM_STATE,
    )
    loadings = pd.DataFrame(loading_records)
    diagnostics = pd.DataFrame(diagnostic_records)
    factor_counts = pd.DataFrame(factor_count_records)
    k_sensitivity = pd.DataFrame(k_sensitivity_rows(pca_fits["benchmark_residual"]))
    loadings.to_csv(OUT_DIR / "13_l1_loadings.csv", index=False)
    diagnostics.to_csv(OUT_DIR / "13_rotation_diagnostics.csv", index=False)
    factor_counts.to_csv(OUT_DIR / "13_factor_count_diagnostics.csv", index=False)
    k_sensitivity.to_csv(OUT_DIR / "13_k_sensitivity.csv", index=False)
    probabilities.to_csv(OUT_DIR / "13_bootstrap_support_probability.csv", index=False)
    stability.to_csv(OUT_DIR / "13_bootstrap_factor_stability.csv", index=False)
    pd.concat(rotation_frames, ignore_index=True).to_csv(
        OUT_DIR / "13_rotation_matrices.csv", index=False
    )
    pd.concat(score_loading_frames, ignore_index=True).to_csv(
        OUT_DIR / "13_factor_score_loadings.csv", index=False
    )
    pd.concat(score_correlation_frames, ignore_index=True).to_csv(
        OUT_DIR / "13_factor_score_correlations.csv", index=False
    )
    pd.concat(score_frames, axis=1).to_parquet(OUT_DIR / "13_factor_scores.parquet")
    plot_loading_comparison(loadings, out_dir=OUT_DIR)
    plot_bootstrap_support(probabilities, out_dir=OUT_DIR)
    plot_factor_diagnostics(factor_counts, stability, out_dir=OUT_DIR)
    print_summary(fits, k_sensitivity, panel, stability, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
