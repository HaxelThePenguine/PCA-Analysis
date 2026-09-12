"""Leakage-controlled benchmark and rolling PCA/L1 residual construction."""

from __future__ import annotations

from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd

from utils.benchmark import residualize_against_benchmarks
from utils.data import validate_panel
from utils.l1_rotation import align_loading_columns, fit_l1_rotation
from utils.pca import fit_pca
from utils.rolling import rolling_starts, session_index


@dataclass(frozen=True)
class FactorConfig:
    """Configuration for one causal rolling factor specification."""

    n_jobs: int = 1
    window_sessions: int = 120
    update_step_sessions: int = 5
    n_components: int = 3
    l1_starts: int = 40
    random_seed: int = 15015
    scale_floor: float = 1e-12


@dataclass(frozen=True)
class FactorFit:
    """All parameters estimated in one historical factor window."""

    stocks: tuple[str, ...]
    benchmarks: tuple[str, ...]
    coefficients: pd.DataFrame
    residual_means: pd.Series
    residual_scales: pd.Series
    pca_explained: np.ndarray
    pca_eigenvalues: np.ndarray
    pca_weights: pd.DataFrame
    factor_projector: pd.DataFrame
    loadings: pd.DataFrame
    rotation: pd.DataFrame
    rotation_condition_number: float
    projector_invariance_error: float
    training_orthogonality_error: float
    training_factor_variance_removed_pct: float
    training_benchmark_variance_removed_pct: float
    training_max_abs_residual_benchmark_corr: float
    l1_rotation_status: str
    l1_optimizer_success_rate: float


@dataclass(frozen=True)
class FactorAdjustmentResult:
    """Minute panels and diagnostics produced by one factor specification."""

    spec_name: str
    config: FactorConfig
    raw_returns: pd.DataFrame
    benchmark_residuals: pd.DataFrame
    factor_adjusted_residuals: pd.DataFrame
    diagnostics: pd.DataFrame
    benchmark_coefficients: pd.DataFrame
    factor_loadings: pd.DataFrame
    factor_metrics: pd.DataFrame


def _check_input(
    panel: pd.DataFrame,
    stocks: Sequence[str],
    benchmarks: Sequence[str],
) -> pd.DataFrame:
    columns = [*stocks, *benchmarks]
    missing = [column for column in columns if column not in panel.columns]
    if missing:
        raise ValueError(f"Factor panel is missing columns: {missing}")
    value = panel.loc[:, columns].copy()
    validate_panel(value, context="Factor adjustment panel", require_complete=True)
    return value


def _projector(loadings: np.ndarray) -> np.ndarray:
    gram = loadings.T @ loadings
    return loadings @ np.linalg.pinv(gram) @ loadings.T


def fit_factor_window(
    panel: pd.DataFrame,
    stocks: Sequence[str],
    benchmarks: Sequence[str],
    config: FactorConfig,
    *,
    executor: Executor | None = None,
) -> FactorFit:
    """Estimate benchmark residuals, correlation PCA, and L1 rotation in-sample."""

    data = _check_input(panel, stocks, benchmarks)
    stocks = tuple(stocks)
    benchmarks = tuple(benchmarks)
    benchmark_fit = residualize_against_benchmarks(
        data, stocks=stocks, benchmarks=benchmarks
    )
    residuals = benchmark_fit["residual_returns"]
    pca = fit_pca(residuals, method="correlation")
    if config.n_components < 2 or config.n_components > len(stocks):
        raise ValueError("Invalid retained factor count.")
    base_weights = pca.eigenvectors[:, : config.n_components]
    base = np.sqrt(len(stocks)) * base_weights
    pca_projector = base @ base.T / len(stocks)
    l1_rotation_status = "ok"
    try:
        rotation_result = fit_l1_rotation(
            pca,
            n_components=config.n_components,
            n_starts=config.l1_starts,
            random_state=config.random_seed,
            n_jobs=config.n_jobs,
            executor=executor,
        )
        loadings = rotation_result.rotated_loadings.copy()
        rotation = rotation_result.rotation.copy()
        l1_optimizer_success_rate = float(rotation_result.optimizer_success_rate)
    except Exception as error:  # pragma: no cover - injected-failure path
        # The L1 rotation is interpretive.  A non-convex optimizer failure must
        # not alter the causal residual target defined by the PCA subspace.
        l1_rotation_status = f"failed:{type(error).__name__}"
        loadings = pca.loadings.iloc[:, : config.n_components].copy()
        rotation = pd.DataFrame(
            np.eye(config.n_components),
            index=[f"PC{i}" for i in range(1, config.n_components + 1)],
            columns=[f"PC{i}" for i in range(1, config.n_components + 1)],
        )
        l1_optimizer_success_rate = 0.0
    l1_projector = _projector(loadings.to_numpy(dtype=float))
    projector_error = float(np.max(np.abs(pca_projector - l1_projector)))

    standardized = pca.analysis_data.to_numpy(dtype=float)
    scores = standardized @ base_weights
    factor_residuals = standardized - standardized @ pca_projector
    orthogonality_error = float(np.max(np.abs(factor_residuals @ base_weights)))
    factor_removed = 100.0 * (
        1.0 - np.square(factor_residuals).mean() / np.square(standardized).mean()
    )
    benchmark_diagnostics = benchmark_fit["diagnostics"]
    return FactorFit(
        stocks=stocks,
        benchmarks=benchmarks,
        coefficients=benchmark_fit["coefficients"].copy(),
        residual_means=residuals.mean(),
        # Reuse the exact scaling stored by correlation PCA so that the
        # out-of-sample projection is numerically identical to the training
        # representation.
        residual_scales=pca.scales.copy().clip(lower=config.scale_floor),
        pca_explained=pca.explained.copy(),
        pca_eigenvalues=pca.eigenvalues.copy(),
        pca_weights=pd.DataFrame(
            base_weights,
            index=stocks,
            columns=[f"PC{i}" for i in range(1, config.n_components + 1)],
        ),
        factor_projector=pd.DataFrame(
            pca_projector,
            index=stocks,
            columns=stocks,
        ),
        loadings=loadings,
        rotation=rotation,
        rotation_condition_number=float(np.linalg.cond(rotation.to_numpy(dtype=float))),
        projector_invariance_error=projector_error,
        training_orthogonality_error=orthogonality_error,
        training_factor_variance_removed_pct=float(factor_removed),
        training_benchmark_variance_removed_pct=float(
            benchmark_diagnostics["variance_removed_pct"].mean()
        ),
        training_max_abs_residual_benchmark_corr=float(
            benchmark_diagnostics[
                [f"corr_resid_{name}" for name in benchmarks]
            ].abs().to_numpy().max()
        ),
        l1_rotation_status=l1_rotation_status,
        l1_optimizer_success_rate=l1_optimizer_success_rate,
    )


def apply_factor_fit(
    panel: pd.DataFrame,
    fit: FactorFit,
) -> dict[str, pd.DataFrame | float]:
    """Apply frozen training parameters to a new block without refitting."""

    data = _check_input(panel, fit.stocks, fit.benchmarks)
    design = np.column_stack(
        [np.ones(len(data)), data.loc[:, list(fit.benchmarks)].to_numpy(dtype=float)]
    )
    coefficients = fit.coefficients.loc[
        list(fit.stocks), ["alpha", *[f"beta_{name}" for name in fit.benchmarks]]
    ].to_numpy(dtype=float)
    benchmark_residual_values = (
        data.loc[:, list(fit.stocks)].to_numpy(dtype=float) - design @ coefficients.T
    )
    benchmark_residuals = pd.DataFrame(
        benchmark_residual_values, index=data.index, columns=fit.stocks
    )
    standardized = (
        benchmark_residuals.subtract(fit.residual_means, axis="columns")
        .divide(fit.residual_scales, axis="columns")
        .to_numpy(dtype=float)
    )
    if hasattr(fit, "factor_projector"):
        projector = fit.factor_projector.to_numpy(dtype=float)
        factor_basis = fit.pca_weights.to_numpy(dtype=float)
    else:  # pragma: no cover - compatibility with pre-v2 serialized fits
        loadings = fit.loadings.to_numpy(dtype=float)
        projector = _projector(loadings)
        factor_basis = loadings
    scores = standardized @ factor_basis
    factor_residuals_z = standardized - standardized @ projector
    factor_adjusted = pd.DataFrame(
        factor_residuals_z * fit.residual_scales.to_numpy(dtype=float),
        index=data.index,
        columns=fit.stocks,
    )
    return {
        "raw_returns": data.loc[:, list(fit.stocks)].copy(),
        "benchmark_residuals": benchmark_residuals,
        "factor_adjusted_residuals": factor_adjusted,
        "standardized_factor_residuals": pd.DataFrame(
            factor_residuals_z, index=data.index, columns=fit.stocks
        ),
        "factor_scores": pd.DataFrame(
            scores, index=data.index, columns=fit.pca_weights.columns
        ),
        "orthogonality_error": float(np.max(np.abs(factor_residuals_z @ factor_basis))),
    }


def run_factor_adjustment(
    panel: pd.DataFrame,
    stocks: Sequence[str],
    benchmarks: Sequence[str],
    config: FactorConfig,
    *,
    spec_name: str,
) -> FactorAdjustmentResult:
    """Run causal rolling fits and apply each fit only to later sessions."""

    data = _check_input(panel, stocks, benchmarks)
    sessions, codes = session_index(data.index)
    starts = rolling_starts(
        len(sessions), config.window_sessions, config.update_step_sessions
    )
    if not starts:
        raise ValueError("The panel is shorter than the factor window.")

    raw_parts, benchmark_parts, factor_parts = [], [], []
    diagnostic_rows, coefficient_rows, loading_rows, metric_rows = [], [], [], []
    previous_loadings: pd.DataFrame | None = None
    if int(config.n_jobs) < 1:
        raise ValueError("n_jobs must be at least one.")
    executor = (
        ProcessPoolExecutor(max_workers=int(config.n_jobs))
        if int(config.n_jobs) > 1
        else None
    )
    progress_step = max(1, len(starts) // 10)
    for update_number, window_start in enumerate(starts):
        train_start = window_start
        train_end = window_start + config.window_sessions - 1
        origin = train_end + 1
        next_origin = (
            starts[update_number + 1] + config.window_sessions
            if update_number + 1 < len(starts)
            else len(sessions)
        )
        score_end = min(next_origin, len(sessions)) - 1
        train_positions = np.flatnonzero(
            (codes >= train_start) & (codes <= train_end)
        )
        score_positions = np.flatnonzero((codes >= origin) & (codes <= score_end))
        if not len(train_positions) or not len(score_positions):
            continue
        fit = fit_factor_window(
            data.iloc[train_positions],
            stocks,
            benchmarks,
            config,
            executor=executor,
        )
        if previous_loadings is not None:
            aligned, order, signs = align_loading_columns(
                previous_loadings.to_numpy(dtype=float),
                fit.loadings.to_numpy(dtype=float),
            )
            rotation = fit.rotation.to_numpy(dtype=float)[:, order] * signs[None, :]
            fit = replace(
                fit,
                loadings=pd.DataFrame(
                    aligned,
                    index=fit.loadings.index,
                    columns=fit.loadings.columns,
                ),
                rotation=pd.DataFrame(
                    rotation,
                    index=fit.rotation.index,
                    columns=fit.rotation.columns,
                ),
            )
        applied = apply_factor_fit(data.iloc[score_positions], fit)
        update_id = f"{config.window_sessions}_{sessions[train_end].date().isoformat()}"
        raw_parts.append(applied["raw_returns"])
        benchmark_parts.append(applied["benchmark_residuals"])
        factor_parts.append(applied["factor_adjusted_residuals"])
        for factor_number, factor in enumerate(fit.loadings.columns, start=1):
            values = fit.loadings[factor].to_numpy(dtype=float)
            support = set(np.flatnonzero(np.abs(values) > 0.05).tolist())
            if previous_loadings is None:
                cosine = np.nan
                jaccard = np.nan
                turnover = np.nan
            else:
                previous = previous_loadings[factor].to_numpy(dtype=float)
                cosine = float(
                    previous @ values
                    / (np.linalg.norm(previous) * np.linalg.norm(values))
                )
                previous_support = set(
                    np.flatnonzero(np.abs(previous) > 0.05).tolist()
                )
                union = support | previous_support
                jaccard = (
                    float(len(support & previous_support) / len(union))
                    if union
                    else 1.0
                )
                turnover = float(1.0 - cosine)
            metric_rows.append(
                {
                    "spec_name": spec_name,
                    "update_id": update_id,
                    "factor": factor,
                    "factor_number": factor_number,
                    "l1_norm": float(np.abs(values).sum()),
                    "n_small_coefficients": int(np.sum(np.abs(values) <= 0.05)),
                    "small_loading_threshold": 0.05,
                    "cosine_similarity_previous": cosine,
                    "support_jaccard_previous": jaccard,
                    "factor_turnover": turnover,
                    "rotation_condition_number": fit.rotation_condition_number,
                    "projector_invariance_error": fit.projector_invariance_error,
                    "l1_rotation_status": fit.l1_rotation_status,
                    "l1_optimizer_success_rate": fit.l1_optimizer_success_rate,
                    "training_orthogonality_error": fit.training_orthogonality_error,
                    "score_orthogonality_error": applied["orthogonality_error"],
                    "training_start": sessions[train_start].date().isoformat(),
                    "training_end": sessions[train_end].date().isoformat(),
                    "score_start": sessions[origin].date().isoformat(),
                    "score_end": sessions[score_end].date().isoformat(),
                }
            )
        previous_loadings = fit.loadings.copy()
        score_dates = data.index[score_positions]
        diagnostic_rows.append(
            {
                "spec_name": spec_name,
                "update_id": update_id,
                "window_sessions": config.window_sessions,
                "update_step_sessions": config.update_step_sessions,
                "factor_origin": sessions[origin].date().isoformat(),
                "training_start": sessions[train_start].date().isoformat(),
                "training_end": sessions[train_end].date().isoformat(),
                "score_start": sessions[origin].date().isoformat(),
                "score_end": sessions[score_end].date().isoformat(),
                "n_training_rows": len(train_positions),
                "n_score_rows": len(score_positions),
                "pca_pc1_explained_pct": 100.0 * fit.pca_explained[0],
                "pca_first3_explained_pct": 100.0
                * fit.pca_explained[: config.n_components].sum(),
                "mean_benchmark_variance_removed_pct": fit.training_benchmark_variance_removed_pct,
                "max_abs_residual_benchmark_corr": fit.training_max_abs_residual_benchmark_corr,
                "factor_variance_removed_pct": fit.training_factor_variance_removed_pct,
                "rotation_condition_number": fit.rotation_condition_number,
                "projector_invariance_error": fit.projector_invariance_error,
                "l1_rotation_status": fit.l1_rotation_status,
                "l1_optimizer_success_rate": fit.l1_optimizer_success_rate,
                "training_residual_orthogonality_error": fit.training_orthogonality_error,
                "score_residual_orthogonality_error": applied["orthogonality_error"],
                "n_score_sessions": score_end - origin + 1,
                "score_first_timestamp": str(score_dates[0]),
                "score_last_timestamp": str(score_dates[-1]),
            }
        )
        coefficients = fit.coefficients.reset_index(names="stock")
        coefficients.insert(0, "spec_name", spec_name)
        coefficients.insert(1, "update_id", update_id)
        coefficients.insert(2, "window_sessions", config.window_sessions)
        coefficients.insert(3, "training_start", sessions[train_start].date().isoformat())
        coefficients.insert(4, "training_end", sessions[train_end].date().isoformat())
        coefficient_rows.extend(coefficients.to_dict("records"))
        for factor_number, factor in enumerate(fit.loadings.columns, start=1):
            for stock in fit.loadings.index:
                loading_rows.append(
                    {
                        "spec_name": spec_name,
                        "update_id": update_id,
                        "window_sessions": config.window_sessions,
                        "training_start": sessions[train_start].date().isoformat(),
                        "training_end": sessions[train_end].date().isoformat(),
                        "factor": factor,
                        "factor_number": factor_number,
                        "stock": stock,
                        "loading": float(fit.loadings.loc[stock, factor]),
                    }
                )
        if int(config.n_jobs) > 1 and (
            update_number == 0
            or (update_number + 1) % progress_step == 0
            or update_number == len(starts) - 1
        ):
            print(
                f"[factor] {spec_name}: block {update_number + 1}/{len(starts)}",
                flush=True,
            )
    if executor is not None:
        executor.shutdown(wait=True)
    if not factor_parts:
        raise ValueError("No factor-adjusted score block was produced.")
    return FactorAdjustmentResult(
        spec_name=spec_name,
        config=config,
        raw_returns=pd.concat(raw_parts).sort_index(),
        benchmark_residuals=pd.concat(benchmark_parts).sort_index(),
        factor_adjusted_residuals=pd.concat(factor_parts).sort_index(),
        diagnostics=pd.DataFrame(diagnostic_rows),
        benchmark_coefficients=pd.DataFrame(coefficient_rows),
        factor_loadings=pd.DataFrame(loading_rows),
        factor_metrics=pd.DataFrame(metric_rows),
    )
