"""Leakage-controlled network HAR estimation and stability summaries."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from utils.data import validate_panel


HORIZONS = ("D", "W", "M")
HAR_LOOKBACK = {"D": 1, "W": 5, "M": 22}


@dataclass(frozen=True)
class HARConfig:
    """Controls for partialling-out HAR estimation and diagnostics."""

    n_jobs: int = 1
    min_training_observations: int = 252
    rolling_training_observations: int = 252
    tuning_frequency: int = 20
    cv_splits: int = 3
    alpha_fractions: tuple[float, ...] = (0.01, 0.03, 0.10, 0.30, 1.00)
    lasso_max_iter: int = 2_000
    lasso_tolerance: float = 1e-6
    coefficient_tolerance: float = 1e-8
    variance_floor: float = 1e-16
    hac_lag: int = 5
    bootstrap_repetitions: int = 20
    bootstrap_block_lengths: tuple[int, ...] = (5, 20)
    bootstrap_checkpoint_step: int = 60
    random_seed: int = 15015


@dataclass(frozen=True)
class HARDesign:
    """One-step-ahead HAR design indexed by forecast origin."""

    origins: pd.DatetimeIndex
    targets: pd.DatetimeIndex
    response: pd.DataFrame
    features: pd.DataFrame


@dataclass(frozen=True)
class PartialOutResult:
    """Residualized response and cross-stock design from one training sample."""

    response_residual: np.ndarray
    cross_residual: np.ndarray
    own_projection: np.ndarray
    cross_projection: np.ndarray
    eval_response_residual: np.ndarray | None = None
    eval_cross_residual: np.ndarray | None = None


@dataclass(frozen=True)
class HARFit:
    """Fitted HAR equation with cross-stock coefficients on two scales."""

    intercept: float
    own_coefficients: np.ndarray
    network_coefficients_original: np.ndarray
    network_coefficients_standardized: np.ndarray
    network_scales: np.ndarray
    cross_feature_names: tuple[str, ...]
    train_predictions: np.ndarray
    train_residuals: np.ndarray
    smearing_factor: float
    selected_penalty: float
    condition_number: float
    column_norm_ratio: float = np.nan
    lasso_converged: bool = True
    lasso_iterations: int = 0
    lasso_kkt_max_violation: float = 0.0
    smearing_clipped_observations: int = 0


@dataclass(frozen=True)
class PenaltyTuning:
    """Chronological cross-validation result for one target equation."""

    alpha_max: float
    min_fraction: float
    one_se_fraction: float
    min_alpha: float
    one_se_alpha: float
    cv_scores: pd.DataFrame
    status: str = "ok"
    n_valid_folds: int = 0
    requested_folds: int = 0
    fallback_reason: str = ""


@dataclass(frozen=True)
class WalkForwardResult:
    """Forecasts and inspectable model histories from one specification."""

    forecasts: pd.DataFrame
    coefficients: pd.DataFrame
    edge_history: pd.DataFrame
    tuning_history: pd.DataFrame


@dataclass(frozen=True)
class _TargetTask:
    """All immutable inputs needed to fit one target through time."""

    target: str
    y_all: np.ndarray
    own_all: np.ndarray
    cross_all: np.ndarray
    origin_logs: np.ndarray
    feature_names: tuple[str, ...]
    cross_pairs: list[tuple[str, str]]
    origins: pd.DatetimeIndex
    targets: pd.DatetimeIndex
    spec_name: str
    factor_window_sessions: int
    har_window: str | int
    config: HARConfig


@dataclass(frozen=True)
class _TargetWalkResult:
    """Rows emitted by one independent target worker."""

    forecasts: list[dict[str, object]]
    coefficients: list[dict[str, object]]
    edges: list[dict[str, object]]
    tuning: list[dict[str, object]]


@dataclass(frozen=True)
class _DescriptiveTask:
    """All inputs needed to fit one descriptive target across checkpoints."""

    target: str
    y_all: np.ndarray
    own_all: np.ndarray
    cross_all: np.ndarray
    feature_names: tuple[str, ...]
    cross_pairs: list[tuple[str, str]]
    origins: pd.DatetimeIndex
    checkpoints: list[int]
    spec_name: str
    factor_window_sessions: int
    har_window: str | int
    config: HARConfig


@dataclass(frozen=True)
class _DescriptiveResult:
    """Rows emitted by one descriptive target worker."""

    edges: list[dict[str, object]]
    tuning: list[dict[str, object]]


@dataclass(frozen=True)
class _BootstrapTask:
    """All inputs needed to bootstrap one target across checkpoints."""

    target: str
    y_all: np.ndarray
    own_all: np.ndarray
    cross_all: np.ndarray
    feature_names: tuple[str, ...]
    cross_pairs: list[tuple[str, str]]
    origins: pd.DatetimeIndex
    checkpoints: list[int]
    samples: dict[int, dict[int, list[np.ndarray]]]
    spec_name: str
    factor_window_sessions: int
    har_window: str | int
    config: HARConfig


@dataclass(frozen=True)
class LassoSolveResult:
    """Coordinate-descent output with explicit convergence diagnostics."""

    coefficients: np.ndarray
    converged: bool
    iterations: int
    kkt_max_violation: float


def _array(value: np.ndarray | Sequence[float], *, ndim: int = 1) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if ndim == 2 and result.ndim == 1:
        result = result[:, None]
    if result.ndim != ndim:
        raise ValueError(f"Expected an array with {ndim} dimensions.")
    if not np.isfinite(result).all():
        raise ValueError("HAR inputs contain non-finite values.")
    return result


def build_har_features(log_variance: pd.DataFrame) -> HARDesign:
    """Build daily, weekly, and monthly predictors ending at origin d."""

    validate_panel(log_variance, context="Log realized-variance panel", require_complete=True)
    if not isinstance(log_variance.index, pd.DatetimeIndex):
        raise TypeError("HAR features require a DatetimeIndex.")
    if len(log_variance) <= max(HAR_LOOKBACK.values()) + 1:
        raise ValueError("The log variance panel is too short for HAR features.")
    origins: list[pd.Timestamp] = []
    targets: list[pd.Timestamp] = []
    feature_rows: list[list[float]] = []
    response_rows: list[np.ndarray] = []
    columns = [(stock, horizon) for stock in log_variance.columns for horizon in HORIZONS]
    values = log_variance.to_numpy(dtype=float)
    for origin_position in range(max(HAR_LOOKBACK.values()) - 1, len(log_variance) - 1):
        row: list[float] = []
        for stock_position in range(values.shape[1]):
            row.extend(
                [
                    values[origin_position, stock_position],
                    values[origin_position - 4 : origin_position + 1, stock_position].mean(),
                    values[origin_position - 21 : origin_position + 1, stock_position].mean(),
                ]
            )
        origins.append(log_variance.index[origin_position])
        targets.append(log_variance.index[origin_position + 1])
        feature_rows.append(row)
        response_rows.append(values[origin_position + 1])
    features = pd.DataFrame(
        feature_rows,
        index=pd.DatetimeIndex(origins, name="forecast_origin"),
        columns=pd.MultiIndex.from_tuples(columns, names=["stock", "horizon"]),
    )
    response = pd.DataFrame(
        response_rows,
        index=features.index,
        columns=log_variance.columns,
    )
    return HARDesign(
        origins=features.index,
        targets=pd.DatetimeIndex(targets, name="target_date"),
        response=response,
        features=features,
    )


def partial_out(
    y: np.ndarray | Sequence[float],
    own: np.ndarray,
    cross: np.ndarray,
    *,
    eval_y: np.ndarray | Sequence[float] | None = None,
    eval_own: np.ndarray | None = None,
    eval_cross: np.ndarray | None = None,
) -> PartialOutResult:
    """Project y and cross predictors on the intercept and own HAR terms."""

    y_train = _array(y)
    own_train = _array(own, ndim=2)
    cross_train = _array(cross, ndim=2)
    if len(y_train) != len(own_train) or len(y_train) != len(cross_train):
        raise ValueError("Partialling-out arrays have inconsistent lengths.")
    z_train = np.column_stack([np.ones(len(y_train)), own_train])
    own_projection = np.linalg.lstsq(z_train, y_train, rcond=None)[0]
    response_residual = y_train - z_train @ own_projection
    if cross_train.shape[1]:
        cross_projection = np.linalg.lstsq(z_train, cross_train, rcond=None)[0]
        cross_residual = cross_train - z_train @ cross_projection
    else:
        cross_projection = np.empty((z_train.shape[1], 0))
        cross_residual = np.empty((len(y_train), 0))

    eval_response_residual = None
    eval_cross_residual = None
    if eval_y is not None or eval_own is not None or eval_cross is not None:
        if eval_y is None or eval_own is None or eval_cross is None:
            raise ValueError("All evaluation arrays are required together.")
        y_eval = _array(eval_y)
        own_eval = _array(eval_own, ndim=2)
        cross_eval = _array(eval_cross, ndim=2)
        if own_eval.shape[1] != own_train.shape[1] or cross_eval.shape[1] != cross_train.shape[1]:
            raise ValueError("Evaluation design dimensions do not match training dimensions.")
        z_eval = np.column_stack([np.ones(len(y_eval)), own_eval])
        eval_response_residual = y_eval - z_eval @ own_projection
        eval_cross_residual = cross_eval - z_eval @ cross_projection
    return PartialOutResult(
        response_residual=response_residual,
        cross_residual=cross_residual,
        own_projection=own_projection,
        cross_projection=cross_projection,
        eval_response_residual=eval_response_residual,
        eval_cross_residual=eval_cross_residual,
    )


def _alpha_max(y: np.ndarray, own: np.ndarray, cross: np.ndarray) -> float:
    partial, _, _ = _prepare_partialling_out(y, own, cross)
    return _alpha_max_from_partial(partial, len(y))


def _cross_scales(cross_residual: np.ndarray) -> np.ndarray:
    """Return the sample scales used by the standardized lasso design."""

    if not cross_residual.shape[1]:
        return np.empty(0)
    scales = np.std(cross_residual, axis=0, ddof=1)
    return np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)


def _alpha_max_from_partial(
    partial: PartialOutResult,
    n_observations: int,
    scales: np.ndarray | None = None,
) -> float:
    """Compute the zero-solution threshold on the solver's standardized scale."""

    if not partial.cross_residual.shape[1]:
        return 0.0
    scales = _cross_scales(partial.cross_residual) if scales is None else np.asarray(scales, dtype=float)
    standardized = partial.cross_residual / scales
    return float(
        np.max(np.abs(standardized.T @ partial.response_residual)) / n_observations
    )


def _soft_threshold(value: float, penalty: float) -> float:
    return float(np.sign(value) * max(abs(value) - penalty, 0.0))


def _lasso_kkt_violation(
    x: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    alpha: float,
) -> float:
    """Return the maximum KKT violation for the no-intercept lasso."""

    gradient = (x.T @ (x @ coefficients - y)) / len(x)
    nonzero = np.abs(coefficients) > 1e-10
    violations = np.where(
        nonzero,
        np.abs(gradient + alpha * np.sign(coefficients)),
        np.maximum(np.abs(gradient) - alpha, 0.0),
    )
    return float(np.max(violations)) if len(violations) else 0.0


def _fit_lasso_result(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    max_iterations: int,
    tolerance: float,
    initial: np.ndarray | None = None,
    gram: np.ndarray | None = None,
    cross_product: np.ndarray | None = None,
) -> LassoSolveResult:
    """Solve the no-intercept lasso and report convergence/KKT diagnostics."""

    if not x.shape[1]:
        return LassoSolveResult(np.empty(0), True, 0, 0.0)
    gram = (x.T @ x) / len(x) if gram is None else gram
    cross = (x.T @ y) / len(x) if cross_product is None else cross_product
    if initial is None:
        coefficients = np.zeros(x.shape[1])
    else:
        coefficients = _array(initial)
        if coefficients.shape != (x.shape[1],):
            raise ValueError("The lasso warm start has an incompatible shape.")
        coefficients = coefficients.copy()
    score_max = float(np.max(np.abs(cross))) if len(cross) else 0.0
    # At alpha_max the all-zero solution is exact.  The explicit branch also
    # prevents a tiny floating-point residual from appearing as a selected
    # edge at the threshold.
    if alpha >= score_max - 1e-12 * max(1.0, score_max):
        return LassoSolveResult(
            np.zeros(x.shape[1]),
            True,
            0,
            _lasso_kkt_violation(x, y, np.zeros(x.shape[1]), alpha),
        )
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        previous = coefficients.copy()
        for column in range(x.shape[1]):
            partial = cross[column] - gram[column] @ coefficients + gram[column, column] * coefficients[column]
            diagonal = gram[column, column]
            coefficients[column] = _soft_threshold(partial, alpha) / diagonal if diagonal > 1e-14 else 0.0
        change = np.max(np.abs(coefficients - previous))
        kkt = _lasso_kkt_violation(x, y, coefficients, alpha)
        if change <= tolerance and kkt <= max(10.0 * tolerance, 1e-10):
            converged = True
            break
    return LassoSolveResult(
        coefficients=coefficients,
        converged=converged,
        iterations=iterations,
        kkt_max_violation=_lasso_kkt_violation(x, y, coefficients, alpha),
    )


def _fit_lasso(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    max_iterations: int,
    tolerance: float,
    initial: np.ndarray | None = None,
    gram: np.ndarray | None = None,
    cross_product: np.ndarray | None = None,
) -> np.ndarray:
    """Compatibility wrapper returning only lasso coefficients."""

    return _fit_lasso_result(
        x,
        y,
        alpha,
        max_iterations=max_iterations,
        tolerance=tolerance,
        initial=initial,
        gram=gram,
        cross_product=cross_product,
    ).coefficients


def _prepare_partialling_out(
    y_train: np.ndarray,
    own_train: np.ndarray,
    cross_train: np.ndarray,
) -> tuple[PartialOutResult, np.ndarray, np.ndarray]:
    """Compute the projection and standardized cross design once per sample."""

    partial = partial_out(y_train, own_train, cross_train)
    scales = np.ones(cross_train.shape[1])
    if cross_train.shape[1]:
        scales = _cross_scales(partial.cross_residual)
        cross_standardized = partial.cross_residual / scales
    else:
        cross_standardized = np.empty((len(y_train), 0))
    return partial, scales, cross_standardized


def _fit_prepared_partialling_out(
    y_train: np.ndarray,
    own_train: np.ndarray,
    cross_train: np.ndarray,
    partial: PartialOutResult,
    scales: np.ndarray,
    cross_standardized: np.ndarray,
    *,
    alpha: float,
    cross_feature_names: Sequence[str],
    config: HARConfig,
    initial: np.ndarray | None = None,
    lasso_gram: np.ndarray | None = None,
    lasso_cross_product: np.ndarray | None = None,
) -> HARFit:
    """Fit one penalty value after the projection has already been computed."""

    lasso_converged = True
    lasso_iterations = 0
    lasso_kkt_max_violation = 0.0
    if cross_train.shape[1]:
        if alpha <= 1e-14:
            network_standardized = np.linalg.lstsq(
                cross_standardized,
                partial.response_residual,
                rcond=None,
            )[0]
            lasso_kkt_max_violation = _lasso_kkt_violation(
                cross_standardized,
                partial.response_residual,
                network_standardized,
                0.0,
            )
        else:
            lasso_result = _fit_lasso_result(
                cross_standardized,
                partial.response_residual,
                float(alpha),
                max_iterations=config.lasso_max_iter,
                tolerance=config.lasso_tolerance,
                initial=initial,
                gram=lasso_gram,
                cross_product=lasso_cross_product,
            )
            network_standardized = lasso_result.coefficients
            lasso_converged = lasso_result.converged
            lasso_iterations = lasso_result.iterations
            lasso_kkt_max_violation = lasso_result.kkt_max_violation
        network_original = network_standardized / scales
    else:
        network_standardized = np.empty(0)
        network_original = np.empty(0)
    z_train = np.column_stack([np.ones(len(y_train)), own_train])
    own_conditional = np.linalg.lstsq(
        z_train,
        y_train - cross_train @ network_original,
        rcond=None,
    )[0]
    predictions = z_train @ own_conditional + cross_train @ network_original
    residuals = y_train - predictions
    clipped_residuals = np.clip(residuals, -30.0, 30.0)
    smearing = float(np.mean(np.exp(clipped_residuals)))
    design = np.column_stack([z_train, cross_train / scales])
    column_norms = np.linalg.norm(design, axis=0)
    column_norm_ratio = float(column_norms.max() / max(column_norms.min(), 1e-12))
    condition_number = float(np.linalg.cond(design))
    return HARFit(
        intercept=float(own_conditional[0]),
        own_coefficients=own_conditional[1:],
        network_coefficients_original=network_original,
        network_coefficients_standardized=network_standardized,
        network_scales=scales,
        cross_feature_names=tuple(cross_feature_names),
        train_predictions=predictions,
        train_residuals=residuals,
        smearing_factor=max(smearing, 1e-12),
        selected_penalty=float(alpha),
        condition_number=condition_number,
        column_norm_ratio=column_norm_ratio,
        lasso_converged=lasso_converged,
        lasso_iterations=lasso_iterations,
        lasso_kkt_max_violation=lasso_kkt_max_violation,
        smearing_clipped_observations=int(np.sum(clipped_residuals != residuals)),
    )


def fit_partialling_out(
    y: np.ndarray | Sequence[float],
    own: np.ndarray,
    cross: np.ndarray,
    *,
    alpha: float = 0.0,
    cross_feature_names: Sequence[str] = (),
    config: HARConfig = HARConfig(),
) -> HARFit:
    """Fit lasso only on cross-stock terms and recover conditional own HAR terms."""

    y_train = _array(y)
    own_train = _array(own, ndim=2)
    cross_train = _array(cross, ndim=2)
    if len(y_train) < 3 or len(y_train) != len(own_train) or len(y_train) != len(cross_train):
        raise ValueError("Invalid partialling-out training sample.")
    if cross_train.shape[1] != len(tuple(cross_feature_names)):
        raise ValueError("cross_feature_names does not match the design.")
    partial, scales, cross_standardized = _prepare_partialling_out(
        y_train, own_train, cross_train
    )
    lasso_gram = (
        (cross_standardized.T @ cross_standardized) / len(y_train)
        if cross_standardized.shape[1]
        else None
    )
    lasso_cross_product = (
        (cross_standardized.T @ partial.response_residual) / len(y_train)
        if cross_standardized.shape[1]
        else None
    )
    return _fit_prepared_partialling_out(
        y_train,
        own_train,
        cross_train,
        partial,
        scales,
        cross_standardized,
        alpha=alpha,
        cross_feature_names=cross_feature_names,
        config=config,
        lasso_gram=lasso_gram,
        lasso_cross_product=lasso_cross_product,
    )


def fit_network_pair(
    y: np.ndarray | Sequence[float],
    own: np.ndarray,
    cross: np.ndarray,
    *,
    one_se_alpha: float,
    min_alpha: float,
    cross_feature_names: Sequence[str] = (),
    config: HARConfig = HARConfig(),
) -> tuple[HARFit, HARFit, HARFit]:
    """Fit the two network penalties and own HAR while reusing one projection."""

    y_train = _array(y)
    own_train = _array(own, ndim=2)
    cross_train = _array(cross, ndim=2)
    names = tuple(cross_feature_names)
    if len(y_train) < 3 or len(y_train) != len(own_train) or len(y_train) != len(cross_train):
        raise ValueError("Invalid partialling-out training sample.")
    if cross_train.shape[1] != len(names):
        raise ValueError("cross_feature_names does not match the design.")
    partial, scales, cross_standardized = _prepare_partialling_out(
        y_train, own_train, cross_train
    )
    lasso_gram = (
        (cross_standardized.T @ cross_standardized) / len(y_train)
        if cross_standardized.shape[1]
        else None
    )
    lasso_cross_product = (
        (cross_standardized.T @ partial.response_residual) / len(y_train)
        if cross_standardized.shape[1]
        else None
    )
    requested = {"one_se": float(one_se_alpha), "min_loss": float(min_alpha)}
    fitted: dict[str, HARFit] = {}
    warm_start: np.ndarray | None = None
    for label, alpha in sorted(requested.items(), key=lambda item: item[1], reverse=True):
        fitted[label] = _fit_prepared_partialling_out(
            y_train,
            own_train,
            cross_train,
            partial,
            scales,
            cross_standardized,
            alpha=alpha,
            cross_feature_names=names,
            config=config,
            initial=warm_start,
            lasso_gram=lasso_gram,
            lasso_cross_product=lasso_cross_product,
        )
        warm_start = fitted[label].network_coefficients_standardized
    fit_own = fit_own_har(y_train, own_train, config=config)
    return fitted["one_se"], fitted["min_loss"], fit_own


def _predict_fit(fit: HARFit, own: np.ndarray, cross: np.ndarray) -> np.ndarray:
    own_values = _array(own, ndim=2)
    cross_values = _array(cross, ndim=2)
    if own_values.shape[1] != len(fit.own_coefficients) or cross_values.shape[1] != len(
        fit.network_coefficients_original
    ):
        raise ValueError("Prediction design dimensions do not match the fitted model.")
    return (
        fit.intercept
        + own_values @ fit.own_coefficients
        + cross_values @ fit.network_coefficients_original
    )


def _chronological_splits(n: int, n_splits: int, min_train: int) -> list[tuple[int, int]]:
    if n_splits <= 0:
        return []
    min_train = min(max(5, min_train), max(1, n - 1))
    endpoints = np.linspace(min_train, n, n_splits + 1, dtype=int)
    splits = []
    for index in range(n_splits):
        train_end = int(endpoints[index])
        validation_end = int(endpoints[index + 1])
        if validation_end > train_end:
            splits.append((train_end, validation_end))
    return splits


def tune_network_penalty(
    y: np.ndarray,
    own: np.ndarray,
    cross: np.ndarray,
    *,
    config: HARConfig = HARConfig(),
    cross_feature_names: Sequence[str] = (),
) -> PenaltyTuning:
    """Select lasso strength by expanding chronological validation and 1-SE."""

    y_values = _array(y)
    own_values = _array(own, ndim=2)
    cross_values = _array(cross, ndim=2)
    if not config.alpha_fractions:
        raise ValueError("alpha_fractions cannot be empty.")
    fractions = tuple(sorted(float(value) for value in config.alpha_fractions if value >= 0))
    if not fractions:
        raise ValueError("alpha_fractions must contain a non-negative value.")
    splits = _chronological_splits(
        len(y_values), config.cv_splits, max(20, own_values.shape[1] + 5)
    )
    prepared_splits = []
    for train_end, validation_end in splits:
        y_train = y_values[:train_end]
        own_train = own_values[:train_end]
        cross_train = cross_values[:train_end]
        partial, scales, cross_standardized = _prepare_partialling_out(
            y_train, own_train, cross_train
        )
        lasso_gram = (
            (cross_standardized.T @ cross_standardized) / train_end
            if cross_standardized.shape[1]
            else None
        )
        lasso_cross_product = (
            (cross_standardized.T @ partial.response_residual) / train_end
            if cross_standardized.shape[1]
            else None
        )
        prepared_splits.append(
            (
                train_end,
                validation_end,
                y_train,
                own_train,
                cross_train,
                partial,
                scales,
                cross_standardized,
                lasso_gram,
                lasso_cross_product,
                _alpha_max_from_partial(partial, train_end, scales),
            )
        )
    rows: list[dict[str, float | int]] = []
    fold_results = {fraction: ([], []) for fraction in fractions}
    for (
        train_end,
        validation_end,
        y_train,
        own_train,
        cross_train,
        partial,
        scales,
        cross_standardized,
        lasso_gram,
        lasso_cross_product,
        alpha_max,
    ) in prepared_splits:
        warm_start = None
        for fraction in reversed(fractions):
            alpha = float(fraction * alpha_max)
            fit = _fit_prepared_partialling_out(
                y_train,
                own_train,
                cross_train,
                partial,
                scales,
                cross_standardized,
                alpha=alpha,
                cross_feature_names=cross_feature_names,
                config=config,
                initial=warm_start,
                lasso_gram=lasso_gram,
                lasso_cross_product=lasso_cross_product,
            )
            predicted = _predict_fit(
                fit,
                own_values[train_end:validation_end],
                cross_values[train_end:validation_end],
            )
            fold_results[fraction][0].append(
                float(np.mean(np.square(y_values[train_end:validation_end] - predicted)))
            )
            fold_results[fraction][1].append(alpha)
            warm_start = fit.network_coefficients_standardized
    for fraction in fractions:
        fold_losses, fold_alphas = fold_results[fraction]
        if fold_losses:
            mean_loss = float(np.mean(fold_losses))
            sd_loss = float(np.std(fold_losses, ddof=1)) if len(fold_losses) > 1 else 0.0
            standard_error = sd_loss / np.sqrt(len(fold_losses))
        else:
            mean_loss = np.nan
            sd_loss = np.nan
            standard_error = np.nan
        rows.append(
            {
                "fraction": fraction,
                "mean_validation_log_mse": mean_loss,
                "sd_validation_log_mse": sd_loss,
                "se_validation_log_mse": standard_error,
                "n_folds": len(fold_losses),
                "mean_fold_alpha": float(np.mean(fold_alphas)) if fold_alphas else np.nan,
            }
        )
    cv_scores = pd.DataFrame(rows)
    full_partial, full_scales, _ = _prepare_partialling_out(
        y_values, own_values, cross_values
    )
    full_alpha_max = _alpha_max_from_partial(
        full_partial, len(y_values), full_scales
    )
    finite = cv_scores.dropna(subset=["mean_validation_log_mse"])
    requested_folds = max(int(config.cv_splits), 0)
    n_valid_folds = len(prepared_splits)
    if n_valid_folds < requested_folds:
        status = "fallback_insufficient_folds"
        fallback_reason = (
            f"requested {requested_folds} chronological folds, "
            f"constructed {n_valid_folds}"
        )
    else:
        status = "ok"
        fallback_reason = ""
    if finite.empty:
        # An empty/insufficient CV sample must not silently choose the weakest
        # penalty.  Use the strongest candidate and expose the fallback in the
        # tuning ledger.
        min_fraction = one_se_fraction = fractions[-1]
        status = "fallback_no_valid_folds"
        fallback_reason = "no finite chronological validation losses"
    else:
        min_loss = float(finite["mean_validation_log_mse"].min())
        min_fraction = float(finite.loc[finite["mean_validation_log_mse"].idxmin(), "fraction"])
        min_se = float(
            finite.loc[finite["mean_validation_log_mse"].idxmin(), "se_validation_log_mse"]
        )
        if not np.isfinite(min_se):
            min_se = 0.0
        eligible = finite[finite["mean_validation_log_mse"] <= min_loss + min_se]
        one_se_fraction = float(eligible["fraction"].max()) if not eligible.empty else min_fraction
    return PenaltyTuning(
        alpha_max=full_alpha_max,
        min_fraction=min_fraction,
        one_se_fraction=one_se_fraction,
        min_alpha=float(min_fraction * full_alpha_max),
        one_se_alpha=float(one_se_fraction * full_alpha_max),
        cv_scores=cv_scores,
        status=status,
        n_valid_folds=n_valid_folds,
        requested_folds=requested_folds,
        fallback_reason=fallback_reason,
    )


def fit_own_har(y: np.ndarray, own: np.ndarray, *, config: HARConfig = HARConfig()) -> HARFit:
    """Fit the unpenalized own-stock HAR benchmark."""

    y_train = _array(y)
    own_train = _array(own, ndim=2)
    if len(y_train) < 3 or len(y_train) != len(own_train):
        raise ValueError("Invalid own-stock HAR training sample.")
    z_train = np.column_stack([np.ones(len(y_train)), own_train])
    coefficients = np.linalg.lstsq(z_train, y_train, rcond=None)[0]
    predictions = z_train @ coefficients
    residuals = y_train - predictions
    clipped_residuals = np.clip(residuals, -30.0, 30.0)
    smearing = float(np.mean(np.exp(clipped_residuals)))
    column_norms = np.linalg.norm(z_train, axis=0)
    column_norm_ratio = float(column_norms.max() / max(column_norms.min(), 1e-12))
    condition_number = float(np.linalg.cond(z_train))
    return HARFit(
        intercept=float(coefficients[0]),
        own_coefficients=coefficients[1:],
        network_coefficients_original=np.empty(0),
        network_coefficients_standardized=np.empty(0),
        network_scales=np.empty(0),
        cross_feature_names=(),
        train_predictions=predictions,
        train_residuals=residuals,
        smearing_factor=max(smearing, 1e-12),
        selected_penalty=0.0,
        condition_number=condition_number,
        column_norm_ratio=column_norm_ratio,
        lasso_converged=True,
        lasso_iterations=0,
        lasso_kkt_max_violation=0.0,
        smearing_clipped_observations=int(np.sum(clipped_residuals != residuals)),
    )


def _walk_forward_target_task(task: _TargetTask) -> _TargetWalkResult:
    """Fit one target over the entire walk-forward history in one worker."""

    config = task.config
    if int(config.n_jobs) < 1:
        raise ValueError("n_jobs must be at least one.")
    forecast_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    tuning: PenaltyTuning | None = None
    last_tuned = -10**9
    last_row = len(task.origins) - 1
    first_row = max(config.min_training_observations, 1)
    for row_number in range(first_row, len(task.origins)):
        if task.har_window == "expanding":
            train_start = 0
        else:
            train_length = int(task.har_window)
            if train_length <= 0:
                raise ValueError("har_window must be positive or 'expanding'.")
            train_start = max(0, row_number - train_length)
        train_end = row_number
        origin_date = task.origins[row_number]
        target_date = task.targets[row_number]
        y_train = task.y_all[train_start:train_end]
        own_train = task.own_all[train_start:train_end]
        cross_train = task.cross_all[train_start:train_end]
        needs_tuning = tuning is None or row_number - last_tuned >= config.tuning_frequency
        if needs_tuning:
            tuning = tune_network_penalty(
                y_train,
                own_train,
                cross_train,
                config=config,
                cross_feature_names=task.feature_names,
            )
            last_tuned = row_number
            cv = tuning.cv_scores.copy()
            cv.insert(0, "stock", task.target)
            cv.insert(0, "spec_name", task.spec_name)
            cv.insert(2, "forecast_origin", origin_date)
            cv["factor_window_sessions"] = task.factor_window_sessions
            cv["har_window"] = task.har_window
            cv["training_start"] = task.origins[train_start]
            cv["training_end"] = task.origins[train_end - 1]
            cv["selected_one_se_fraction"] = tuning.one_se_fraction
            cv["selected_min_loss_fraction"] = tuning.min_fraction
            tuning_rows.extend(cv.to_dict("records"))
        if tuning is None:
            raise RuntimeError("Penalty tuning did not produce a result.")
        fit_one_se, fit_min, fit_own = fit_network_pair(
            y_train,
            own_train,
            cross_train,
            one_se_alpha=tuning.one_se_alpha,
            min_alpha=tuning.min_alpha,
            cross_feature_names=task.feature_names,
            config=config,
        )
        own_current = task.own_all[row_number : row_number + 1]
        cross_current = task.cross_all[row_number : row_number + 1]
        actual_log = float(task.y_all[row_number])
        origin_log = float(task.origin_logs[row_number])
        actual_variance = max(
            float(np.exp(np.clip(actual_log, -40.0, 40.0))),
            config.variance_floor,
        )
        for model_name, fit in (
            ("own_har", fit_own),
            ("network_har_l1_1se", fit_one_se),
            ("network_har_l1_min_loss", fit_min),
        ):
            prediction_cross = (
                cross_current
                if len(fit.network_coefficients_original)
                else np.empty((1, 0))
            )
            predicted_log = float(_predict_fit(fit, own_current, prediction_cross)[0])
            predicted_variance = max(
                float(np.exp(np.clip(predicted_log, -40.0, 40.0)) * fit.smearing_factor),
                config.variance_floor,
            )
            forecast_rows.append(
                {
                    "spec_name": task.spec_name,
                    "factor_window_sessions": task.factor_window_sessions,
                    "har_window": task.har_window,
                    "forecast_origin": origin_date,
                    "target_date": target_date,
                    "stock": task.target,
                    "model": model_name,
                    "actual_log_variance": actual_log,
                    "predicted_log_variance": predicted_log,
                    "actual_variance": actual_variance,
                    "predicted_variance": predicted_variance,
                    "qlike": _qlike(actual_variance, predicted_variance, config.variance_floor),
                    "log_mse": float((actual_log - predicted_log) ** 2),
                    "log_mae": float(abs(actual_log - predicted_log)),
                    "directional_hit": _directional_hit(actual_log, predicted_log, origin_log),
                    "selected_penalty": fit.selected_penalty,
                    "smearing_factor": fit.smearing_factor,
                    "design_condition_number": fit.condition_number,
                    "n_train_observations": len(y_train),
                    "training_start": task.origins[train_start],
                    "training_end": task.origins[train_end - 1],
                }
            )
        persistence_variance = max(
            float(np.exp(np.clip(origin_log, -40.0, 40.0))),
            config.variance_floor,
        )
        forecast_rows.append(
            {
                "spec_name": task.spec_name,
                "factor_window_sessions": task.factor_window_sessions,
                "har_window": task.har_window,
                "forecast_origin": origin_date,
                "target_date": target_date,
                "stock": task.target,
                "model": "persistence",
                "actual_log_variance": actual_log,
                "predicted_log_variance": origin_log,
                "actual_variance": actual_variance,
                "predicted_variance": persistence_variance,
                "qlike": _qlike(actual_variance, persistence_variance, config.variance_floor),
                "log_mse": float((actual_log - origin_log) ** 2),
                "log_mae": float(abs(actual_log - origin_log)),
                "directional_hit": np.nan,
                "selected_penalty": np.nan,
                "smearing_factor": 1.0,
                "design_condition_number": np.nan,
                "n_train_observations": len(y_train),
                "training_start": task.origins[train_start],
                "training_end": task.origins[train_end - 1],
            }
        )
        for model_name, fit in (
            ("network_har_l1_1se", fit_one_se),
            ("network_har_l1_min_loss", fit_min),
        ):
            for coefficient, standardized, (source, horizon) in zip(
                fit.network_coefficients_original,
                fit.network_coefficients_standardized,
                task.cross_pairs,
            ):
                edge_rows.append(
                    {
                        "spec_name": task.spec_name,
                        "factor_window_sessions": task.factor_window_sessions,
                        "har_window": task.har_window,
                        "forecast_origin": origin_date,
                        "training_start": task.origins[train_start],
                        "training_end": task.origins[train_end - 1],
                        "source": source,
                        "target": task.target,
                        "edge_id": f"{source}->{task.target}",
                        "model": model_name,
                        "horizon": horizon,
                        "coefficient_original": float(coefficient),
                        "coefficient_standardized": float(standardized),
                        "selected": bool(abs(coefficient) > config.coefficient_tolerance),
                        "sign": int(np.sign(coefficient)),
                        "n_train_observations": len(y_train),
                    }
                )
        if needs_tuning or row_number == last_row:
            for model_name, fit in (
                ("own_har", fit_own),
                ("network_har_l1_1se", fit_one_se),
                ("network_har_l1_min_loss", fit_min),
            ):
                coefficient_rows.append(
                    {
                        "spec_name": task.spec_name,
                        "factor_window_sessions": task.factor_window_sessions,
                        "har_window": task.har_window,
                        "forecast_origin": origin_date,
                        "training_start": task.origins[train_start],
                        "training_end": task.origins[train_end - 1],
                        "stock": task.target,
                        "model": model_name,
                        "predictor_stock": task.target,
                        "horizon": "Intercept",
                        "coefficient_original": fit.intercept,
                        "coefficient_standardized": np.nan,
                        "selected": True,
                        "selected_penalty": fit.selected_penalty,
                    }
                )
                for horizon, coefficient in zip(HORIZONS, fit.own_coefficients):
                    coefficient_rows.append(
                        {
                            "spec_name": task.spec_name,
                            "factor_window_sessions": task.factor_window_sessions,
                            "har_window": task.har_window,
                            "forecast_origin": origin_date,
                            "training_start": task.origins[train_start],
                            "training_end": task.origins[train_end - 1],
                            "stock": task.target,
                            "model": model_name,
                            "predictor_stock": task.target,
                            "horizon": horizon,
                            "coefficient_original": float(coefficient),
                            "coefficient_standardized": np.nan,
                            "selected": True,
                            "selected_penalty": fit.selected_penalty,
                        }
                    )
    return _TargetWalkResult(
        forecasts=forecast_rows,
        coefficients=coefficient_rows,
        edges=edge_rows,
        tuning=tuning_rows,
    )


def _descriptive_target_task(task: _DescriptiveTask) -> _DescriptiveResult:
    """Fit one descriptive target over every requested checkpoint."""

    rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    for row_number in task.checkpoints:
        train_start = (
            0
            if task.har_window == "expanding"
            else max(0, row_number - int(task.har_window))
        )
        y = task.y_all[train_start:row_number]
        own = task.own_all[train_start:row_number]
        cross = task.cross_all[train_start:row_number]
        tuning = tune_network_penalty(
            y,
            own,
            cross,
            config=task.config,
            cross_feature_names=task.feature_names,
        )
        fit = fit_partialling_out(
            y,
            own,
            cross,
            alpha=tuning.one_se_alpha,
            cross_feature_names=task.feature_names,
            config=task.config,
        )
        for fraction_row in tuning.cv_scores.to_dict("records"):
            tuning_rows.append(
                {
                    **fraction_row,
                    "spec_name": task.spec_name,
                    "factor_window_sessions": task.factor_window_sessions,
                    "har_window": task.har_window,
                    "checkpoint_date": task.origins[row_number - 1],
                    "training_start": task.origins[train_start],
                    "training_end": task.origins[row_number - 1],
                    "stock": task.target,
                    "selected_one_se_fraction": tuning.one_se_fraction,
                    "selected_min_loss_fraction": tuning.min_fraction,
                }
            )
        for coefficient, standardized, (source, horizon) in zip(
            fit.network_coefficients_original,
            fit.network_coefficients_standardized,
            task.cross_pairs,
        ):
            rows.append(
                {
                    "spec_name": task.spec_name,
                    "factor_window_sessions": task.factor_window_sessions,
                    "har_window": task.har_window,
                    "forecast_origin": task.origins[row_number - 1],
                    "training_start": task.origins[train_start],
                    "training_end": task.origins[row_number - 1],
                    "source": source,
                    "target": task.target,
                    "edge_id": f"{source}->{task.target}",
                    "model": "descriptive_network",
                    "horizon": horizon,
                    "coefficient_original": float(coefficient),
                    "coefficient_standardized": float(standardized),
                    "selected": bool(abs(coefficient) > task.config.coefficient_tolerance),
                    "sign": int(np.sign(coefficient)),
                    "n_train_observations": len(y),
                }
            )
    return _DescriptiveResult(edges=rows, tuning=tuning_rows)


def _feature_matrix(
    design: HARDesign,
    target: str,
    stocks: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], list[tuple[str, str]]]:
    own = design.features.loc[:, [(target, horizon) for horizon in HORIZONS]].to_numpy(dtype=float)
    cross_pairs = [
        (source, horizon)
        for source in stocks
        if source != target
        for horizon in HORIZONS
    ]
    cross = design.features.loc[:, cross_pairs].to_numpy(dtype=float)
    names = tuple(f"{source}:{horizon}" for source, horizon in cross_pairs)
    return own, cross, names, cross_pairs


def _qlike(actual: float, predicted: float, floor: float) -> float:
    actual_value = max(float(actual), floor)
    predicted_value = max(float(predicted), floor)
    ratio = actual_value / predicted_value
    return float(ratio - np.log(ratio) - 1.0)


def _directional_hit(actual_log: float, predicted_log: float, origin_log: float) -> float:
    actual_change = actual_log - origin_log
    predicted_change = predicted_log - origin_log
    if actual_change == 0.0 or predicted_change == 0.0:
        return np.nan
    return float(np.sign(actual_change) == np.sign(predicted_change))


def _legacy_walk_forward_forecasts(
    log_variance: pd.DataFrame,
    *,
    spec_name: str,
    factor_window_sessions: int,
    har_window: str | int = "expanding",
    config: HARConfig = HARConfig(),
) -> WalkForwardResult:
    """Estimate one-step forecasts without using the current target observation."""

    design = build_har_features(log_variance)
    stocks = list(log_variance.columns)
    forecast_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    cached_tuning: dict[str, PenaltyTuning] = {}
    last_tuned: dict[str, int] = {}
    target_designs = {
        target: _feature_matrix(design, target, stocks) for target in stocks
    }
    response_arrays = {
        target: design.response[target].to_numpy(dtype=float) for target in stocks
    }
    last_row = len(design.origins) - 1
    first_row = max(config.min_training_observations, 1)
    if int(config.n_jobs) < 1:
        raise ValueError("n_jobs must be at least one.")
    executor = (
        ProcessPoolExecutor(max_workers=int(config.n_jobs))
        if int(config.n_jobs) > 1
        else None
    )

    def map_tasks(function, tasks):
        if executor is None:
            return [function(task) for task in tasks]
        return list(executor.map(function, tasks))

    try:
        for row_number in range(first_row, len(design.origins)):
            if har_window == "expanding":
                train_start = 0
            else:
                train_length = int(har_window)
                if train_length <= 0:
                    raise ValueError("har_window must be positive or 'expanding'.")
                train_start = max(0, row_number - train_length)
            train_end = row_number
            origin_date = design.origins[row_number]
            target_date = design.targets[row_number]
            target_windows: list[_TargetWindow] = []
            tuning_targets: list[str] = []
            tuning_tasks = []
            for target in stocks:
                own_all, cross_all, feature_names, cross_pairs = target_designs[target]
                y_all = response_arrays[target]
                own_train = own_all[train_start:train_end]
                cross_train = cross_all[train_start:train_end]
                y_train = y_all[train_start:train_end]
                needs_tuning = (
                    target not in cached_tuning
                    or row_number - last_tuned[target] >= config.tuning_frequency
                )
                target_windows.append(
                    _TargetWindow(
                        target=target,
                        y_all=y_all,
                        own_all=own_all,
                        cross_all=cross_all,
                        y_train=y_train,
                        own_train=own_train,
                        cross_train=cross_train,
                        feature_names=feature_names,
                        cross_pairs=cross_pairs,
                        needs_tuning=needs_tuning,
                    )
                )
                if needs_tuning:
                    tuning_targets.append(target)
                    tuning_tasks.append((y_train, own_train, cross_train, feature_names, config))

            for target, tuning in zip(tuning_targets, map_tasks(_tune_target_task, tuning_tasks)):
                cached_tuning[target] = tuning
                last_tuned[target] = row_number
                cv = tuning.cv_scores.copy()
                cv.insert(0, "stock", target)
                cv.insert(0, "spec_name", spec_name)
                cv.insert(2, "forecast_origin", origin_date)
                cv["factor_window_sessions"] = factor_window_sessions
                cv["har_window"] = har_window
                cv["training_start"] = design.origins[train_start]
                cv["training_end"] = design.origins[train_end - 1]
                cv["selected_one_se_fraction"] = tuning.one_se_fraction
                cv["selected_min_loss_fraction"] = tuning.min_fraction
                tuning_rows.extend(cv.to_dict("records"))

            fit_tasks = [
                (
                    item.y_train,
                    item.own_train,
                    item.cross_train,
                    item.feature_names,
                    cached_tuning[item.target],
                    config,
                )
                for item in target_windows
            ]
            fit_results = map_tasks(_fit_target_task, fit_tasks)
            for item, (fit_one_se, fit_min, fit_own) in zip(target_windows, fit_results):
                target = item.target
                own_current = item.own_all[row_number : row_number + 1]
                cross_current = item.cross_all[row_number : row_number + 1]
                actual_log = float(item.y_all[row_number])
                origin_log = float(log_variance.loc[origin_date, target])
                actual_variance = max(
                    float(np.exp(np.clip(actual_log, -40.0, 40.0))),
                    config.variance_floor,
                )
                model_fits = (
                    ("own_har", fit_own),
                    ("network_har_l1_1se", fit_one_se),
                    ("network_har_l1_min_loss", fit_min),
                )
                for model_name, fit in model_fits:
                    prediction_cross = (
                        cross_current
                        if len(fit.network_coefficients_original)
                        else np.empty((1, 0))
                    )
                    predicted_log = float(_predict_fit(fit, own_current, prediction_cross)[0])
                    predicted_variance = max(
                        float(np.exp(np.clip(predicted_log, -40.0, 40.0)) * fit.smearing_factor),
                        config.variance_floor,
                    )
                    forecast_rows.append(
                        {
                            "spec_name": spec_name,
                            "factor_window_sessions": factor_window_sessions,
                            "har_window": har_window,
                            "forecast_origin": origin_date,
                            "target_date": target_date,
                            "stock": target,
                            "model": model_name,
                            "actual_log_variance": actual_log,
                            "predicted_log_variance": predicted_log,
                            "actual_variance": actual_variance,
                            "predicted_variance": predicted_variance,
                            "qlike": _qlike(actual_variance, predicted_variance, config.variance_floor),
                            "log_mse": float((actual_log - predicted_log) ** 2),
                            "log_mae": float(abs(actual_log - predicted_log)),
                            "directional_hit": _directional_hit(actual_log, predicted_log, origin_log),
                            "selected_penalty": fit.selected_penalty,
                            "smearing_factor": fit.smearing_factor,
                            "design_condition_number": fit.condition_number,
                            "n_train_observations": len(item.y_train),
                            "training_start": design.origins[train_start],
                            "training_end": design.origins[train_end - 1],
                        }
                    )
                persistence_variance = max(
                    float(np.exp(np.clip(origin_log, -40.0, 40.0))),
                    config.variance_floor,
                )
                forecast_rows.append(
                    {
                        "spec_name": spec_name,
                        "factor_window_sessions": factor_window_sessions,
                        "har_window": har_window,
                        "forecast_origin": origin_date,
                        "target_date": target_date,
                        "stock": target,
                        "model": "persistence",
                        "actual_log_variance": actual_log,
                        "predicted_log_variance": origin_log,
                        "actual_variance": actual_variance,
                        "predicted_variance": persistence_variance,
                        "qlike": _qlike(actual_variance, persistence_variance, config.variance_floor),
                        "log_mse": float((actual_log - origin_log) ** 2),
                        "log_mae": float(abs(actual_log - origin_log)),
                        "directional_hit": np.nan,
                        "selected_penalty": np.nan,
                        "smearing_factor": 1.0,
                        "design_condition_number": np.nan,
                        "n_train_observations": len(item.y_train),
                        "training_start": design.origins[train_start],
                        "training_end": design.origins[train_end - 1],
                    }
                )
                for model_name, fit in (
                    ("network_har_l1_1se", fit_one_se),
                    ("network_har_l1_min_loss", fit_min),
                ):
                    for coefficient, standardized, (source, horizon) in zip(
                        fit.network_coefficients_original,
                        fit.network_coefficients_standardized,
                        item.cross_pairs,
                    ):
                        edge_rows.append(
                            {
                                "spec_name": spec_name,
                                "factor_window_sessions": factor_window_sessions,
                                "har_window": har_window,
                                "forecast_origin": origin_date,
                                "training_start": design.origins[train_start],
                                "training_end": design.origins[train_end - 1],
                                "source": source,
                                "target": target,
                                "edge_id": f"{source}->{target}",
                                "model": model_name,
                                "horizon": horizon,
                                "coefficient_original": float(coefficient),
                                "coefficient_standardized": float(standardized),
                                "selected": bool(abs(coefficient) > config.coefficient_tolerance),
                                "sign": int(np.sign(coefficient)),
                                "n_train_observations": len(item.y_train),
                            }
                        )
                if item.needs_tuning or row_number == last_row:
                    for model_name, fit in (
                        ("own_har", fit_own),
                        ("network_har_l1_1se", fit_one_se),
                        ("network_har_l1_min_loss", fit_min),
                    ):
                        coefficient_rows.append(
                            {
                                "spec_name": spec_name,
                                "factor_window_sessions": factor_window_sessions,
                                "har_window": har_window,
                                "forecast_origin": origin_date,
                                "training_start": design.origins[train_start],
                                "training_end": design.origins[train_end - 1],
                                "stock": target,
                                "model": model_name,
                                "predictor_stock": target,
                                "horizon": "Intercept",
                                "coefficient_original": fit.intercept,
                                "coefficient_standardized": np.nan,
                                "selected": True,
                                "selected_penalty": fit.selected_penalty,
                            }
                        )
                        for horizon, coefficient in zip(HORIZONS, fit.own_coefficients):
                            coefficient_rows.append(
                                {
                                    "spec_name": spec_name,
                                    "factor_window_sessions": factor_window_sessions,
                                    "har_window": har_window,
                                    "forecast_origin": origin_date,
                                    "training_start": design.origins[train_start],
                                    "training_end": design.origins[train_end - 1],
                                    "stock": target,
                                    "model": model_name,
                                    "predictor_stock": target,
                                    "horizon": horizon,
                                    "coefficient_original": float(coefficient),
                                    "coefficient_standardized": np.nan,
                                    "selected": True,
                                    "selected_penalty": fit.selected_penalty,
                                }
                            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    forecasts = pd.DataFrame(forecast_rows)
    if forecasts.empty:
        raise ValueError("The outer forecast sample is empty.")
    return WalkForwardResult(
        forecasts=forecasts,
        coefficients=pd.DataFrame(coefficient_rows),
        edge_history=pd.DataFrame(edge_rows),
        tuning_history=pd.DataFrame(tuning_rows),
    )


def walk_forward_forecasts(
    log_variance: pd.DataFrame,
    *,
    spec_name: str,
    factor_window_sessions: int,
    har_window: str | int = "expanding",
    config: HARConfig = HARConfig(),
) -> WalkForwardResult:
    """Estimate forecasts with one persistent worker per target equation."""

    design = build_har_features(log_variance)
    stocks = list(log_variance.columns)
    if int(config.n_jobs) < 1:
        raise ValueError("n_jobs must be at least one.")
    tasks = []
    for target in stocks:
        own, cross, feature_names, cross_pairs = _feature_matrix(design, target, stocks)
        tasks.append(
            _TargetTask(
                target=target,
                y_all=design.response[target].to_numpy(dtype=float),
                own_all=own,
                cross_all=cross,
                origin_logs=log_variance.loc[design.origins, target].to_numpy(dtype=float),
                feature_names=feature_names,
                cross_pairs=cross_pairs,
                origins=design.origins,
                targets=design.targets,
                spec_name=spec_name,
                factor_window_sessions=factor_window_sessions,
                har_window=har_window,
                config=config,
            )
        )
    executor = (
        ProcessPoolExecutor(max_workers=min(int(config.n_jobs), len(tasks)))
        if int(config.n_jobs) > 1 and len(tasks) > 1
        else None
    )
    try:
        results = (
            list(executor.map(_walk_forward_target_task, tasks, chunksize=1))
            if executor is not None
            else [_walk_forward_target_task(task) for task in tasks]
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    forecasts = pd.DataFrame([row for result in results for row in result.forecasts])
    if forecasts.empty:
        raise ValueError("The outer forecast sample is empty.")
    return WalkForwardResult(
        forecasts=forecasts,
        coefficients=pd.DataFrame([row for result in results for row in result.coefficients]),
        edge_history=pd.DataFrame([row for result in results for row in result.edges]),
        tuning_history=pd.DataFrame([row for result in results for row in result.tuning]),
    )


def summarize_forecasts(
    forecasts: pd.DataFrame,
    dimensions: Sequence[str] = ("spec_name", "model"),
) -> pd.DataFrame:
    """Return pooled, stock, and period performance summaries."""

    required = {"qlike", "log_mse", "log_mae", "directional_hit", *dimensions}
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Forecast table is missing columns: {sorted(missing)}")
    result = (
        forecasts.groupby(list(dimensions), dropna=False)
        .agg(
            n_forecasts=("qlike", "size"),
            qlike=("qlike", "mean"),
            log_mse=("log_mse", "mean"),
            log_mae=("log_mae", "mean"),
            directional_accuracy=("directional_hit", "mean"),
            first_target_date=("target_date", "min"),
            last_target_date=("target_date", "max"),
        )
        .reset_index()
    )
    return result


def _bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    if finite.any():
        order = np.argsort(p[finite])
        ordered = p[finite][order]
        values = ordered * len(ordered) / np.arange(1, len(ordered) + 1)
        values = np.minimum.accumulate(values[::-1])[::-1]
        values = np.clip(values, 0.0, 1.0)
        target = np.flatnonzero(finite)[order]
        adjusted[target] = values
    return adjusted


def hac_mean_test(values: Sequence[float], max_lag: int = 5) -> dict[str, float]:
    """Test a mean loss differential with a Bartlett HAC estimator."""

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"n": float(len(x)), "mean_difference": np.nan, "hac_se": np.nan, "z": np.nan, "p_value": np.nan}
    mean = float(x.mean())
    centered = x - mean
    lag = min(max(int(max_lag), 0), len(x) - 1)
    long_run = float(np.dot(centered, centered) / len(x))
    for ell in range(1, lag + 1):
        covariance = float(np.dot(centered[ell:], centered[:-ell]) / len(x))
        long_run += 2.0 * (1.0 - ell / (lag + 1.0)) * covariance
    variance_mean = max(long_run / len(x), 0.0)
    se = float(np.sqrt(variance_mean))
    z = mean / se if se > 0 else (np.inf if mean > 0 else -np.inf if mean < 0 else 0.0)
    p_value = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else 0.0
    return {"n": float(len(x)), "mean_difference": mean, "hac_se": se, "z": float(z), "p_value": p_value}


def build_hac_tests(forecasts: pd.DataFrame, *, max_lag: int = 5) -> pd.DataFrame:
    """Compare network HAR and own HAR using HAC DM-style loss differences."""

    def interpretation(mean_difference: float) -> str:
        if not np.isfinite(mean_difference):
            return "unavailable"
        if mean_difference < 0.0:
            return "negative favors network HAR"
        if mean_difference > 0.0:
            return "positive favors own HAR"
        return "zero loss difference"

    keys = ["spec_name", "factor_window_sessions", "har_window", "forecast_origin", "target_date", "stock"]
    own = forecasts[forecasts["model"] == "own_har"].loc[:, keys + ["qlike"]].rename(columns={"qlike": "own_qlike"})
    network = forecasts[forecasts["model"] == "network_har_l1_1se"].loc[:, keys + ["qlike"]].rename(columns={"qlike": "network_qlike"})
    joined = own.merge(network, on=keys, how="inner")
    joined["loss_difference"] = joined["network_qlike"] - joined["own_qlike"]
    rows: list[dict[str, object]] = []
    for (spec_name, factor_window, har_window), group in joined.groupby(
        ["spec_name", "factor_window_sessions", "har_window"], dropna=False
    ):
        pooled = group.groupby("target_date", as_index=False)["loss_difference"].mean()
        result = hac_mean_test(pooled["loss_difference"], max_lag=max_lag)
        rows.append(
            {
                "spec_name": spec_name,
                "factor_window_sessions": factor_window,
                "har_window": har_window,
                "level": "pooled",
                "stock": "ALL",
                **result,
                "interpretation": interpretation(result["mean_difference"]),
            }
        )
        stock_rows: list[dict[str, object]] = []
        for stock, stock_group in group.groupby("stock"):
            result = hac_mean_test(stock_group["loss_difference"], max_lag=max_lag)
            stock_rows.append(
                {
                    "spec_name": spec_name,
                    "factor_window_sessions": factor_window,
                    "har_window": har_window,
                    "level": "stock",
                    "stock": stock,
                    **result,
                    "interpretation": interpretation(result["mean_difference"]),
                }
            )
        p_adjusted = _bh_adjust([row["p_value"] for row in stock_rows])
        for row, adjusted in zip(stock_rows, p_adjusted):
            row["bh_adjusted_p_value"] = float(adjusted) if np.isfinite(adjusted) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _edge_origin_table(edge_history: pd.DataFrame) -> pd.DataFrame:
    return (
        edge_history.groupby(
            ["spec_name", "factor_window_sessions", "har_window", "model", "forecast_origin", "source", "target", "edge_id"],
            dropna=False,
        )
        .agg(
            selected_any=("selected", "max"),
            absolute_strength=("coefficient_original", lambda x: float(np.abs(x).sum())),
            signed_strength=("coefficient_original", "sum"),
        )
        .reset_index()
    )


def edge_stability(
    edge_history: pd.DataFrame,
    *,
    threshold: float = 0.70,
    model: str = "network_har_l1_1se",
) -> pd.DataFrame:
    """Summarize directed edge selection, signs, support stability, and strength."""

    if edge_history.empty:
        return pd.DataFrame()
    history = edge_history[edge_history["model"] == model].copy()
    if history.empty:
        return pd.DataFrame()
    origin = _edge_origin_table(history)
    rows: list[dict[str, object]] = []
    group_columns = ["spec_name", "factor_window_sessions", "har_window", "model", "source", "target", "edge_id"]
    for keys, group in history.groupby(group_columns, dropna=False):
        group = group.sort_values(["forecast_origin", "horizon"])
        selected_coefficients = group.loc[group["selected"], "coefficient_original"]
        signs = np.sign(selected_coefficients.to_numpy(dtype=float))
        positive = int(np.sum(signs > 0))
        negative = int(np.sum(signs < 0))
        edge_origin = origin[
            (origin["spec_name"] == keys[0])
            & (origin["factor_window_sessions"] == keys[1])
            & (origin["har_window"] == keys[2])
            & (origin["model"] == keys[3])
            & (origin["source"] == keys[4])
            & (origin["target"] == keys[5])
        ].sort_values("forecast_origin")
        supports: list[set[str]] = []
        for _, period in group.groupby("forecast_origin"):
            supports.append(set(period.loc[period["selected"], "horizon"].astype(str)))
        jaccards = [
            len(left & right) / len(left | right) if left | right else 1.0
            for left, right in zip(supports[:-1], supports[1:])
        ]
        selection_dates = group.loc[group["selected"], "forecast_origin"]
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_origin_periods": int(len(edge_origin)),
                "selection_probability": float(edge_origin["selected_any"].mean()),
                "sign_consistency": float(max(positive, negative) / len(signs)) if len(signs) else np.nan,
                "median_coefficient": float(selected_coefficients.median()) if len(selected_coefficients) else 0.0,
                "coefficient_q1": float(selected_coefficients.quantile(0.25)) if len(selected_coefficients) else 0.0,
                "coefficient_q3": float(selected_coefficients.quantile(0.75)) if len(selected_coefficients) else 0.0,
                "median_absolute_strength": float(edge_origin["absolute_strength"].median()),
                "support_jaccard": float(np.mean(jaccards)) if jaccards else np.nan,
                "first_selection_date": selection_dates.min() if len(selection_dates) else pd.NaT,
                "last_selection_date": selection_dates.max() if len(selection_dates) else pd.NaT,
                "stable_edge": bool(edge_origin["selected_any"].mean() >= threshold),
                "stability_threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def network_density(edge_history: pd.DataFrame, *, threshold: float = 0.70) -> pd.DataFrame:
    """Return average directed edge density and stable-edge density."""

    if edge_history.empty:
        return pd.DataFrame()
    origin = _edge_origin_table(edge_history)
    n_stocks = len(set(origin["source"]) | set(origin["target"]))
    possible = max(n_stocks * (n_stocks - 1), 1)
    density = (
        origin.groupby(["spec_name", "factor_window_sessions", "har_window", "model", "forecast_origin"])["selected_any"]
        .sum()
        .div(possible)
        .rename("edge_density")
        .reset_index()
    )
    stable_tables = [
        edge_stability(edge_history, threshold=threshold, model=model_name)
        for model_name in edge_history["model"].dropna().unique()
    ]
    stable = pd.concat([table for table in stable_tables if not table.empty], ignore_index=True) if any(
        not table.empty for table in stable_tables
    ) else pd.DataFrame()
    result = (
        density.groupby(["spec_name", "factor_window_sessions", "har_window", "model"])
        .agg(
            n_origin_periods=("edge_density", "size"),
            mean_edge_density=("edge_density", "mean"),
            min_edge_density=("edge_density", "min"),
            max_edge_density=("edge_density", "max"),
        )
        .reset_index()
    )
    if not stable.empty:
        stable_count = (
            stable.groupby(["spec_name", "factor_window_sessions", "har_window", "model"])["stable_edge"]
            .sum()
            .rename("n_stable_edges")
            .reset_index()
        )
        result = result.merge(stable_count, how="left", on=["spec_name", "factor_window_sessions", "har_window", "model"])
        result["stable_edge_density"] = result["n_stable_edges"] / possible
    else:
        result["n_stable_edges"] = 0
        result["stable_edge_density"] = 0.0
    result["possible_directed_edges"] = possible
    result["stability_threshold"] = threshold
    return result


def network_centrality(edge_stability_table: pd.DataFrame) -> pd.DataFrame:
    """Calculate outgoing and incoming weighted centrality from edge summaries."""

    if edge_stability_table.empty:
        return pd.DataFrame()
    group_columns = ["spec_name", "factor_window_sessions", "har_window", "model"]
    outgoing = (
        edge_stability_table.groupby(group_columns + ["source"])
        .agg(
            outgoing_strength=("median_absolute_strength", "sum"),
            outgoing_selection_probability=("selection_probability", "mean"),
            outgoing_stable_edges=("stable_edge", "sum"),
        )
        .reset_index()
        .rename(columns={"source": "stock"})
    )
    incoming = (
        edge_stability_table.groupby(group_columns + ["target"])
        .agg(
            incoming_strength=("median_absolute_strength", "sum"),
            incoming_selection_probability=("selection_probability", "mean"),
            incoming_stable_edges=("stable_edge", "sum"),
        )
        .reset_index()
        .rename(columns={"target": "stock"})
    )
    return outgoing.merge(incoming, how="outer", on=group_columns + ["stock"]).fillna(0.0)


def group_connectivity(
    edge_stability_table: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Summarize within- and between-group directed connectivity."""

    if edge_stability_table.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in edge_stability_table.groupby(
        ["spec_name", "factor_window_sessions", "har_window", "model"], dropna=False
    ):
        for source_group in groups:
            for target_group in groups:
                selected = group[
                    group["source"].isin(groups[source_group])
                    & group["target"].isin(groups[target_group])
                ]
                possible = len(groups[source_group]) * len(groups[target_group])
                if source_group == target_group:
                    possible -= len(groups[source_group])
                rows.append(
                    {
                        "spec_name": keys[0],
                        "factor_window_sessions": keys[1],
                        "har_window": keys[2],
                        "model": keys[3],
                        "source_group": source_group,
                        "target_group": target_group,
                        "within_group": source_group == target_group,
                        "possible_directed_edges": max(possible, 0),
                        "mean_selection_probability": float(selected["selection_probability"].mean()) if not selected.empty else 0.0,
                        "median_edge_strength": float(selected["median_absolute_strength"].median()) if not selected.empty else 0.0,
                        "stable_edges": int(selected["stable_edge"].sum()) if not selected.empty else 0,
                    }
                )
    result = pd.DataFrame(rows)
    result["stable_edge_density"] = result["stable_edges"] / result["possible_directed_edges"].replace(0, np.nan)
    return result


def _legacy_descriptive_network_edges(
    log_variance: pd.DataFrame,
    *,
    spec_name: str,
    factor_window_sessions: int,
    har_window: str | int = "expanding",
    config: HARConfig = HARConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit checkpointed descriptive network architectures for control panels."""

    design = build_har_features(log_variance)
    stocks = list(log_variance.columns)
    checkpoint_step = max(config.bootstrap_checkpoint_step, 1)
    rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    first = max(config.min_training_observations, 1)
    target_designs = {
        target: _feature_matrix(design, target, stocks) for target in stocks
    }
    checkpoints = list(range(first, len(design.origins), checkpoint_step))
    if checkpoints and checkpoints[-1] != len(design.origins) - 1:
        checkpoints.append(len(design.origins) - 1)
    for row_number in checkpoints:
        train_start = 0 if har_window == "expanding" else max(0, row_number - int(har_window))
        for target in stocks:
            own_all, cross_all, feature_names, cross_pairs = target_designs[target]
            y_all = design.response[target].to_numpy(dtype=float)
            y = y_all[train_start:row_number]
            own = own_all[train_start:row_number]
            cross = cross_all[train_start:row_number]
            tuning = tune_network_penalty(y, own, cross, config=config, cross_feature_names=feature_names)
            fit = fit_partialling_out(
                y,
                own,
                cross,
                alpha=tuning.one_se_alpha,
                cross_feature_names=feature_names,
                config=config,
            )
            for fraction_row in tuning.cv_scores.to_dict("records"):
                tuning_rows.append(
                    {
                        **fraction_row,
                        "spec_name": spec_name,
                        "factor_window_sessions": factor_window_sessions,
                        "har_window": har_window,
                        "checkpoint_date": design.origins[row_number - 1],
                        "training_start": design.origins[train_start],
                        "training_end": design.origins[row_number - 1],
                        "stock": target,
                        "selected_one_se_fraction": tuning.one_se_fraction,
                        "selected_min_loss_fraction": tuning.min_fraction,
                    }
                )
            for coefficient, standardized, (source, horizon) in zip(
                fit.network_coefficients_original,
                fit.network_coefficients_standardized,
                cross_pairs,
            ):
                rows.append(
                    {
                        "spec_name": spec_name,
                        "factor_window_sessions": factor_window_sessions,
                        "har_window": har_window,
                        "forecast_origin": design.origins[row_number - 1],
                        "training_start": design.origins[train_start],
                        "training_end": design.origins[row_number - 1],
                        "source": source,
                        "target": target,
                        "edge_id": f"{source}->{target}",
                        "model": "descriptive_network",
                        "horizon": horizon,
                        "coefficient_original": float(coefficient),
                        "coefficient_standardized": float(standardized),
                        "selected": bool(abs(coefficient) > config.coefficient_tolerance),
                        "sign": int(np.sign(coefficient)),
                        "n_train_observations": len(y),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(tuning_rows)


def descriptive_network_edges(
    log_variance: pd.DataFrame,
    *,
    spec_name: str,
    factor_window_sessions: int,
    har_window: str | int = "expanding",
    config: HARConfig = HARConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit checkpointed descriptive networks with one worker per target."""

    design = build_har_features(log_variance)
    stocks = list(log_variance.columns)
    if int(config.n_jobs) < 1:
        raise ValueError("n_jobs must be at least one.")
    checkpoint_step = max(config.bootstrap_checkpoint_step, 1)
    first = max(config.min_training_observations, 1)
    checkpoints = list(range(first, len(design.origins), checkpoint_step))
    if checkpoints and checkpoints[-1] != len(design.origins) - 1:
        checkpoints.append(len(design.origins) - 1)
    tasks = []
    for target in stocks:
        own, cross, feature_names, cross_pairs = _feature_matrix(design, target, stocks)
        tasks.append(
            _DescriptiveTask(
                target=target,
                y_all=design.response[target].to_numpy(dtype=float),
                own_all=own,
                cross_all=cross,
                feature_names=feature_names,
                cross_pairs=cross_pairs,
                origins=design.origins,
                checkpoints=checkpoints,
                spec_name=spec_name,
                factor_window_sessions=factor_window_sessions,
                har_window=har_window,
                config=config,
            )
        )
    executor = (
        ProcessPoolExecutor(max_workers=min(int(config.n_jobs), len(tasks)))
        if int(config.n_jobs) > 1 and len(tasks) > 1
        else None
    )
    try:
        results = (
            list(executor.map(_descriptive_target_task, tasks, chunksize=1))
            if executor is not None
            else [_descriptive_target_task(task) for task in tasks]
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return (
        pd.DataFrame([row for result in results for row in result.edges]),
        pd.DataFrame([row for result in results for row in result.tuning]),
    )


def _block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if block_length <= 0:
        raise ValueError("Bootstrap block length must be positive.")
    starts = rng.integers(0, max(1, n - block_length + 1), size=int(np.ceil(n / block_length)))
    indices = np.concatenate([np.arange(start, min(start + block_length, n)) for start in starts])
    return indices[:n]


def _bootstrap_target_task(task: _BootstrapTask) -> list[dict[str, object]]:
    """Bootstrap one target across all checkpoints in one worker."""

    rows: list[dict[str, object]] = []
    for checkpoint in task.checkpoints:
        if task.har_window == "expanding":
            train_start = 0
        else:
            train_length = int(task.har_window)
            if train_length <= 0:
                raise ValueError("har_window must be positive or 'expanding'.")
            train_start = max(0, checkpoint - train_length)
        y = task.y_all[train_start:checkpoint]
        own = task.own_all[train_start:checkpoint]
        cross = task.cross_all[train_start:checkpoint]
        tuning = tune_network_penalty(
            y,
            own,
            cross,
            config=task.config,
            cross_feature_names=task.feature_names,
        )
        for block_length in task.config.bootstrap_block_lengths:
            selections: dict[tuple[str, str, str], list[bool]] = {}
            signs: dict[tuple[str, str, str], list[int]] = {}
            for sample in task.samples[checkpoint][int(block_length)]:
                fit = fit_partialling_out(
                    y[sample],
                    own[sample],
                    cross[sample],
                    alpha=tuning.one_se_alpha,
                    cross_feature_names=task.feature_names,
                    config=task.config,
                )
                for coefficient, (source, horizon) in zip(
                    fit.network_coefficients_original,
                    task.cross_pairs,
                ):
                    key = (source, task.target, horizon)
                    selections.setdefault(key, []).append(
                        bool(abs(coefficient) > task.config.coefficient_tolerance)
                    )
                    signs.setdefault(key, []).append(int(np.sign(coefficient)))
            for (source, target, horizon), selected in selections.items():
                selected_array = np.asarray(selected, dtype=bool)
                sign_array = np.asarray(signs[(source, target, horizon)], dtype=int)
                nonzero = sign_array[sign_array != 0]
                rows.append(
                    {
                        "spec_name": task.spec_name,
                        "factor_window_sessions": task.factor_window_sessions,
                        "har_window": task.har_window,
                        "checkpoint_date": task.origins[checkpoint - 1],
                        "training_start": task.origins[train_start],
                        "training_end": task.origins[checkpoint - 1],
                        "block_length": block_length,
                        "source": source,
                        "target": target,
                        "edge_id": f"{source}->{target}",
                        "horizon": horizon,
                        "bootstrap_selection_probability": float(selected_array.mean()),
                        "bootstrap_sign_consistency": float(
                            max(np.sum(nonzero > 0), np.sum(nonzero < 0)) / len(nonzero)
                        )
                        if len(nonzero)
                        else np.nan,
                        "n_bootstrap": len(selected_array),
                    }
                )
    return rows


def _legacy_bootstrap_edge_selection(
    log_variance: pd.DataFrame,
    *,
    spec_name: str,
    factor_window_sessions: int,
    har_window: str | int = "expanding",
    config: HARConfig = HARConfig(),
) -> pd.DataFrame:
    """Estimate block-bootstrap selection probabilities at checkpoint dates."""

    design = build_har_features(log_variance)
    stocks = list(log_variance.columns)
    first = max(config.min_training_observations, 1)
    checkpoints = list(range(first, len(design.origins), max(config.bootstrap_checkpoint_step, 1)))
    if checkpoints and checkpoints[-1] != len(design.origins) - 1:
        checkpoints.append(len(design.origins) - 1)
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, object]] = []
    target_designs = {
        target: _feature_matrix(design, target, stocks) for target in stocks
    }
    for checkpoint in checkpoints:
        train_start = 0 if har_window == "expanding" else max(0, checkpoint - int(har_window))
        target_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], list[tuple[str, str]], PenaltyTuning]] = {}
        for target in stocks:
            own_all, cross_all, feature_names, cross_pairs = target_designs[target]
            y_all = design.response[target].to_numpy(dtype=float)
            y = y_all[train_start:checkpoint]
            own = own_all[train_start:checkpoint]
            cross = cross_all[train_start:checkpoint]
            tuning = tune_network_penalty(y, own, cross, config=config, cross_feature_names=feature_names)
            target_data[target] = (y, own, cross, feature_names, cross_pairs, tuning)
        for block_length in config.bootstrap_block_lengths:
            selections: dict[tuple[str, str, str], list[bool]] = {}
            signs: dict[tuple[str, str, str], list[int]] = {}
            for _ in range(config.bootstrap_repetitions):
                for target, (y, own, cross, names, pairs, tuning) in target_data.items():
                    sample = _block_indices(len(y), block_length, rng)
                    fit = fit_partialling_out(
                        y[sample],
                        own[sample],
                        cross[sample],
                        alpha=tuning.one_se_alpha,
                        cross_feature_names=names,
                        config=config,
                    )
                    for coefficient, (source, horizon) in zip(fit.network_coefficients_original, pairs):
                        key = (source, target, horizon)
                        selections.setdefault(key, []).append(bool(abs(coefficient) > config.coefficient_tolerance))
                        signs.setdefault(key, []).append(int(np.sign(coefficient)))
            for (source, target, horizon), selected in selections.items():
                selected_array = np.asarray(selected, dtype=bool)
                sign_array = np.asarray(signs[(source, target, horizon)], dtype=int)
                nonzero = sign_array[sign_array != 0]
                rows.append(
                    {
                        "spec_name": spec_name,
                        "factor_window_sessions": factor_window_sessions,
                        "har_window": har_window,
                        "checkpoint_date": design.origins[checkpoint - 1],
                        "training_start": design.origins[train_start],
                        "training_end": design.origins[checkpoint - 1],
                        "block_length": block_length,
                        "source": source,
                        "target": target,
                        "edge_id": f"{source}->{target}",
                        "horizon": horizon,
                        "bootstrap_selection_probability": float(selected_array.mean()),
                        "bootstrap_sign_consistency": float(
                            max(np.sum(nonzero > 0), np.sum(nonzero < 0)) / len(nonzero)
                        )
                        if len(nonzero)
                        else np.nan,
                        "n_bootstrap": len(selected_array),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_edge_selection(
    log_variance: pd.DataFrame,
    *,
    spec_name: str,
    factor_window_sessions: int,
    har_window: str | int = "expanding",
    config: HARConfig = HARConfig(),
) -> pd.DataFrame:
    """Bootstrap edge persistence with one persistent worker per target."""

    design = build_har_features(log_variance)
    stocks = list(log_variance.columns)
    if int(config.n_jobs) < 1:
        raise ValueError("n_jobs must be at least one.")
    first = max(config.min_training_observations, 1)
    checkpoint_step = max(config.bootstrap_checkpoint_step, 1)
    checkpoints = list(range(first, len(design.origins), checkpoint_step))
    if checkpoints and checkpoints[-1] != len(design.origins) - 1:
        checkpoints.append(len(design.origins) - 1)
    target_designs = {
        target: _feature_matrix(design, target, stocks) for target in stocks
    }
    samples = {
        target: {
            checkpoint: {int(block): [] for block in config.bootstrap_block_lengths}
            for checkpoint in checkpoints
        }
        for target in stocks
    }
    rng = np.random.default_rng(config.random_seed)
    for checkpoint in checkpoints:
        if har_window == "expanding":
            train_start = 0
        else:
            train_length = int(har_window)
            if train_length <= 0:
                raise ValueError("har_window must be positive or 'expanding'.")
            train_start = max(0, checkpoint - train_length)
        n_training = checkpoint - train_start
        for block_length in config.bootstrap_block_lengths:
            for _ in range(config.bootstrap_repetitions):
                for target in stocks:
                    samples[target][checkpoint][int(block_length)].append(
                        _block_indices(n_training, int(block_length), rng)
                    )
    tasks = []
    for target in stocks:
        own, cross, feature_names, cross_pairs = target_designs[target]
        tasks.append(
            _BootstrapTask(
                target=target,
                y_all=design.response[target].to_numpy(dtype=float),
                own_all=own,
                cross_all=cross,
                feature_names=feature_names,
                cross_pairs=cross_pairs,
                origins=design.origins,
                checkpoints=checkpoints,
                samples=samples[target],
                spec_name=spec_name,
                factor_window_sessions=factor_window_sessions,
                har_window=har_window,
                config=config,
            )
        )
    executor = (
        ProcessPoolExecutor(max_workers=min(int(config.n_jobs), len(tasks)))
        if int(config.n_jobs) > 1 and len(tasks) > 1
        else None
    )
    try:
        rows = (
            [row for result in executor.map(_bootstrap_target_task, tasks, chunksize=1) for row in result]
            if executor is not None
            else [row for task in tasks for row in _bootstrap_target_task(task)]
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return pd.DataFrame(rows)
