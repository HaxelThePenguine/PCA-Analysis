"""Causal factor-augmented HAR forecasts for benchmark-residual variance.

Stage 17 asks a different question from Stage 16.  The response is realized
variance after removing only SPY and XLF, while the predictors add common and
economically localized factor-volatility histories.  Forecast issuance remains
separate from outcome scoring and never reads the current target observation.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from utils.network_har import (
    HORIZONS,
    HARFit,
    PenaltyTuning,
    _predict_fit,
    fit_own_har,
    fit_partialling_out,
    tune_network_penalty,
)
from utils.oos_common import (
    calendar_time_maps,
    normalize_calendar,
    safe_positive_variance,
    session_date_index,
    training_bounds,
    utc_now_iso,
)
from utils.oos_har_network import (
    OOSConfig,
    ForecastRunResult,
    build_calendar_har_features,
    validate_identical_model_keys,
)


OWN_MODEL = "own_har"
AGGREGATE_FACTOR_MODEL = "aggregate_factor_har"
LOCAL_FACTOR_MODEL = "local_factor_har"
NETWORK_MODEL = "benchmark_network_har_l1_1se"
HYBRID_MODEL = "local_factor_network_har_l1_1se"
PERSISTENCE_MODEL = "persistence"
FACTOR_HAR_MODELS = (
    OWN_MODEL,
    AGGREGATE_FACTOR_MODEL,
    LOCAL_FACTOR_MODEL,
    NETWORK_MODEL,
    HYBRID_MODEL,
    PERSISTENCE_MODEL,
)


@dataclass(frozen=True)
class FactorHARDesign:
    """Aligned bank and factor HAR designs on one exchange calendar."""

    origins: pd.DatetimeIndex
    targets: pd.DatetimeIndex
    response: pd.DataFrame
    bank_features: pd.DataFrame
    aggregate_features: pd.DataFrame
    local_features: pd.DataFrame


@dataclass(frozen=True)
class _FactorOriginTask:
    target: str
    y_train: np.ndarray
    own_train: np.ndarray
    aggregate_train: np.ndarray
    local_train: np.ndarray
    cross_train: np.ndarray
    current_own: np.ndarray
    current_aggregate: np.ndarray
    current_local: np.ndarray
    current_cross: np.ndarray
    aggregate_names: tuple[str, ...]
    local_names: tuple[str, ...]
    cross_names: tuple[str, ...]
    cross_pairs: tuple[tuple[str, str], ...]
    network_tuning: PenaltyTuning
    hybrid_tuning: PenaltyTuning
    common: Mapping[str, object]
    config: OOSConfig


@dataclass(frozen=True)
class _FactorOriginResult:
    forecasts: list[dict[str, object]]
    coefficients: list[dict[str, object]]
    edges: list[dict[str, object]]
    diagnostic: dict[str, object]


def build_factor_har_design(
    benchmark_log_variance: pd.DataFrame,
    aggregate_factor_log_variance: pd.DataFrame,
    local_factor_log_variance: pd.DataFrame,
    calendar: pd.DataFrame,
) -> FactorHARDesign:
    """Create strictly aligned D/W/M predictors for banks and factor RV."""

    bank = build_calendar_har_features(benchmark_log_variance, calendar)
    aggregate = build_calendar_har_features(aggregate_factor_log_variance, calendar)
    local = build_calendar_har_features(local_factor_log_variance, calendar)
    for candidate, label in ((aggregate, "aggregate"), (local, "local")):
        if not candidate.origins.equals(bank.origins):
            raise ValueError(f"The {label} factor origins do not match the bank origins.")
        if not candidate.targets.equals(bank.targets):
            raise ValueError(f"The {label} factor targets do not match the bank targets.")
    return FactorHARDesign(
        origins=bank.origins,
        targets=bank.targets,
        response=bank.response,
        bank_features=bank.features,
        aggregate_features=aggregate.features,
        local_features=local.features,
    )


def _target_arrays(
    design: FactorHARDesign,
    target: str,
    stocks: Sequence[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
]:
    own_pairs = [(target, horizon) for horizon in HORIZONS]
    cross_pairs = tuple(
        (source, horizon)
        for source in stocks
        if source != target
        for horizon in HORIZONS
    )
    aggregate_pairs = list(design.aggregate_features.columns)
    local_pairs = list(design.local_features.columns)
    return (
        design.response[target].to_numpy(dtype=float),
        design.bank_features.loc[:, own_pairs].to_numpy(dtype=float),
        design.aggregate_features.to_numpy(dtype=float),
        design.local_features.to_numpy(dtype=float),
        design.bank_features.loc[:, list(cross_pairs)].to_numpy(dtype=float),
        tuple(f"aggregate:{factor}:{horizon}" for factor, horizon in aggregate_pairs),
        tuple(f"local:{factor}:{horizon}" for factor, horizon in local_pairs),
        tuple(f"bank:{source}:{horizon}" for source, horizon in cross_pairs),
        cross_pairs,
    )


def _common_mask(
    y: np.ndarray,
    own: np.ndarray,
    aggregate: np.ndarray,
    local: np.ndarray,
    cross: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """Use one training sample for every model in a matched comparison."""

    return (
        np.isfinite(y[start:end])
        & np.isfinite(own[start:end]).all(axis=1)
        & np.isfinite(aggregate[start:end]).all(axis=1)
        & np.isfinite(local[start:end]).all(axis=1)
        & np.isfinite(cross[start:end]).all(axis=1)
    )


def _coefficient_rows(
    fit: HARFit,
    *,
    model: str,
    names: Sequence[str],
    common: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = [
        {
            **common,
            "model": model,
            "predictor": "Intercept",
            "coefficient_original": fit.intercept,
            "coefficient_standardized": np.nan,
            "selected": True,
        }
    ]
    rows.extend(
        {
            **common,
            "model": model,
            "predictor": name,
            "coefficient_original": float(coefficient),
            "coefficient_standardized": np.nan,
            "selected": True,
        }
        for name, coefficient in zip(names, fit.own_coefficients)
    )
    return rows


def _forecast_row(
    fit: HARFit,
    *,
    model: str,
    own: np.ndarray,
    cross: np.ndarray,
    common: Mapping[str, object],
    config: OOSConfig,
) -> dict[str, object]:
    predicted_log = float(_predict_fit(fit, own, cross)[0])
    predicted_variance, clipped = safe_positive_variance(
        predicted_log,
        fit.smearing_factor,
        config.variance.variance_floor,
    )
    return {
        **common,
        "model": model,
        "selected_penalty": fit.selected_penalty,
        "predicted_log_variance": predicted_log,
        "predicted_variance": predicted_variance,
        "smearing_factor": fit.smearing_factor,
        "prediction_exponent_clipped": clipped,
        "design_condition_number": fit.condition_number,
        "column_norm_ratio": fit.column_norm_ratio,
        "lasso_converged": fit.lasso_converged,
        "lasso_iterations": fit.lasso_iterations,
        "lasso_kkt_max_violation": fit.lasso_kkt_max_violation,
    }


def _fit_factor_origin(task: _FactorOriginTask) -> _FactorOriginResult:
    own_names = tuple(f"own:{task.target}:{horizon}" for horizon in HORIZONS)
    own_fit = fit_own_har(task.y_train, task.own_train, config=task.config.har)
    aggregate_base = np.column_stack([task.own_train, task.aggregate_train])
    aggregate_fit = fit_own_har(task.y_train, aggregate_base, config=task.config.har)
    local_base = np.column_stack([task.own_train, task.local_train])
    local_fit = fit_own_har(task.y_train, local_base, config=task.config.har)
    network_fit = fit_partialling_out(
        task.y_train,
        task.own_train,
        task.cross_train,
        alpha=task.network_tuning.one_se_alpha,
        cross_feature_names=task.cross_names,
        config=task.config.har,
    )
    hybrid_fit = fit_partialling_out(
        task.y_train,
        local_base,
        task.cross_train,
        alpha=task.hybrid_tuning.one_se_alpha,
        cross_feature_names=task.cross_names,
        config=task.config.har,
    )
    empty = np.empty((1, 0))
    forecasts = [
        _forecast_row(
            own_fit,
            model=OWN_MODEL,
            own=task.current_own,
            cross=empty,
            common=task.common,
            config=task.config,
        ),
        _forecast_row(
            aggregate_fit,
            model=AGGREGATE_FACTOR_MODEL,
            own=np.column_stack([task.current_own, task.current_aggregate]),
            cross=empty,
            common=task.common,
            config=task.config,
        ),
        _forecast_row(
            local_fit,
            model=LOCAL_FACTOR_MODEL,
            own=np.column_stack([task.current_own, task.current_local]),
            cross=empty,
            common=task.common,
            config=task.config,
        ),
        _forecast_row(
            network_fit,
            model=NETWORK_MODEL,
            own=task.current_own,
            cross=task.current_cross,
            common=task.common,
            config=task.config,
        ),
        _forecast_row(
            hybrid_fit,
            model=HYBRID_MODEL,
            own=np.column_stack([task.current_own, task.current_local]),
            cross=task.current_cross,
            common=task.common,
            config=task.config,
        ),
    ]
    persistence_variance, persistence_clipped = safe_positive_variance(
        float(task.current_own[0, 0]), 1.0, task.config.variance.variance_floor
    )
    forecasts.append(
        {
            **task.common,
            "model": PERSISTENCE_MODEL,
            "selected_penalty": np.nan,
            "predicted_log_variance": float(task.current_own[0, 0]),
            "predicted_variance": persistence_variance,
            "smearing_factor": 1.0,
            "prediction_exponent_clipped": persistence_clipped,
            "design_condition_number": np.nan,
            "column_norm_ratio": np.nan,
            "lasso_converged": True,
            "lasso_iterations": 0,
            "lasso_kkt_max_violation": 0.0,
        }
    )

    coefficients: list[dict[str, object]] = []
    coefficients.extend(
        _coefficient_rows(own_fit, model=OWN_MODEL, names=own_names, common=task.common)
    )
    coefficients.extend(
        _coefficient_rows(
            aggregate_fit,
            model=AGGREGATE_FACTOR_MODEL,
            names=(*own_names, *task.aggregate_names),
            common=task.common,
        )
    )
    coefficients.extend(
        _coefficient_rows(
            local_fit,
            model=LOCAL_FACTOR_MODEL,
            names=(*own_names, *task.local_names),
            common=task.common,
        )
    )
    coefficients.extend(
        _coefficient_rows(
            network_fit, model=NETWORK_MODEL, names=own_names, common=task.common
        )
    )
    coefficients.extend(
        _coefficient_rows(
            hybrid_fit,
            model=HYBRID_MODEL,
            names=(*own_names, *task.local_names),
            common=task.common,
        )
    )
    edges: list[dict[str, object]] = []
    for model, fit in ((NETWORK_MODEL, network_fit), (HYBRID_MODEL, hybrid_fit)):
        for original, standardized, (source, horizon) in zip(
            fit.network_coefficients_original,
            fit.network_coefficients_standardized,
            task.cross_pairs,
        ):
            edges.append(
                {
                    **task.common,
                    "model": model,
                    "source": source,
                    "target": task.target,
                    "edge_id": f"{source}->{task.target}",
                    "horizon": horizon,
                    "coefficient_original": float(original),
                    "coefficient_standardized": float(standardized),
                    "selected": bool(
                        abs(original) > task.config.har.coefficient_tolerance
                    ),
                }
            )
    return _FactorOriginResult(
        forecasts=forecasts,
        coefficients=coefficients,
        edges=edges,
        diagnostic={
            "spec_name": task.common["spec_name"],
            "forecast_origin": task.common["forecast_origin"],
            "target_date": task.common["target_date"],
            "stock": task.target,
            "n_train_observations": len(task.y_train),
            "status": "issued",
            "target_data_present_at_issue": False,
        },
    )


def issue_factor_har_forecasts(
    benchmark_log_variance: pd.DataFrame,
    aggregate_factor_log_variance: pd.DataFrame,
    local_factor_log_variance: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    spec_name: str,
    config: OOSConfig = OOSConfig(),
    run_id: str = "uncommitted-factor-har-run",
    factor_metadata: pd.DataFrame | None = None,
    har_window: str | int = "expanding",
    mode: str = "historical_pseudo_oos",
    freeze_timestamp: str | None = None,
    issuance_timestamp: str | None = None,
) -> ForecastRunResult:
    """Issue matched factor, network, hybrid and benchmark forecasts."""

    if mode not in {"historical_pseudo_oos", "prospective"}:
        raise ValueError("Unsupported forecast mode.")
    calendar_value = normalize_calendar(calendar)
    design = build_factor_har_design(
        benchmark_log_variance,
        aggregate_factor_log_variance,
        local_factor_log_variance,
        calendar_value,
    )
    stocks = list(benchmark_log_variance.columns)
    opens, closes = calendar_time_maps(calendar_value)
    factor_lookup: dict[pd.Timestamp, dict[str, object]] = {}
    if factor_metadata is not None and not factor_metadata.empty:
        metadata = factor_metadata.copy()
        metadata["session_date"] = session_date_index(metadata["session_date"])
        factor_lookup = {
            pd.Timestamp(row["session_date"]): row for row in metadata.to_dict("records")
        }
    arrays = {target: _target_arrays(design, target, stocks) for target in stocks}
    forecasts: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    tuning_cache: dict[tuple[str, str], PenaltyTuning] = {}
    last_tuned: dict[tuple[str, str], int] = {}
    issued_counts = {target: 0 for target in stocks}

    freeze = pd.Timestamp(freeze_timestamp) if freeze_timestamp is not None else None
    if freeze is not None:
        freeze = freeze.tz_localize("UTC") if freeze.tzinfo is None else freeze.tz_convert("UTC")
    if mode == "prospective" and freeze is None:
        raise ValueError("Prospective issuance requires a freeze timestamp.")
    issued_at = pd.Timestamp(issuance_timestamp or utc_now_iso())
    issued_at = issued_at.tz_localize("UTC") if issued_at.tzinfo is None else issued_at.tz_convert("UTC")
    executor = (
        ProcessPoolExecutor(max_workers=min(config.har.n_jobs, len(stocks)))
        if config.har.n_jobs > 1 and len(stocks) > 1
        else None
    )
    try:
        for row_number, (origin_date, target_date) in enumerate(
            zip(design.origins, design.targets)
        ):
            origin_date = pd.Timestamp(origin_date)
            target_date = pd.Timestamp(target_date)
            origin_as_of = closes.get(origin_date)
            target_open = opens.get(target_date)
            target_close = closes.get(target_date)
            if origin_as_of is None or target_open is None or target_close is None:
                continue
            if not origin_as_of < target_open:
                raise ValueError("Forecast information-clock invariant failed.")
            if mode == "prospective" and (
                target_open.tz_convert("UTC") <= freeze
                or issued_at >= target_open.tz_convert("UTC")
            ):
                continue
            factor_row = factor_lookup.get(origin_date, {})
            tasks: list[_FactorOriginTask] = []
            for target in stocks:
                (
                    y_all,
                    own_all,
                    aggregate_all,
                    local_all,
                    cross_all,
                    aggregate_names,
                    local_names,
                    cross_names,
                    cross_pairs,
                ) = arrays[target]
                if mode == "prospective" and np.isfinite(y_all[row_number]):
                    continue
                current = (
                    own_all[row_number : row_number + 1],
                    aggregate_all[row_number : row_number + 1],
                    local_all[row_number : row_number + 1],
                    cross_all[row_number : row_number + 1],
                )
                if not all(np.isfinite(value).all() for value in current):
                    continue
                train_start, train_end = training_bounds(row_number, har_window)
                mask = _common_mask(
                    y_all,
                    own_all,
                    aggregate_all,
                    local_all,
                    cross_all,
                    train_start,
                    train_end,
                )
                y_train = y_all[train_start:train_end][mask]
                if len(y_train) < config.min_valid_training_observations:
                    continue
                own_train = own_all[train_start:train_end][mask]
                aggregate_train = aggregate_all[train_start:train_end][mask]
                local_train = local_all[train_start:train_end][mask]
                cross_train = cross_all[train_start:train_end][mask]
                issued_counts[target] += 1
                model_bases = {
                    NETWORK_MODEL: own_train,
                    HYBRID_MODEL: np.column_stack([own_train, local_train]),
                }
                for model, base in model_bases.items():
                    key = (target, model)
                    if key not in tuning_cache or (
                        issued_counts[target] - last_tuned.get(key, -10**9)
                        >= config.har.tuning_frequency
                    ):
                        tuning_cache[key] = tune_network_penalty(
                            y_train,
                            base,
                            cross_train,
                            config=config.har,
                            cross_feature_names=cross_names,
                        )
                        last_tuned[key] = issued_counts[target]
                        for cv_row in tuning_cache[key].cv_scores.to_dict("records"):
                            tuning_rows.append(
                                {
                                    **cv_row,
                                    "protocol_version": config.protocol_version,
                                    "run_id": run_id,
                                    "spec_name": spec_name,
                                    "model": model,
                                    "stock": target,
                                    "forecast_origin": origin_date,
                                    "selected_one_se_fraction": tuning_cache[key].one_se_fraction,
                                    "alpha_max_full_sample": tuning_cache[key].alpha_max,
                                }
                            )
                training_targets = design.targets[train_start:train_end][mask]
                training_cutoff = pd.Timestamp(training_targets.max())
                if not closes[training_cutoff] <= origin_as_of < target_open:
                    raise ValueError("A training label was unavailable at issuance.")
                common = {
                    "protocol_version": config.protocol_version,
                    "run_id": run_id,
                    "spec_name": spec_name,
                    "har_window": har_window,
                    "stock": target,
                    "forecast_origin": origin_date,
                    "as_of": origin_as_of,
                    "target_date": target_date,
                    "target_open": target_open,
                    "target_outcome_available_at": target_close,
                    "issued_at_utc": issued_at,
                    "training_label_cutoff": training_cutoff,
                    "training_label_available_at": closes[training_cutoff],
                    "factor_fit_id": factor_row.get("factor_fit_id", np.nan),
                    "factor_training_end": factor_row.get("factor_training_end", np.nan),
                    "factor_window_sessions": factor_row.get("factor_window_sessions", np.nan),
                    "target_data_present_at_issue": False,
                    "n_train_observations": len(y_train),
                }
                tasks.append(
                    _FactorOriginTask(
                        target=target,
                        y_train=y_train,
                        own_train=own_train,
                        aggregate_train=aggregate_train,
                        local_train=local_train,
                        cross_train=cross_train,
                        current_own=current[0],
                        current_aggregate=current[1],
                        current_local=current[2],
                        current_cross=current[3],
                        aggregate_names=aggregate_names,
                        local_names=local_names,
                        cross_names=cross_names,
                        cross_pairs=cross_pairs,
                        network_tuning=tuning_cache[(target, NETWORK_MODEL)],
                        hybrid_tuning=tuning_cache[(target, HYBRID_MODEL)],
                        common=common,
                        config=config,
                    )
                )
            results = (
                list(executor.map(_fit_factor_origin, tasks, chunksize=1))
                if executor is not None and tasks
                else [_fit_factor_origin(task) for task in tasks]
            )
            for result in results:
                forecasts.extend(result.forecasts)
                coefficients.extend(result.coefficients)
                edges.extend(result.edges)
                diagnostics.append(result.diagnostic)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    forecast_frame = pd.DataFrame(forecasts)
    if not forecast_frame.empty:
        validate_identical_model_keys(forecast_frame, FACTOR_HAR_MODELS)
    return ForecastRunResult(
        forecasts=forecast_frame,
        coefficients=pd.DataFrame(coefficients),
        edge_history=pd.DataFrame(edges),
        tuning_history=pd.DataFrame(tuning_rows),
        origin_diagnostics=pd.DataFrame(diagnostics),
        status="complete" if mode == "historical_pseudo_oos" else (
            "issued_pending_outcomes" if forecasts else "awaiting_future_data"
        ),
    )


def matched_qlike_comparisons(
    scored: pd.DataFrame,
    *,
    pairs: Sequence[tuple[str, str]] = (
        (AGGREGATE_FACTOR_MODEL, OWN_MODEL),
        (LOCAL_FACTOR_MODEL, OWN_MODEL),
        (LOCAL_FACTOR_MODEL, AGGREGATE_FACTOR_MODEL),
        (NETWORK_MODEL, OWN_MODEL),
        (HYBRID_MODEL, LOCAL_FACTOR_MODEL),
        (HYBRID_MODEL, NETWORK_MODEL),
    ),
) -> pd.DataFrame:
    """Summarize matched QLIKE differences; negative means model A wins."""

    eligible = scored[
        scored.get("comparison_eligible", False).astype(bool)
        & scored["qlike"].notna()
    ].copy()
    keys = ["spec_name", "stock", "forecast_origin", "target_date"]
    wide = eligible.pivot(index=keys, columns="model", values="qlike")
    rows: list[dict[str, object]] = []
    for model_a, model_b in pairs:
        if model_a not in wide or model_b not in wide:
            continue
        difference = (wide[model_a] - wide[model_b]).dropna()
        for spec_name, group in difference.groupby(level=0):
            values = group.to_numpy(dtype=float)
            benchmark_loss = wide.loc[group.index, model_b].to_numpy(dtype=float)
            rows.append(
                {
                    "spec_name": spec_name,
                    "stock": "ALL",
                    "model_a": model_a,
                    "model_b": model_b,
                    "n": len(values),
                    "mean_qlike_difference": float(values.mean()),
                    "qlike_improvement_pct": float(
                        -100.0 * values.mean() / max(float(benchmark_loss.mean()), 1e-16)
                    ),
                    "win_rate": float(np.mean(values < 0.0)),
                }
            )
        for (spec_name, stock), group in difference.groupby(level=[0, 1]):
            values = group.to_numpy(dtype=float)
            rows.append(
                {
                    "spec_name": spec_name,
                    "stock": stock,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n": len(values),
                    "mean_qlike_difference": float(values.mean()),
                    "qlike_improvement_pct": float(
                        -100.0 * values.mean()
                        / max(float(wide.loc[group.index, model_b].mean()), 1e-16)
                    ),
                    "win_rate": float(np.mean(values < 0.0)),
                }
            )
    return pd.DataFrame(rows)
