"""Auditable Stage 16 out-of-sample HAR and Network HAR utilities.

The module deliberately keeps forecast issuance separate from outcome scoring.
It is therefore possible to persist a forecast ledger while the target session
is still in the future and to join realized variance only in a later scoring
run.  The historical Stage 15 implementation remains available for legacy
comparison; this module is the corrected, versioned protocol.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config import BAD_SESSION_DATES, BENCHMARKS, CORE_UNIVERSE, NY_TZ
from utils.factor_adjusted_residuals import FactorConfig, run_factor_adjustment
from utils.network_har import (
    HAR_LOOKBACK,
    HORIZONS,
    HARConfig,
    PenaltyTuning,
    _predict_fit,
    fit_network_pair,
    fit_partialling_out,
    hac_mean_test,
    tune_network_penalty,
)
from utils.oos_common import (
    calendar_time_maps as _calendar_time_maps,
    dataframe_hash,
    normalize_calendar,
    safe_positive_variance as _safe_positive_variance,
    session_date_index as _session_date_index,
    stable_hash,
    strict_clean_return_panel,
    training_bounds as _training_bounds,
    utc_now_iso,
    write_json,
)
from utils.realized_volatility import (
    RealizedVarianceConfig,
    RealizedVarianceResult,
    aggregate_realized_variance,
)


PRIMARY_SPEC_NAME = "factor_adjusted_120_expanding"
PRIMARY_NETWORK_MODEL = "network_har_l1_1se"
OWN_MODEL = "own_har"
PERSISTENCE_MODEL = "persistence"
NETWORK_MIN_MODEL = "network_har_l1_min_loss"
FORECAST_MODELS = (OWN_MODEL, PRIMARY_NETWORK_MODEL, NETWORK_MIN_MODEL, PERSISTENCE_MODEL)
GROUPS = {
    "MS_C": ("MS", "C"),
    "large_banks": ("JPM", "BAC", "WFC", "C"),
    "regional_banks": ("USB", "TFC", "KEY", "RF", "FITB", "CFG", "HBAN"),
}


@dataclass(frozen=True)
class OOSConfig:
    """Frozen Stage 16 decisions and operational controls."""

    protocol_version: str = "oos-har-network-v2.0.0"
    factor_windows: tuple[int, ...] = (120, 60)
    factor_update_step: int = 5
    factor_components: int = 3
    factor_l1_starts: int = 40
    factor_seed: int = 16016
    har_window_primary: str | int = "expanding"
    har_window_robustness: int = 252
    factor_window_primary: int = 120
    min_valid_training_observations: int = 252
    target_min_valid_intervals: int = 1
    target_min_valid_bar_fraction: float = 0.90
    confirmation_horizon_sessions: int = 252
    confirmation_checkpoints: tuple[int, ...] = (63, 126)
    loss_bootstrap_repetitions: int = 2_000
    loss_bootstrap_block_lengths: tuple[int, ...] = (20, 5)
    edge_bootstrap_repetitions: int = 200
    edge_bootstrap_checkpoint_step: int = 126
    edge_bootstrap_block_lengths: tuple[int, ...] = (20, 5)
    checkpoint_frequency_origins: int = 100
    variance: RealizedVarianceConfig = field(default_factory=RealizedVarianceConfig)
    har: HARConfig = field(
        default_factory=lambda: HARConfig(
            n_jobs=1,
            min_training_observations=252,
            rolling_training_observations=252,
            tuning_frequency=20,
            cv_splits=3,
            alpha_fractions=(0.01, 0.03, 0.10, 0.30, 1.00),
            hac_lag=5,
            bootstrap_repetitions=200,
            bootstrap_block_lengths=(5, 20),
            bootstrap_checkpoint_step=126,
            random_seed=16016,
        )
    )


@dataclass(frozen=True)
class CalendarHARDesign:
    """One-session HAR design indexed by the exchange calendar."""

    calendar_dates: pd.DatetimeIndex
    origins: pd.DatetimeIndex
    targets: pd.DatetimeIndex
    response: pd.DataFrame
    features: pd.DataFrame
    origin_positions: np.ndarray


@dataclass(frozen=True)
class FactorVintageRun:
    """Causal factor-adjusted minute panels and daily RV vintages."""

    spec_name: str
    factor_window_sessions: int
    factor_result: Any
    variance_result: RealizedVarianceResult
    daily_log_variance: pd.DataFrame
    daily_variance: pd.DataFrame
    benchmark_daily_log_variance: pd.DataFrame
    benchmark_daily_variance: pd.DataFrame
    pca_factor_daily_log_variance: pd.DataFrame
    pca_factor_daily_variance: pd.DataFrame
    local_factor_daily_log_variance: pd.DataFrame
    local_factor_daily_variance: pd.DataFrame
    aggregate_factor_daily_log_variance: pd.DataFrame
    aggregate_factor_daily_variance: pd.DataFrame
    factor_metadata: pd.DataFrame
    rv_metadata: pd.DataFrame


@dataclass(frozen=True)
class ForecastRunResult:
    """Forecast and diagnostic ledgers from one or more specifications."""

    forecasts: pd.DataFrame
    coefficients: pd.DataFrame
    edge_history: pd.DataFrame
    tuning_history: pd.DataFrame
    origin_diagnostics: pd.DataFrame
    status: str


def build_calendar_har_features(
    log_daily_variance: pd.DataFrame,
    calendar: pd.DataFrame,
) -> CalendarHARDesign:
    """Construct D/W/M features on exchange-session dates, allowing unknown targets."""

    if not isinstance(log_daily_variance, pd.DataFrame) or log_daily_variance.empty:
        raise ValueError("A non-empty daily log-variance panel is required.")
    if log_daily_variance.columns.has_duplicates:
        raise ValueError("Daily log-variance columns must be unique.")
    dates = _session_date_index(log_daily_variance.index)
    if dates.has_duplicates:
        raise ValueError("Daily log-variance sessions must be unique.")
    calendar_value = normalize_calendar(calendar)
    calendar_dates = pd.DatetimeIndex(calendar_value["session_date"])
    observed = log_daily_variance.copy()
    observed.index = dates
    observed = observed.reindex(calendar_dates)
    values = observed.to_numpy(dtype=float)
    first_position = max(HAR_LOOKBACK.values()) - 1
    if len(calendar_dates) <= first_position + 1:
        raise ValueError("The calendar is too short for 22-session HAR features.")

    origins: list[pd.Timestamp] = []
    targets: list[pd.Timestamp] = []
    feature_rows: list[list[float]] = []
    response_rows: list[np.ndarray] = []
    columns = [(stock, horizon) for stock in observed.columns for horizon in HORIZONS]
    for position in range(first_position, len(calendar_dates) - 1):
        row: list[float] = []
        for stock_position in range(values.shape[1]):
            row.extend(
                [
                    values[position, stock_position],
                    np.nanmean(values[position - 4 : position + 1, stock_position])
                    if np.isfinite(values[position - 4 : position + 1, stock_position]).all()
                    else np.nan,
                    np.nanmean(values[position - 21 : position + 1, stock_position])
                    if np.isfinite(values[position - 21 : position + 1, stock_position]).all()
                    else np.nan,
                ]
            )
        origins.append(calendar_dates[position])
        targets.append(calendar_dates[position + 1])
        feature_rows.append(row)
        response_rows.append(values[position + 1])
    features = pd.DataFrame(
        feature_rows,
        index=pd.DatetimeIndex(origins, name="forecast_origin"),
        columns=pd.MultiIndex.from_tuples(columns, names=["stock", "horizon"]),
    )
    response = pd.DataFrame(
        response_rows,
        index=features.index,
        columns=observed.columns,
    )
    return CalendarHARDesign(
        calendar_dates=calendar_dates,
        origins=features.index,
        targets=pd.DatetimeIndex(targets, name="target_date"),
        response=response,
        features=features,
        origin_positions=np.arange(first_position, len(calendar_dates) - 1, dtype=int),
    )


def _factor_assignment_table(
    diagnostics: pd.DataFrame,
    calendar: pd.DataFrame,
    spec_name: str,
) -> pd.DataFrame:
    """Expand factor score blocks into one row per exchange session."""

    calendar_value = normalize_calendar(calendar)
    dates = pd.DatetimeIndex(calendar_value["session_date"])
    rows: list[dict[str, object]] = []
    for row in diagnostics.to_dict("records"):
        start = pd.Timestamp(row["score_start"])
        end = pd.Timestamp(row["score_end"])
        selected = dates[(dates >= start) & (dates <= end)]
        for session_date in selected:
            training_end = pd.Timestamp(row["training_end"])
            rows.append(
                {
                    "spec_name": spec_name,
                    "session_date": session_date,
                    "factor_fit_id": f"{spec_name}:{row['update_id']}",
                    "factor_training_start": row["training_start"],
                    "factor_training_end": row["training_end"],
                    "factor_as_of": row["training_end"],
                    "factor_origin": row["factor_origin"],
                    "factor_window_sessions": row["window_sessions"],
                    "factor_update_step_sessions": row["update_step_sessions"],
                    "factor_components": 3,
                    "l1_rotation_status": row.get("l1_rotation_status", "unknown"),
                    "l1_optimizer_success_rate": row.get("l1_optimizer_success_rate", np.nan),
                    "training_cutoff_invariant": bool(training_end < session_date),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=["spec_name", "session_date", "factor_fit_id", "factor_training_start", "factor_training_end"]
        )
    return result.drop_duplicates(["spec_name", "session_date"], keep="last").sort_values("session_date")


def _rv_metadata(
    variance_result: RealizedVarianceResult,
    factor_metadata: pd.DataFrame,
    calendar: pd.DataFrame,
    spec_name: str,
    config: OOSConfig,
) -> pd.DataFrame:
    calendar_value = normalize_calendar(calendar)
    dates = pd.DataFrame({"session_date": pd.DatetimeIndex(calendar_value["session_date"])})
    interval = variance_result.interval_diagnostics.reset_index()
    interval["session_date"] = _session_date_index(interval["session_date"])
    metadata = dates.merge(interval, on="session_date", how="left")
    metadata = metadata.merge(
        factor_metadata.drop(columns=["spec_name"], errors="ignore"),
        on="session_date",
        how="left",
    )
    metadata["spec_name"] = spec_name
    daily = variance_result.daily_ivar.get("factor_adjusted", pd.DataFrame())
    available_dates = _session_date_index(daily.index) if not daily.empty else pd.DatetimeIndex([])
    metadata["rv_available"] = metadata["session_date"].isin(available_dates)
    metadata["target_quality_eligible"] = (
        metadata["rv_available"]
        & metadata["n_valid_intervals"].fillna(0).ge(config.target_min_valid_intervals)
        & metadata["valid_bar_fraction"].fillna(0).ge(config.target_min_valid_bar_fraction)
    )
    metadata["target_quality_reason"] = np.select(
        [
            ~metadata["rv_available"],
            metadata["n_valid_intervals"].fillna(0).lt(config.target_min_valid_intervals),
            metadata["valid_bar_fraction"].fillna(0).lt(config.target_min_valid_bar_fraction),
        ],
        ["outcome_unavailable", "too_few_valid_intervals", "insufficient_valid_bar_fraction"],
        default="eligible_by_predeclared_rule",
    )
    metadata["variance_floor"] = config.variance.variance_floor
    metadata["variance_floor_applied"] = False
    if not daily.empty:
        floor_dates = _session_date_index(daily.index)
        floor_values = daily.to_numpy(dtype=float)
        metadata.loc[metadata["session_date"].isin(floor_dates), "variance_floor_applied"] = (
            floor_values <= config.variance.variance_floor
        ).any(axis=1)
    return metadata.sort_values("session_date").reset_index(drop=True)


def build_factor_vintage_run(
    complete_return_panel: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    factor_window_sessions: int,
    config: OOSConfig = OOSConfig(),
    spec_name: str | None = None,
) -> FactorVintageRun:
    """Fit a causal factor process and construct its versioned RV vintages."""

    name = spec_name or f"factor_adjusted_{factor_window_sessions}"
    factor_config = FactorConfig(
        n_jobs=config.har.n_jobs,
        window_sessions=int(factor_window_sessions),
        update_step_sessions=config.factor_update_step,
        n_components=config.factor_components,
        l1_starts=config.factor_l1_starts,
        random_seed=config.factor_seed + int(factor_window_sessions),
    )
    factor_result = run_factor_adjustment(
        complete_return_panel,
        CORE_UNIVERSE,
        BENCHMARKS,
        factor_config,
        spec_name=name,
    )
    variance_result = aggregate_realized_variance(
        {
            "raw": factor_result.raw_returns,
            "benchmark_residual": factor_result.benchmark_residuals,
            "factor_adjusted": factor_result.factor_adjusted_residuals,
        },
        normalize_calendar(calendar),
        config.variance,
    )
    pca_factor_variance_result = aggregate_realized_variance(
        {"pca_factor": factor_result.pca_factor_scores},
        normalize_calendar(calendar),
        config.variance,
    )
    local_factor_variance_result = aggregate_realized_variance(
        {"local_factor": factor_result.local_factor_scores},
        normalize_calendar(calendar),
        config.variance,
    )
    factor_metadata = _factor_assignment_table(
        factor_result.diagnostics,
        calendar,
        name,
    )
    rv_metadata = _rv_metadata(variance_result, factor_metadata, calendar, name, config)
    daily_variance = variance_result.daily_ivar["factor_adjusted"].copy()
    daily_log_variance = variance_result.log_daily_ivar["factor_adjusted"].copy()
    benchmark_daily_variance = variance_result.daily_ivar["benchmark_residual"].copy()
    benchmark_daily_log_variance = variance_result.log_daily_ivar["benchmark_residual"].copy()
    pca_factor_daily_variance = pca_factor_variance_result.daily_ivar["pca_factor"].copy()
    pca_factor_daily_log_variance = pca_factor_variance_result.log_daily_ivar["pca_factor"].copy()
    local_factor_daily_variance = local_factor_variance_result.daily_ivar["local_factor"].copy()
    local_factor_daily_log_variance = local_factor_variance_result.log_daily_ivar["local_factor"].copy()
    aggregate_factor_daily_variance = pca_factor_daily_variance.sum(axis=1).to_frame(
        "COMMON"
    )
    aggregate_factor_daily_log_variance = np.log(
        aggregate_factor_daily_variance.clip(lower=config.variance.variance_floor)
    )
    for frame in (
        daily_variance,
        daily_log_variance,
        benchmark_daily_variance,
        benchmark_daily_log_variance,
        pca_factor_daily_variance,
        pca_factor_daily_log_variance,
        local_factor_daily_variance,
        local_factor_daily_log_variance,
        aggregate_factor_daily_variance,
        aggregate_factor_daily_log_variance,
    ):
        frame.index = _session_date_index(frame.index)
        frame.index.name = "session_date"
    return FactorVintageRun(
        spec_name=name,
        factor_window_sessions=int(factor_window_sessions),
        factor_result=factor_result,
        variance_result=variance_result,
        daily_log_variance=daily_log_variance,
        daily_variance=daily_variance,
        benchmark_daily_log_variance=benchmark_daily_log_variance,
        benchmark_daily_variance=benchmark_daily_variance,
        pca_factor_daily_log_variance=pca_factor_daily_log_variance,
        pca_factor_daily_variance=pca_factor_daily_variance,
        local_factor_daily_log_variance=local_factor_daily_log_variance,
        local_factor_daily_variance=local_factor_daily_variance,
        aggregate_factor_daily_log_variance=aggregate_factor_daily_log_variance,
        aggregate_factor_daily_variance=aggregate_factor_daily_variance,
        factor_metadata=factor_metadata,
        rv_metadata=rv_metadata,
    )


def serialize_config(config: OOSConfig) -> dict[str, object]:
    """Return the complete frozen configuration as JSON-compatible content."""

    value = asdict(config)
    value["har"] = asdict(config.har)
    value["variance"] = asdict(config.variance)
    return value


def forecast_checkpoint_specification(
    config: OOSConfig,
    *,
    har_window: str | int,
    mode: str,
    freeze_timestamp: str | None,
) -> dict[str, object]:
    """Return only settings that can change forecast/checkpoint contents.

    The complete protocol manifest intentionally contains operational and
    inference settings too.  Those settings are not a valid resume key:
    changing worker count, checkpoint cadence, HAC lag, or bootstrap controls
    cannot change an issued forecast.  Keeping a narrower signature makes
    checkpoints portable across machines without weakening statistical
    compatibility checks.
    """

    har = asdict(config.har)
    for name in (
        "n_jobs",
        "hac_lag",
        "bootstrap_repetitions",
        "bootstrap_block_lengths",
        "bootstrap_checkpoint_step",
        "random_seed",
    ):
        har.pop(name, None)
    return {
        "protocol_version": config.protocol_version,
        "mode": mode,
        "har_window": har_window,
        "min_valid_training_observations": config.min_valid_training_observations,
        "variance_floor": config.variance.variance_floor,
        "har": har,
        "freeze_timestamp": freeze_timestamp if mode == "prospective" else None,
    }


def deserialize_config(value: Mapping[str, object]) -> OOSConfig:
    """Reconstruct a frozen configuration from a stored manifest."""

    content = dict(value)
    har_content = dict(content.pop("har", {}))
    variance_content = dict(content.pop("variance", {}))
    tuple_fields = {
        "factor_windows",
        "confirmation_checkpoints",
        "loss_bootstrap_block_lengths",
        "edge_bootstrap_block_lengths",
    }
    for name in tuple_fields:
        if name in content:
            content[name] = tuple(content[name])
    allowed = {item.name for item in fields(OOSConfig)}
    outer = {name: item for name, item in content.items() if name in allowed}
    if har_content:
        outer["har"] = HARConfig(**har_content)
    if variance_content:
        outer["variance"] = RealizedVarianceConfig(**variance_content)
    return OOSConfig(**outer)


def build_protocol_manifest(
    config: OOSConfig,
    *,
    freeze_timestamp: str,
    last_data_examined: str | None,
    data_provenance: Mapping[str, object],
    code_provenance: Mapping[str, object] | None = None,
    mode: str,
    status: str,
    origin_calendar: Sequence[object] | None = None,
) -> dict[str, object]:
    """Build the machine-readable protocol/manifest required before confirmation."""

    config_value = serialize_config(config)
    protocol_hash = stable_hash(config_value)
    return {
        "protocol_version": config.protocol_version,
        "protocol_hash": protocol_hash,
        "freeze_timestamp_utc": freeze_timestamp,
        "mode": mode,
        "status": status,
        "last_data_examined": last_data_examined,
        "universe": list(CORE_UNIVERSE),
        "benchmarks": list(BENCHMARKS),
        "target_convention": "one-session-ahead factor-adjusted realized variance",
        "target_formula": "RV[i,d]=sum_b(sum_{t in valid five-minute bin b}u[i,t])^2",
        "feature_convention": "D=y[d], W=mean(y[d-4:d]), M=mean(y[d-21:d]) on exchange sessions",
        "preprocessing_rules": {
            "timezone": NY_TZ,
            "strict_consecutive_one_minute_returns": True,
            "non_overlapping_five_minute_bins": True,
            "missing_intervals": "exclude; no interpolation across gaps",
            "bad_sessions_excluded": [str(value) for value in BAD_SESSION_DATES],
            "target_min_valid_intervals": config.target_min_valid_intervals,
            "target_min_valid_bar_fraction": config.target_min_valid_bar_fraction,
        },
        "factor_schedule": {
            "dimension": config.factor_components,
            "window_sessions": config.factor_window_primary,
            "update_step_sessions": config.factor_update_step,
            "l1_rotation_starts": config.factor_l1_starts,
            "l1_seed": config.factor_seed,
            "l1_role": "interpretation only; PCA projector defines target residuals",
        },
        "har_schedule": {
            "window_primary": config.har_window_primary,
            "window_robustness": config.har_window_robustness,
            "min_training_observations": config.min_valid_training_observations,
            "coefficient_updates": "every forecast origin",
            "tuning_frequency": config.har.tuning_frequency,
            "chronological_cv_folds": config.har.cv_splits,
            "alpha_fractions": list(config.har.alpha_fractions),
            "absolute_alpha_rule": (
                "fraction times alpha_max recomputed within each CV fold and "
                "on the full training sample"
            ),
            "selection_rule": (
                "strongest penalty within one standard error of minimum "
                "chronological log-MSE"
            ),
        },
        "evaluation": {
            "primary_loss": "QLIKE(RV,RV_hat)=RV/RV_hat-log(RV/RV_hat)-1",
            "primary_comparison": f"{PRIMARY_NETWORK_MODEL} versus {OWN_MODEL} on matched dates/stocks",
            "persistence_baseline": True,
            "hac_primary_lag": config.har.hac_lag,
            "hac_sensitivity_lag": 20,
            "loss_bootstrap_repetitions": config.loss_bootstrap_repetitions,
            "loss_bootstrap_block_length_primary": 20,
            "loss_bootstrap_block_length_sensitivity": 5,
            "edge_bootstrap_repetitions": config.edge_bootstrap_repetitions,
            "confirmation_horizon_sessions": config.confirmation_horizon_sessions,
            "confirmation_checkpoints": list(config.confirmation_checkpoints),
        },
        "origin_calendar": list(origin_calendar or []),
        "data_provenance": dict(data_provenance),
        "code_provenance": dict(code_provenance or {}),
        "configuration": config_value,
    }


def _feature_matrix(
    design: CalendarHARDesign,
    target: str,
    stocks: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], list[tuple[str, str]]]:
    own_pairs = [(target, horizon) for horizon in HORIZONS]
    cross_pairs = [
        (source, horizon)
        for source in stocks
        if source != target
        for horizon in HORIZONS
    ]
    own = design.features.loc[:, own_pairs].to_numpy(dtype=float)
    cross = design.features.loc[:, cross_pairs].to_numpy(dtype=float)
    names = tuple(f"{source}:{horizon}" for source, horizon in cross_pairs)
    return own, cross, names, cross_pairs


def _target_training_mask(
    y: np.ndarray,
    own: np.ndarray,
    cross: np.ndarray,
    end: int,
) -> np.ndarray:
    return (
        np.isfinite(y[:end])
        & np.isfinite(own[:end]).all(axis=1)
        & np.isfinite(cross[:end]).all(axis=1)
    )


def _serialize_tuning(tuning: PenaltyTuning) -> dict[str, object]:
    return {
        "alpha_max": tuning.alpha_max,
        "min_fraction": tuning.min_fraction,
        "one_se_fraction": tuning.one_se_fraction,
        "min_alpha": tuning.min_alpha,
        "one_se_alpha": tuning.one_se_alpha,
        "cv_scores": tuning.cv_scores.to_dict("records"),
        "status": tuning.status,
        "n_valid_folds": tuning.n_valid_folds,
        "requested_folds": tuning.requested_folds,
        "fallback_reason": tuning.fallback_reason,
    }


def _deserialize_tuning(value: Mapping[str, object]) -> PenaltyTuning:
    return PenaltyTuning(
        alpha_max=float(value["alpha_max"]),
        min_fraction=float(value["min_fraction"]),
        one_se_fraction=float(value["one_se_fraction"]),
        min_alpha=float(value["min_alpha"]),
        one_se_alpha=float(value["one_se_alpha"]),
        cv_scores=pd.DataFrame(value.get("cv_scores", [])),
        status=str(value.get("status", "ok")),
        n_valid_folds=int(value.get("n_valid_folds", 0)),
        requested_folds=int(value.get("requested_folds", 0)),
        fallback_reason=str(value.get("fallback_reason", "")),
    )


def _empty_forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "protocol_version",
            "run_id",
            "spec_name",
            "model",
            "stock",
            "forecast_origin",
            "as_of",
            "target_date",
            "target_open",
            "target_outcome_available_at",
            "training_label_cutoff",
            "training_label_available_at",
            "factor_fit_id",
            "target_data_present_at_issue",
            "selected_penalty",
            "predicted_log_variance",
            "predicted_variance",
        ]
    )


def _write_forecast_checkpoint(
    directory: Path,
    *,
    forecasts: list[dict[str, object]],
    coefficients: list[dict[str, object]],
    edges: list[dict[str, object]],
    tuning_rows: list[dict[str, object]],
    origin_rows: list[dict[str, object]],
    state: Mapping[str, object],
) -> None:
    def checkpoint_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        time_columns = (
            "forecast_origin",
            "as_of",
            "target_date",
            "target_open",
            "target_outcome_available_at",
            "issued_at_utc",
            "training_label_cutoff",
            "training_label_available_at",
            "factor_training_end",
            "training_start",
            "training_end",
        )
        for column in time_columns:
            if column in frame.columns:
                frame[column] = frame[column].map(
                    lambda value: value.isoformat()
                    if isinstance(value, (pd.Timestamp, datetime))
                    else value
                ).astype("string")
        # Legacy CSV inference can turn a descriptive value such as the
        # rolling window label "252" into an integer.  New rows retain the
        # string representation, so normalize every remaining object column
        # before Arrow schema inference.
        for column in frame.columns:
            dtype = frame[column].dtype
            if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype):
                frame[column] = frame[column].astype("string")
        return frame

    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_frame(forecasts).to_parquet(directory / "forecast_ledger.parquet", index=False)
    checkpoint_frame(coefficients).to_parquet(directory / "har_coefficients.parquet", index=False)
    checkpoint_frame(edges).to_parquet(directory / "edge_history.parquet", index=False)
    checkpoint_frame(tuning_rows).to_parquet(directory / "tuning_history.parquet", index=False)
    checkpoint_frame(origin_rows).to_parquet(directory / "forecast_diagnostics.parquet", index=False)
    write_json(directory / "forecast_state.json", state)


@dataclass(frozen=True)
class _OOSOriginFitTask:
    """One independent target equation at one forecast origin."""

    y_train: np.ndarray
    own_train: np.ndarray
    cross_train: np.ndarray
    current_own: np.ndarray
    current_cross: np.ndarray
    feature_names: tuple[str, ...]
    cross_pairs: list[tuple[str, str]]
    tuning: PenaltyTuning
    common: dict[str, object]
    target: str
    origin_log: float
    config: OOSConfig


@dataclass(frozen=True)
class _OOSOriginFitResult:
    forecasts: list[dict[str, object]]
    coefficients: list[dict[str, object]]
    edges: list[dict[str, object]]
    diagnostic: dict[str, object]


def _fit_oos_origin_target(task: _OOSOriginFitTask) -> _OOSOriginFitResult:
    """Fit and format one target equation; safe for a process worker."""

    fit_one_se, fit_min, fit_own = fit_network_pair(
        task.y_train,
        task.own_train,
        task.cross_train,
        one_se_alpha=task.tuning.one_se_alpha,
        min_alpha=task.tuning.min_alpha,
        cross_feature_names=task.feature_names,
        config=task.config.har,
    )
    forecasts: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    model_fits = (
        (OWN_MODEL, fit_own, np.empty((1, 0))),
        (PRIMARY_NETWORK_MODEL, fit_one_se, task.current_cross),
        (NETWORK_MIN_MODEL, fit_min, task.current_cross),
    )
    for model_name, fit, prediction_cross in model_fits:
        predicted_log = float(_predict_fit(fit, task.current_own, prediction_cross)[0])
        predicted_variance, exponent_clipped = _safe_positive_variance(
            predicted_log,
            fit.smearing_factor,
            task.config.variance.variance_floor,
        )
        forecasts.append(
            {
                **task.common,
                "model": model_name,
                "selected_penalty": fit.selected_penalty,
                "predicted_log_variance": predicted_log,
                "predicted_variance": predicted_variance,
                "smearing_factor": fit.smearing_factor,
                "smearing_clipped_observations": fit.smearing_clipped_observations,
                "prediction_exponent_clipped": exponent_clipped,
                "design_condition_number": fit.condition_number,
                "column_norm_ratio": fit.column_norm_ratio,
                "lasso_converged": fit.lasso_converged,
                "lasso_iterations": fit.lasso_iterations,
                "lasso_kkt_max_violation": fit.lasso_kkt_max_violation,
            }
        )
        coefficients.append(
            {
                **task.common,
                "model": model_name,
                "predictor_stock": task.target,
                "horizon": "Intercept",
                "coefficient_original": fit.intercept,
                "coefficient_standardized": np.nan,
                "selected": True,
            }
        )
        for horizon, coefficient in zip(HORIZONS, fit.own_coefficients):
            coefficients.append(
                {
                    **task.common,
                    "model": model_name,
                    "predictor_stock": task.target,
                    "horizon": horizon,
                    "coefficient_original": float(coefficient),
                    "coefficient_standardized": np.nan,
                    "selected": True,
                }
            )
    persistence_variance, persistence_clipped = _safe_positive_variance(
        task.origin_log,
        1.0,
        task.config.variance.variance_floor,
    )
    forecasts.append(
        {
            **task.common,
            "model": PERSISTENCE_MODEL,
            "selected_penalty": np.nan,
            "predicted_log_variance": task.origin_log,
            "predicted_variance": persistence_variance,
            "smearing_factor": 1.0,
            "smearing_clipped_observations": 0,
            "prediction_exponent_clipped": persistence_clipped,
            "design_condition_number": np.nan,
            "column_norm_ratio": np.nan,
            "lasso_converged": True,
            "lasso_iterations": 0,
            "lasso_kkt_max_violation": 0.0,
        }
    )
    coefficients.append(
        {
            **task.common,
            "model": PERSISTENCE_MODEL,
            "predictor_stock": task.target,
            "horizon": "Persistence",
            "coefficient_original": np.nan,
            "coefficient_standardized": np.nan,
            "selected": True,
        }
    )
    for model_name, fit in (
        (PRIMARY_NETWORK_MODEL, fit_one_se),
        (NETWORK_MIN_MODEL, fit_min),
    ):
        for coefficient, standardized, (source, horizon) in zip(
            fit.network_coefficients_original,
            fit.network_coefficients_standardized,
            task.cross_pairs,
        ):
            edges.append(
                {
                    **task.common,
                    "model": model_name,
                    "source": source,
                    "target": task.target,
                    "edge_id": f"{source}->{task.target}",
                    "horizon": horizon,
                    "coefficient_original": float(coefficient),
                    "coefficient_standardized": float(standardized),
                    "selected": bool(
                        abs(coefficient) > task.config.har.coefficient_tolerance
                    ),
                    "sign": int(np.sign(coefficient)),
                    "selection_frequency_unit": "sequential_forecast_fit",
                }
            )
    diagnostic = {
        "spec_name": task.common["spec_name"],
        "forecast_origin": task.common["forecast_origin"],
        "target_date": task.common["target_date"],
        "stock": task.target,
        "n_train_observations": len(task.y_train),
        "status": "issued",
        "target_data_present_at_issue": False,
        "factor_fit_id": task.common.get("factor_fit_id", np.nan),
        "training_label_cutoff": task.common["training_label_cutoff"],
        "as_of": task.common["as_of"],
    }
    return _OOSOriginFitResult(forecasts, coefficients, edges, diagnostic)


def issue_target_free_forecasts(
    log_daily_variance: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    spec_name: str,
    config: OOSConfig = OOSConfig(),
    run_id: str = "uncommitted-forecast-run",
    factor_metadata: pd.DataFrame | None = None,
    har_window: str | int = "expanding",
    mode: str = "historical_pseudo_oos",
    freeze_timestamp: str | None = None,
    issuance_timestamp: str | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    stop_after_origins: int | None = None,
) -> ForecastRunResult:
    """Issue forecasts without reading the current target outcome.

    The current target row is deliberately never used in this function.  The
    response values used for fitting stop at the previous origin, while the
    current row contributes only its already-observed D/W/M features.  The
    returned ledger contains no realized target or loss columns.
    """

    if mode not in {"historical_pseudo_oos", "prospective"}:
        raise ValueError("mode must be 'historical_pseudo_oos' or 'prospective'.")
    if stop_after_origins is not None and stop_after_origins <= 0:
        raise ValueError("stop_after_origins must be positive when supplied.")
    calendar_value = normalize_calendar(calendar)
    design = build_calendar_har_features(log_daily_variance, calendar_value)
    stocks = list(log_daily_variance.columns)
    opens, closes = _calendar_time_maps(calendar_value)
    factor_lookup: dict[pd.Timestamp, dict[str, object]] = {}
    if factor_metadata is not None and not factor_metadata.empty:
        factor_value = factor_metadata.copy()
        factor_value["session_date"] = _session_date_index(factor_value["session_date"])
        for row in factor_value.to_dict("records"):
            factor_lookup[pd.Timestamp(row["session_date"])] = row

    protocol_hash = stable_hash(serialize_config(config))
    checkpoint_specification_hash = stable_hash(
        forecast_checkpoint_specification(
            config,
            har_window=har_window,
            mode=mode,
            freeze_timestamp=freeze_timestamp,
        )
    )
    forecasts: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    origin_rows: list[dict[str, object]] = []
    tuning_cache: dict[str, PenaltyTuning] = {}
    last_tuned_count: dict[str, int] = {}
    issued_counts: dict[str, int] = {stock: 0 for stock in stocks}
    start_row = 0
    completed_checkpoint_frames: dict[str, pd.DataFrame] = {}
    # Stage 15 keeps one immutable design array per target worker.  The
    # target-free protocol cannot reuse its whole-history worker unchanged
    # because it must checkpoint at completed origins, but it can retain the
    # same low-cost design precomputation.
    target_design: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], list[tuple[str, str]]]] = {}
    for target in stocks:
        own_all, cross_all, feature_names, cross_pairs = _feature_matrix(
            design, target, stocks
        )
        target_design[target] = (
            design.response[target].to_numpy(dtype=float),
            own_all,
            cross_all,
            feature_names,
            cross_pairs,
        )

    if checkpoint_dir is not None:
        state_path = Path(checkpoint_dir) / "forecast_state.json"
        if resume:
            if not state_path.exists():
                raise FileNotFoundError(f"Cannot resume without {state_path}.")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("protocol_version") != config.protocol_version:
                raise ValueError("Checkpoint protocol version is incompatible.")
            if state.get("spec_name") != spec_name or state.get("run_id") != run_id:
                raise ValueError("Checkpoint run or specification is incompatible.")
            stored_specification_hash = state.get("forecast_specification_hash")
            if (
                stored_specification_hash is not None
                and stored_specification_hash != checkpoint_specification_hash
            ):
                raise ValueError("Checkpoint forecast specification is incompatible.")
            start_row = int(state["next_row"])
            if start_row < 0 or start_row > len(design.origins):
                raise ValueError("Checkpoint next_row is outside the current design.")
            last_origin = state.get("last_completed_origin")
            if start_row and last_origin:
                last_date = pd.Timestamp(last_origin).normalize()
                if last_date != pd.Timestamp(design.origins[start_row - 1]).normalize():
                    raise ValueError("Checkpoint origin position is incompatible with the calendar.")
                source = log_daily_variance.copy()
                source.index = _session_date_index(source.index)
                source = source.loc[source.index <= last_date]
                if state.get("data_prefix_hash") != dataframe_hash(source):
                    raise ValueError("Checkpoint data prefix is incompatible with the supplied panel.")
                stored_calendar_hash = state.get("calendar_prefix_hash")
                if stored_calendar_hash is not None:
                    last_target = pd.Timestamp(design.targets[start_row - 1]).normalize()
                    calendar_prefix = calendar_value.loc[
                        calendar_value["session_date"] <= last_target,
                        ["session_date", "session_open", "session_close"],
                    ]
                    if stored_calendar_hash != dataframe_hash(calendar_prefix):
                        raise ValueError("Checkpoint calendar prefix is incompatible.")
            table_format = str(state.get("table_format", "csv"))
            suffix = ".parquet" if table_format == "parquet" else ".csv"
            for stem, target in (
                ("forecast_ledger", forecasts),
                ("har_coefficients", coefficient_rows),
                ("edge_history", edge_rows),
                ("tuning_history", tuning_rows),
                ("forecast_diagnostics", origin_rows),
            ):
                path = Path(checkpoint_dir) / f"{stem}{suffix}"
                if not path.exists():
                    raise FileNotFoundError(f"Checkpoint is incomplete: {path}.")
                if table_format == "csv":
                    if path.stat().st_size == 0:
                        continue
                    with path.open("rb") as handle:
                        if not handle.read(4_096).strip():
                            continue
                frame = (
                    pd.read_parquet(path)
                    if table_format == "parquet"
                    else pd.read_csv(path)
                )
                if start_row == len(design.origins):
                    completed_checkpoint_frames[stem] = frame
                else:
                    target.extend(frame.to_dict("records"))
            if start_row == len(design.origins):
                forecast_frame = completed_checkpoint_frames["forecast_ledger"]
                validate_forecast_ledger(forecast_frame)
                return ForecastRunResult(
                    forecasts=forecast_frame,
                    coefficients=completed_checkpoint_frames["har_coefficients"],
                    edge_history=completed_checkpoint_frames["edge_history"],
                    tuning_history=completed_checkpoint_frames["tuning_history"],
                    origin_diagnostics=completed_checkpoint_frames["forecast_diagnostics"],
                    status="complete",
                )
            tuning_cache = {
                stock: _deserialize_tuning(value)
                for stock, value in state.get("tuning_cache", {}).items()
            }
            last_tuned_count = {
                stock: int(value) for stock, value in state.get("last_tuned_count", {}).items()
            }
            issued_counts.update(
                {stock: int(value) for stock, value in state.get("issued_counts", {}).items()}
            )
        elif state_path.exists():
            raise FileExistsError(
                f"Published checkpoint exists at {state_path}; pass resume=True explicitly."
            )

    freeze = None
    if freeze_timestamp is not None:
        freeze = pd.Timestamp(freeze_timestamp)
        if freeze.tzinfo is None:
            freeze = freeze.tz_localize("UTC")
        else:
            freeze = freeze.tz_convert("UTC")
    if mode == "prospective" and freeze is None:
        raise ValueError("Prospective issuance requires a protocol freeze timestamp.")
    issued_at = pd.Timestamp(issuance_timestamp or utc_now_iso())
    if issued_at.tzinfo is None:
        issued_at = issued_at.tz_localize("UTC")
    else:
        issued_at = issued_at.tz_convert("UTC")
    if int(config.har.n_jobs) < 1:
        raise ValueError("n_jobs must be at least one.")
    executor = (
        ProcessPoolExecutor(max_workers=min(int(config.har.n_jobs), len(stocks)))
        if int(config.har.n_jobs) > 1 and len(stocks) > 1
        else None
    )

    processed_this_call = 0
    for row_number in range(start_row, len(design.origins)):
        origin_date = pd.Timestamp(design.origins[row_number])
        target_date = pd.Timestamp(design.targets[row_number])
        origin_as_of = closes.get(origin_date)
        target_open = opens.get(target_date)
        target_close = closes.get(target_date)
        if origin_as_of is None or target_open is None or target_close is None:
            origin_rows.append(
                {
                    "spec_name": spec_name,
                    "forecast_origin": origin_date,
                    "target_date": target_date,
                    "status": "not_issued_missing_calendar_timestamp",
                }
            )
            processed_this_call += 1
            continue
        if not origin_as_of < target_open:
            raise ValueError("Forecast clock violation: origin close is not before target open.")
        if mode == "prospective" and (freeze is None or target_open.tz_convert("UTC") <= freeze):
            origin_rows.append(
                {
                    "spec_name": spec_name,
                    "forecast_origin": origin_date,
                    "target_date": target_date,
                    "status": "not_issued_before_protocol_freeze",
                }
            )
            processed_this_call += 1
            continue
        if mode == "prospective" and issued_at >= target_open.tz_convert("UTC"):
            origin_rows.append(
                {
                    "spec_name": spec_name,
                    "forecast_origin": origin_date,
                    "target_date": target_date,
                    "status": "not_issued_target_already_opened",
                    "issued_at_utc": issued_at,
                }
            )
            processed_this_call += 1
            continue

        factor_row = factor_lookup.get(origin_date, {})
        fit_tasks: list[_OOSOriginFitTask] = []
        for target in stocks:
            y_all, own_all, cross_all, feature_names, cross_pairs = target_design[target]
            if mode == "prospective" and np.isfinite(y_all[row_number]):
                origin_rows.append(
                    {
                        "spec_name": spec_name,
                        "forecast_origin": origin_date,
                        "target_date": target_date,
                        "stock": target,
                        "status": "not_issued_target_already_observed",
                        "issued_at_utc": issued_at,
                    }
                )
                continue
            current_own = own_all[row_number : row_number + 1]
            current_cross = cross_all[row_number : row_number + 1]
            if not (
                np.isfinite(current_own).all()
                and np.isfinite(current_cross).all()
            ):
                origin_rows.append(
                    {
                        "spec_name": spec_name,
                        "forecast_origin": origin_date,
                        "target_date": target_date,
                        "stock": target,
                        "status": "not_issued_missing_origin_features",
                    }
                )
                continue
            train_start, train_end = _training_bounds(row_number, har_window)
            full_mask = _target_training_mask(y_all, own_all, cross_all, train_end)
            mask = full_mask[train_start:train_end]
            y_train = y_all[train_start:train_end][mask]
            own_train = own_all[train_start:train_end][mask]
            cross_train = cross_all[train_start:train_end][mask]
            if len(y_train) < config.min_valid_training_observations:
                origin_rows.append(
                    {
                        "spec_name": spec_name,
                        "forecast_origin": origin_date,
                        "target_date": target_date,
                        "stock": target,
                        "n_train_observations": len(y_train),
                        "status": "not_issued_insufficient_training_history",
                    }
                )
                continue

            issued_counts[target] += 1
            needs_tuning = target not in tuning_cache or (
                issued_counts[target] - last_tuned_count.get(target, -10**9)
                >= config.har.tuning_frequency
            )
            if needs_tuning:
                tuning_cache[target] = tune_network_penalty(
                    y_train,
                    own_train,
                    cross_train,
                    config=config.har,
                    cross_feature_names=feature_names,
                )
                last_tuned_count[target] = issued_counts[target]
                for row in tuning_cache[target].cv_scores.to_dict("records"):
                    tuning_rows.append(
                        {
                            **row,
                            "protocol_version": config.protocol_version,
                            "run_id": run_id,
                            "spec_name": spec_name,
                            "stock": target,
                            "forecast_origin": origin_date,
                            "training_start": design.origins[train_start],
                            "training_end": design.origins[train_end - 1],
                            "selected_one_se_fraction": tuning_cache[target].one_se_fraction,
                            "selected_min_loss_fraction": tuning_cache[target].min_fraction,
                            "alpha_max_full_sample": tuning_cache[target].alpha_max,
                            "cv_status": tuning_cache[target].status,
                            "cv_fallback_reason": tuning_cache[target].fallback_reason,
                            "n_valid_folds": tuning_cache[target].n_valid_folds,
                            "requested_folds": tuning_cache[target].requested_folds,
                        }
                    )

            tuning = tuning_cache[target]
            training_targets = design.targets[train_start:train_end][mask]
            training_label_cutoff = pd.Timestamp(training_targets.max())
            training_label_available_at = closes[training_label_cutoff]
            if not training_label_available_at <= origin_as_of < target_open:
                raise ValueError("Forecast information-clock invariant failed.")
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
                "training_label_cutoff": training_label_cutoff,
                "training_label_available_at": training_label_available_at,
                "factor_fit_id": factor_row.get("factor_fit_id", np.nan),
                "factor_training_end": factor_row.get("factor_training_end", np.nan),
                "target_data_present_at_issue": False,
                "n_train_observations": len(y_train),
                "cv_status": tuning.status,
                "cv_fallback_reason": tuning.fallback_reason,
                "alpha_max_full_sample": tuning.alpha_max,
                "factor_window_sessions": factor_row.get("factor_window_sessions", np.nan),
            }
            fit_tasks.append(
                _OOSOriginFitTask(
                    y_train=y_train,
                    own_train=own_train,
                    cross_train=cross_train,
                    current_own=current_own,
                    current_cross=current_cross,
                    feature_names=feature_names,
                    cross_pairs=cross_pairs,
                    tuning=tuning,
                    common=common,
                    target=target,
                    origin_log=float(current_own[0, 0]),
                    config=config,
                )
            )
        fit_results = (
            list(executor.map(_fit_oos_origin_target, fit_tasks, chunksize=1))
            if executor is not None and fit_tasks
            else [_fit_oos_origin_target(task) for task in fit_tasks]
        )
        for fit_result in fit_results:
            forecasts.extend(fit_result.forecasts)
            coefficient_rows.extend(fit_result.coefficients)
            edge_rows.extend(fit_result.edges)
            origin_rows.append(fit_result.diagnostic)
        processed_this_call += 1
        if checkpoint_dir is not None and (
            processed_this_call % max(config.checkpoint_frequency_origins, 1) == 0
            or row_number == len(design.origins) - 1
            or (stop_after_origins is not None and processed_this_call >= stop_after_origins)
        ):
            source = log_daily_variance.copy()
            source.index = _session_date_index(source.index)
            source = source.loc[source.index <= origin_date]
            calendar_prefix = calendar_value.loc[
                calendar_value["session_date"] <= target_date,
                ["session_date", "session_open", "session_close"],
            ]
            state = {
                "protocol_hash": protocol_hash,
                "forecast_specification_hash": checkpoint_specification_hash,
                "protocol_version": config.protocol_version,
                "run_id": run_id,
                "spec_name": spec_name,
                "next_row": row_number + 1,
                "last_completed_origin": origin_date,
                "data_prefix_hash": dataframe_hash(source),
                "calendar_prefix_hash": dataframe_hash(calendar_prefix),
                "table_format": "parquet",
                "tuning_cache": {stock: _serialize_tuning(value) for stock, value in tuning_cache.items()},
                "last_tuned_count": last_tuned_count,
                "issued_counts": issued_counts,
            }
            _write_forecast_checkpoint(
                Path(checkpoint_dir),
                forecasts=forecasts,
                coefficients=coefficient_rows,
                edges=edge_rows,
                tuning_rows=tuning_rows,
                origin_rows=origin_rows,
                state=state,
            )
            if len(design.origins) >= 200:
                print(
                    f"[forecast] {spec_name}: origin {row_number + 1}/{len(design.origins)} checkpointed",
                    flush=True,
                )
        if stop_after_origins is not None and processed_this_call >= stop_after_origins:
            break

    if executor is not None:
        executor.shutdown()
    status = "complete" if start_row + processed_this_call >= len(design.origins) else "checkpointed_partial"
    return ForecastRunResult(
        forecasts=pd.DataFrame(forecasts) if forecasts else _empty_forecast_frame(),
        coefficients=pd.DataFrame(coefficient_rows),
        edge_history=pd.DataFrame(edge_rows),
        tuning_history=pd.DataFrame(tuning_rows),
        origin_diagnostics=pd.DataFrame(origin_rows),
        status=status,
    )


FORECAST_KEY_COLUMNS = ("spec_name", "model", "stock", "forecast_origin", "target_date")
MODEL_KEY_COLUMNS = ("spec_name", "stock", "forecast_origin", "target_date")


def validate_forecast_ledger(forecasts: pd.DataFrame) -> None:
    """Validate an immutable forecast ledger and reject accidental target joins."""

    required = set(FORECAST_KEY_COLUMNS) | {
        "as_of",
        "predicted_log_variance",
        "predicted_variance",
        "target_data_present_at_issue",
    }
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Forecast ledger is missing columns: {sorted(missing)}")
    forbidden = {
        "actual_rv",
        "actual_log_variance",
        "qlike",
        "loss_difference",
    }
    present = forbidden.intersection(forecasts.columns)
    if present:
        raise ValueError(f"Forecast ledger must not contain scored outcome columns: {sorted(present)}")
    if forecasts.duplicated(list(FORECAST_KEY_COLUMNS)).any():
        raise ValueError("Forecast ledger contains duplicate forecast keys.")
    if not forecasts.empty and forecasts["target_data_present_at_issue"].astype(bool).any():
        raise ValueError("A forecast was marked as using target data at issuance.")
    if not forecasts.empty and (forecasts["predicted_variance"].astype(float) <= 0).any():
        raise ValueError("Forecast variances must be strictly positive.")


def validate_identical_model_keys(
    forecasts: pd.DataFrame,
    models: Sequence[str] = FORECAST_MODELS,
) -> pd.DataFrame:
    """Return model-key counts and fail if a model silently drops a key."""

    validate_forecast_ledger(forecasts)
    if forecasts.empty:
        return pd.DataFrame(columns=[*MODEL_KEY_COLUMNS, "models", "n_models"])
    requested = {str(model) for model in models}
    requested_order = tuple(sorted(requested))
    present = set(forecasts["model"].dropna().astype(str))
    missing_models = requested.difference(present)
    if missing_models:
        raise ValueError(f"Forecast ledger is missing required models: {sorted(missing_models)}")
    # Counting distinct labels is sufficient because duplicate forecast keys
    # were rejected above and the panel is restricted to the requested model
    # set.  Avoid aggregating Python sets: pandas 3 can coerce those objects to
    # ndarrays during boolean indexing, yielding an unhashable-index error.
    requested_label = "|".join(requested_order)
    model_labels = forecasts["model"].astype("string")
    requested_rows = model_labels.isin(requested_order)
    requested_forecasts = forecasts.loc[requested_rows].copy()
    requested_forecasts["model"] = model_labels.loc[requested_rows]
    rows = (
        requested_forecasts
        .groupby(list(MODEL_KEY_COLUMNS), dropna=False)["model"]
        .nunique(dropna=True)
        .rename("n_models")
        .reset_index()
    )
    if rows["n_models"].ne(len(requested)).any():
        raise ValueError("Models do not share identical forecast keys.")
    rows.insert(len(MODEL_KEY_COLUMNS), "models", requested_label)
    return rows


def _target_metadata_lookup(metadata: pd.DataFrame | None) -> pd.DataFrame:
    if metadata is None or metadata.empty:
        return pd.DataFrame(
            columns=["session_date", "target_quality_eligible", "target_quality_reason"]
        )
    value = metadata.copy()
    if "session_date" not in value.columns:
        raise ValueError("RV metadata must contain session_date.")
    value["session_date"] = _session_date_index(value["session_date"])
    return value.drop_duplicates("session_date", keep="last")


def _qlike_positive(actual_rv: float, predicted_rv: float) -> float:
    if actual_rv <= 0 or predicted_rv <= 0:
        return np.nan
    ratio = float(actual_rv / predicted_rv)
    return float(ratio - math.log(ratio) - 1.0)


def score_forecasts(
    forecasts: pd.DataFrame,
    realized_variance_by_spec: Mapping[str, pd.DataFrame],
    metadata_by_spec: Mapping[str, pd.DataFrame] | None = None,
    *,
    config: OOSConfig = OOSConfig(),
    models: Sequence[str] = FORECAST_MODELS,
) -> pd.DataFrame:
    """Join realized outcomes after issuance and retain every score status.

    Raw positive RV is used directly in QLIKE.  A non-positive target is
    recorded as unscorable rather than being silently replaced by a floor.
    """

    validate_identical_model_keys(forecasts, models)
    value = forecasts.copy()
    value["target_date"] = _session_date_index(value["target_date"])
    value["forecast_origin"] = _session_date_index(value["forecast_origin"])
    value["score_status"] = "outcome_unavailable"
    value["target_quality_eligible"] = False
    value["target_quality_reason"] = "outcome_unavailable"
    value["actual_rv"] = np.nan
    value["actual_log_rv"] = np.nan
    value["actual_rv_floor_applied"] = False
    value["qlike"] = np.nan
    value["log_mse"] = np.nan
    value["log_mae"] = np.nan
    for spec_name, group in value.groupby("spec_name", dropna=False):
        rv = realized_variance_by_spec.get(str(spec_name))
        if rv is None or rv.empty:
            continue
        if not isinstance(rv, pd.DataFrame):
            raise TypeError("Realized variance inputs must be DataFrames.")
        rv_value = rv.copy()
        rv_value.index = _session_date_index(rv_value.index)
        if rv_value.columns.has_duplicates:
            raise ValueError(f"Realized variance columns are duplicated for {spec_name}.")
        long = (
            rv_value.rename_axis("target_date")
            .reset_index()
            .melt(id_vars=["target_date"], var_name="stock", value_name="actual_rv")
        )
        long["target_date"] = _session_date_index(long["target_date"])
        metadata = _target_metadata_lookup((metadata_by_spec or {}).get(str(spec_name)))
        eligible = long.merge(
            metadata[["session_date", "target_quality_eligible", "target_quality_reason"]]
            if not metadata.empty
            else pd.DataFrame(columns=["session_date", "target_quality_eligible", "target_quality_reason"]),
            left_on="target_date",
            right_on="session_date",
            how="left",
        ).drop(columns=["session_date"], errors="ignore")
        eligible["target_quality_eligible"] = eligible["target_quality_eligible"].fillna(False).astype(bool)
        eligible["target_quality_reason"] = eligible["target_quality_reason"].fillna("no_quality_metadata")
        eligible["actual_rv_floor_applied"] = eligible["actual_rv"].fillna(np.nan).le(
            config.variance.variance_floor
        )
        joined = value.loc[value["spec_name"] == spec_name, ["target_date", "stock"]].merge(
            eligible,
            on=["target_date", "stock"],
            how="left",
        )
        rows_by_index = joined.index
        target_indices = value.index[value["spec_name"] == spec_name]
        value.loc[target_indices, "actual_rv"] = joined.loc[rows_by_index, "actual_rv"].to_numpy()
        value.loc[target_indices, "target_quality_eligible"] = joined.loc[
            rows_by_index, "target_quality_eligible"
        ].fillna(False).to_numpy()
        value.loc[target_indices, "target_quality_reason"] = joined.loc[
            rows_by_index, "target_quality_reason"
        ].fillna("outcome_unavailable").to_numpy()
        value.loc[target_indices, "actual_rv_floor_applied"] = joined.loc[
            rows_by_index, "actual_rv_floor_applied"
        ].fillna(False).to_numpy()
        actual = value.loc[target_indices, "actual_rv"].to_numpy(dtype=float)
        positive = np.isfinite(actual) & (actual > 0)
        quality = value.loc[target_indices, "target_quality_eligible"].to_numpy(dtype=bool)
        score_ok = positive & quality
        statuses = np.where(
            ~np.isfinite(actual),
            "outcome_unavailable",
            np.where(
                ~quality,
                value.loc[target_indices, "target_quality_reason"],
                np.where(~positive, "nonpositive_target", "scored"),
            ),
        )
        value.loc[target_indices, "score_status"] = statuses
        log_values = np.full(len(actual), np.nan)
        log_values[positive] = np.log(actual[positive])
        value.loc[target_indices, "actual_log_rv"] = log_values
        qlike = np.full(len(actual), np.nan)
        qlike[score_ok] = [
            _qlike_positive(
                float(actual[index]),
                float(value.loc[target_indices[index], "predicted_variance"]),
            )
            for index in np.flatnonzero(score_ok)
        ]
        value.loc[target_indices, "qlike"] = qlike
        value.loc[target_indices, "log_mse"] = (
            value.loc[target_indices, "predicted_log_variance"].to_numpy(dtype=float) - log_values
        ) ** 2
        value.loc[target_indices, "log_mae"] = np.abs(
            value.loc[target_indices, "predicted_log_variance"].to_numpy(dtype=float) - log_values
        )
    scored_key_check = value.drop(
        columns=[
            "actual_rv",
            "actual_log_rv",
            "actual_rv_floor_applied",
            "score_status",
            "target_quality_eligible",
            "target_quality_reason",
            "qlike",
            "log_mse",
            "log_mae",
        ],
        errors="ignore",
    )
    validate_identical_model_keys(scored_key_check, models)
    # A date/stock is comparison-eligible only if every declared model has an
    # eligible outcome.  This prevents a failing model from benefiting by
    # dropping difficult targets.
    scored_counts = (
        value[value["score_status"] == "scored"]
        .groupby(list(MODEL_KEY_COLUMNS), dropna=False)
        .size()
        .rename("n_models_scored")
        .reset_index()
    )
    value = value.merge(scored_counts, on=list(MODEL_KEY_COLUMNS), how="left")
    value["n_models_scored"] = value["n_models_scored"].fillna(0).astype(int)
    value["comparison_eligible"] = value["n_models_scored"].eq(len(tuple(models)))
    return value


def _bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    if finite.any():
        positions = np.flatnonzero(finite)
        order = positions[np.argsort(p[finite])]
        ordered = p[order]
        values = ordered * len(ordered) / np.arange(1, len(ordered) + 1)
        values = np.minimum.accumulate(values[::-1])[::-1]
        adjusted[order] = np.clip(values, 0.0, 1.0)
    return adjusted


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if len(finite):
        order = finite[np.argsort(p[finite])]
        values = (len(order) - np.arange(len(order))) * p[order]
        values = np.maximum.accumulate(values)
        adjusted[order] = np.clip(values, 0.0, 1.0)
    return adjusted


def paired_loss_differentials(
    scored: pd.DataFrame,
    *,
    network_model: str = PRIMARY_NETWORK_MODEL,
    own_model: str = OWN_MODEL,
) -> pd.DataFrame:
    """Return matched network-minus-own QLIKE/log-loss differences."""

    required = set(MODEL_KEY_COLUMNS) | {"model", "qlike", "log_mse", "log_mae", "comparison_eligible"}
    missing = required.difference(scored.columns)
    if missing:
        raise ValueError(f"Scored ledger is missing columns: {sorted(missing)}")
    value = scored[scored["comparison_eligible"]].copy()
    value = value[value["model"].isin([own_model, network_model])]
    index_columns = list(MODEL_KEY_COLUMNS)
    wide = value.pivot_table(
        index=index_columns,
        columns="model",
        values=["qlike", "log_mse", "log_mae"],
        aggfunc="first",
    )
    needed = [(metric, own_model) for metric in ("qlike", "log_mse", "log_mae")] + [
        (metric, network_model) for metric in ("qlike", "log_mse", "log_mae")
    ]
    for column in needed:
        if column not in wide.columns:
            raise ValueError(f"Matched loss table is missing {column}.")
    wide = wide.reset_index()
    wide["qlike_difference"] = wide[("qlike", network_model)] - wide[("qlike", own_model)]
    wide["log_mse_difference"] = wide[("log_mse", network_model)] - wide[("log_mse", own_model)]
    wide["log_mae_difference"] = wide[("log_mae", network_model)] - wide[("log_mae", own_model)]
    wide.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else str(column)
        for column in wide.columns
    ]
    return wide


def summarize_scored_forecasts(
    scored: pd.DataFrame,
    dimensions: Sequence[str] = ("spec_name", "model"),
) -> pd.DataFrame:
    """Summarize only realized, eligible observations."""

    value = scored[scored["score_status"] == "scored"].copy()
    required = {"qlike", "log_mse", "log_mae", *dimensions}
    missing = required.difference(value.columns)
    if missing:
        raise ValueError(f"Scored ledger is missing columns: {sorted(missing)}")
    if value.empty:
        return pd.DataFrame(columns=[*dimensions, "n_forecasts", "qlike", "log_mse", "log_mae"])
    return (
        value.groupby(list(dimensions), dropna=False)
        .agg(
            n_forecasts=("qlike", "size"),
            qlike=("qlike", "mean"),
            log_mse=("log_mse", "mean"),
            log_mae=("log_mae", "mean"),
            first_target_date=("target_date", "min"),
            last_target_date=("target_date", "max"),
        )
        .reset_index()
    )


def build_hac_inference(
    scored: pd.DataFrame,
    *,
    max_lags: Sequence[int] = (5, 20),
    network_model: str = PRIMARY_NETWORK_MODEL,
    own_model: str = OWN_MODEL,
) -> pd.DataFrame:
    """Aggregate the stock panel by date before HAC inference."""

    differences = paired_loss_differentials(
        scored,
        network_model=network_model,
        own_model=own_model,
    )
    if differences.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for lag in max_lags:
        for spec_name, group in differences.groupby("spec_name", dropna=False):
            date_panel = group.groupby("target_date", as_index=False)[
                ["qlike_difference", "log_mse_difference", "log_mae_difference"]
            ].mean()
            result = hac_mean_test(date_panel["qlike_difference"], max_lag=int(lag))
            rows.append(
                {
                    "spec_name": spec_name,
                    "level": "pooled_date_aggregated",
                    "stock": "ALL",
                    "lag": int(lag),
                    "n_dates": int(len(date_panel)),
                    "n_stock_date": int(len(group)),
                    "mean_difference": result["mean_difference"],
                    "hac_se": result["hac_se"],
                    "z": result["z"],
                    "p_value": result["p_value"],
                    "log_mse_difference": float(date_panel["log_mse_difference"].mean()),
                    "log_mae_difference": float(date_panel["log_mae_difference"].mean()),
                    "interpretation": (
                        "negative favors network HAR"
                        if result["mean_difference"] < 0
                        else "positive favors own HAR"
                    ),
                }
            )
            stock_rows: list[dict[str, object]] = []
            for stock, stock_group in group.groupby("stock"):
                stock_result = hac_mean_test(stock_group["qlike_difference"], max_lag=int(lag))
                stock_rows.append(
                    {
                        "spec_name": spec_name,
                        "level": "stock",
                        "stock": stock,
                        "lag": int(lag),
                        "n_dates": int(len(stock_group)),
                        "n_stock_date": int(len(stock_group)),
                        "mean_difference": stock_result["mean_difference"],
                        "hac_se": stock_result["hac_se"],
                        "z": stock_result["z"],
                        "p_value": stock_result["p_value"],
                        "log_mse_difference": float(stock_group["log_mse_difference"].mean()),
                        "log_mae_difference": float(stock_group["log_mae_difference"].mean()),
                        "interpretation": (
                            "negative favors network HAR"
                            if stock_result["mean_difference"] < 0
                            else "positive favors own HAR"
                        ),
                    }
                )
            bh = _bh_adjust([row["p_value"] for row in stock_rows])
            holm = _holm_adjust([row["p_value"] for row in stock_rows])
            for row, bh_value, holm_value in zip(stock_rows, bh, holm):
                row["bh_adjusted_p_value"] = (
                    float(bh_value) if np.isfinite(bh_value) else np.nan
                )
                row["holm_adjusted_p_value"] = (
                    float(holm_value) if np.isfinite(holm_value) else np.nan
                )
                rows.append(row)
    return pd.DataFrame(rows)


def moving_block_indices(
    n_dates: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one circular moving-block index vector."""

    if n_dates <= 0:
        raise ValueError("Cannot bootstrap an empty date series.")
    if block_length <= 0:
        raise ValueError("block_length must be positive.")
    starts = rng.integers(0, n_dates, size=int(np.ceil(n_dates / block_length)))
    blocks = [
        (start + np.arange(block_length, dtype=int)) % n_dates
        for start in starts
    ]
    return np.concatenate(blocks)[:n_dates]


def draw_shared_date_indices(
    n_dates: int,
    repetitions: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    """Draw date indices shared across every stock/equation in a replicate."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    rng = np.random.default_rng(seed)
    return np.vstack(
        [moving_block_indices(n_dates, block_length, rng) for _ in range(repetitions)]
    )


def bootstrap_loss_series(
    scored: pd.DataFrame,
    *,
    repetitions: int = 2_000,
    block_length: int = 20,
    seed: int = 16016,
    network_model: str = PRIMARY_NETWORK_MODEL,
    own_model: str = OWN_MODEL,
) -> pd.DataFrame:
    """Paired moving-block bootstrap of date-aggregated QLIKE differences.

    A single date-index vector is applied to the full stock panel in each
    replication.  The interval is percentile-based for the observed mean; the
    p-value uses the centered null distribution.
    """

    differences = paired_loss_differentials(
        scored,
        network_model=network_model,
        own_model=own_model,
    )
    if differences.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for spec_name, group in differences.groupby("spec_name", dropna=False):
        panel = group.pivot(index="target_date", columns="stock", values="qlike_difference").sort_index()
        panel = panel.dropna(axis=0, how="any")
        matrix = panel.to_numpy(dtype=float)
        if not len(matrix):
            continue
        date_mean = matrix.mean(axis=1)
        observed = float(date_mean.mean())
        indices = draw_shared_date_indices(len(matrix), repetitions, block_length, seed)
        bootstrap_means = matrix[indices].mean(axis=(1, 2))
        centered_date = date_mean - observed
        null_means = centered_date[indices].mean(axis=1)
        rows.append(
            {
                "spec_name": spec_name,
                "loss": "qlike_difference",
                "n_dates": int(len(matrix)),
                "n_stocks": int(matrix.shape[1]),
                "n_replications": int(repetitions),
                "block_length": int(block_length),
                "observed_mean_difference": observed,
                "bootstrap_ci_low": float(np.quantile(bootstrap_means, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap_means, 0.975)),
                "centered_null_p_value": float(np.mean(np.abs(null_means) >= abs(observed))),
                "shared_date_indices": True,
                "assumption": (
                    "weak dependence across adjacent date-level loss differentials; "
                    "stock panel aggregated within date"
                ),
            }
        )
    return pd.DataFrame(rows)


def _target_arrays_for_design(
    design: CalendarHARDesign,
    target: str,
    stocks: Sequence[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    list[tuple[str, str]],
]:
    own, cross, names, pairs = _feature_matrix(design, target, stocks)
    return design.response[target].to_numpy(dtype=float), own, cross, names, pairs


def conditional_edge_bootstrap(
    log_daily_variance: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    spec_name: str,
    config: OOSConfig = OOSConfig(),
    repetitions: int | None = None,
    block_lengths: Sequence[int] | None = None,
    checkpoint_step: int | None = None,
) -> pd.DataFrame:
    """Estimate conditional edge-selection frequency at fixed checkpoints.

    The factor/RV features and the one-SE penalty selected on the observed
    training sample are held fixed conceptually; only the training rows are
    resampled.  One shared date draw is reused across target equations at a
    checkpoint, so graph summaries retain cross-equation dependence.
    """

    design = build_calendar_har_features(log_daily_variance, calendar)
    stocks = list(log_daily_variance.columns)
    repetitions = (
        config.edge_bootstrap_repetitions
        if repetitions is None
        else int(repetitions)
    )
    block_lengths = tuple(
        config.edge_bootstrap_block_lengths
        if block_lengths is None
        else block_lengths
    )
    checkpoint_step = (
        config.edge_bootstrap_checkpoint_step
        if checkpoint_step is None
        else int(checkpoint_step)
    )
    first = max(config.min_valid_training_observations, 1)
    checkpoints = list(range(first, len(design.origins), max(checkpoint_step, 1)))
    if checkpoints and checkpoints[-1] != len(design.origins) - 1:
        checkpoints.append(len(design.origins) - 1)
    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        target_info: dict[
            str,
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                tuple[str, ...],
                list[tuple[str, str]],
                PenaltyTuning,
            ],
        ] = {}
        n_training_by_target: dict[str, int] = {}
        for target in stocks:
            y_all, own_all, cross_all, names, pairs = _target_arrays_for_design(
                design, target, stocks
            )
            mask = _target_training_mask(y_all, own_all, cross_all, checkpoint)
            y = y_all[:checkpoint][mask]
            own = own_all[:checkpoint][mask]
            cross = cross_all[:checkpoint][mask]
            if len(y) < config.min_valid_training_observations:
                continue
            tuning = tune_network_penalty(
                y,
                own,
                cross,
                config=config.har,
                cross_feature_names=names,
            )
            target_info[target] = (y, own, cross, names, pairs, tuning)
            n_training_by_target[target] = len(y)
        if not target_info:
            continue
        rng_seed = config.factor_seed + checkpoint + len(spec_name)
        for block_length in block_lengths:
            samples_by_rep = {
                rep: moving_block_indices(
                    max(n_training_by_target.values()),
                    int(block_length),
                    np.random.default_rng(rng_seed + rep),
                )
                for rep in range(repetitions)
            }
            # A separate index vector is required for each target if missing
            # rows make training samples differ, but the draw starts are shared
            # conceptually and the common prefix is used for joint diagnostics.
            selections: dict[tuple[str, str, str], list[bool]] = {}
            signs: dict[tuple[str, str, str], list[int]] = {}
            for rep in range(repetitions):
                for target, (y, own, cross, names, pairs, tuning) in target_info.items():
                    sample = samples_by_rep[rep] % len(y)
                    fit = fit_partialling_out(
                        y[sample],
                        own[sample],
                        cross[sample],
                        alpha=tuning.one_se_alpha,
                        cross_feature_names=names,
                        config=config.har,
                    )
                    for coefficient, (source, horizon) in zip(
                        fit.network_coefficients_original,
                        pairs,
                    ):
                        key = (source, target, horizon)
                        selections.setdefault(key, []).append(
                            bool(abs(coefficient) > config.har.coefficient_tolerance)
                        )
                        signs.setdefault(key, []).append(int(np.sign(coefficient)))
            for (source, target, horizon), selected in selections.items():
                selected_array = np.asarray(selected, dtype=bool)
                sign_array = np.asarray(signs[(source, target, horizon)], dtype=int)
                nonzero = sign_array[sign_array != 0]
                rows.append(
                    {
                        "protocol_version": config.protocol_version,
                        "spec_name": spec_name,
                        "checkpoint_date": design.origins[checkpoint - 1],
                        "training_start": design.origins[0],
                        "training_end": design.origins[checkpoint - 1],
                        "block_length": int(block_length),
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
                        "n_bootstrap": int(len(selected_array)),
                        "uncertainty_scope": "conditional_on_generated_features_and_fixed_observed_penalty",
                        "shared_date_draws_across_targets": True,
                    }
                )
    return pd.DataFrame(rows)
