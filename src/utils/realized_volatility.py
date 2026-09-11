"""Session-safe five-minute realized variance construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from utils.data import validate_panel


@dataclass(frozen=True)
class RealizedVarianceConfig:
    """Numerical and calendar rules for intraday aggregation."""

    interval_minutes: int = 5
    variance_floor: float = 1e-16
    material_floor_multiplier: float = 100.0


@dataclass(frozen=True)
class RealizedVarianceResult:
    """Five-minute returns, daily IVAR controls, and data-quality diagnostics."""

    interval_returns: dict[str, pd.DataFrame]
    daily_ivar: dict[str, pd.DataFrame]
    log_daily_ivar: dict[str, pd.DataFrame]
    interval_diagnostics: pd.DataFrame
    floor_diagnostics: pd.DataFrame


def _calendar_rules(calendar: pd.DataFrame | None) -> dict[pd.Timestamp, tuple[int, int, int]]:
    """Map each session to open minute, close minute, and expected bar count."""

    if calendar is None:
        return {}
    required = {"date", "open_minute", "close_minute", "expected_bars"}
    missing = required.difference(calendar.columns)
    if missing:
        raise ValueError(f"Market calendar is missing columns: {sorted(missing)}")
    rules: dict[pd.Timestamp, tuple[int, int, int]] = {}
    for row in calendar.itertuples(index=False):
        date = pd.Timestamp(row.date)
        date = date.tz_localize(None) if date.tzinfo is not None else date
        date = date.normalize()
        rules[date] = (
            int(row.open_minute),
            int(row.close_minute),
            int(row.expected_bars),
        )
    return rules


def _validate_controls(controls: Mapping[str, pd.DataFrame]) -> tuple[pd.DatetimeIndex, list[str]]:
    if not controls:
        raise ValueError("At least one realized-variance control is required.")
    names = list(controls)
    first = controls[names[0]]
    validate_panel(first, context=f"Realized variance control {names[0]}", require_complete=True)
    for name in names[1:]:
        frame = controls[name]
        validate_panel(frame, context=f"Realized variance control {name}", require_complete=True)
        if not frame.index.equals(first.index):
            raise ValueError("Realized variance controls must share the same index.")
        if list(frame.columns) != list(first.columns):
            raise ValueError("Realized variance controls must share the same columns.")
    if not isinstance(first.index, pd.DatetimeIndex):
        raise TypeError("Realized variance controls require a DatetimeIndex.")
    return first.index, list(first.columns)


def _valid_interval_groups(
    index: pd.DatetimeIndex,
    calendar_rules: dict[pd.Timestamp, tuple[int, int, int]],
    interval_minutes: int,
) -> tuple[list[tuple[pd.Timestamp, int, np.ndarray]], pd.DataFrame]:
    """Find complete, non-overlapping clock-time intervals within sessions.

    The return at 09:30 belongs to the first interval.  Groups are anchored at
    the session open and are accepted only when all five minute labels are
    present exactly once, which prevents gaps from being silently bridged.
    """

    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive.")
    sessions = index.tz_localize(None).normalize() if index.tz is not None else index.normalize()
    minute_of_day = index.hour * 60 + index.minute
    rows: list[tuple[pd.Timestamp, int, np.ndarray]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for session in pd.DatetimeIndex(pd.unique(sessions)).sort_values():
        positions = np.flatnonzero(sessions == session)
        if session in calendar_rules:
            open_minute, close_minute, expected_bars = calendar_rules[session]
        else:
            observed = minute_of_day[positions]
            open_minute = int(observed.min())
            close_minute = int(observed.max()) + 1
            expected_bars = close_minute - open_minute
        session_minutes = minute_of_day[positions]
        in_session = (session_minutes >= open_minute) & (session_minutes < close_minute)
        positions = positions[in_session]
        session_minutes = session_minutes[in_session]
        candidate_count = int(np.ceil(max(expected_bars, 0) / interval_minutes))
        valid_count = 0
        invalid_count = 0
        for interval_number in range(candidate_count):
            start = open_minute + interval_number * interval_minutes
            expected = np.arange(start, min(start + interval_minutes, close_minute))
            selected = positions[(session_minutes >= start) & (session_minutes < start + interval_minutes)]
            observed = session_minutes[(session_minutes >= start) & (session_minutes < start + interval_minutes)]
            valid = (
                len(expected) == interval_minutes
                and len(selected) == interval_minutes
                and len(np.unique(observed)) == interval_minutes
                and np.array_equal(np.sort(observed), expected)
            )
            if valid:
                rows.append((session, interval_number, selected))
                valid_count += 1
            else:
                invalid_count += 1
        diagnostic_rows.append(
            {
                "session_date": session,
                "expected_bars": expected_bars,
                "observed_bars_in_session": int(len(positions)),
                "n_candidate_intervals": candidate_count,
                "n_valid_intervals": valid_count,
                "n_invalid_intervals": invalid_count,
                "valid_bar_fraction": float(valid_count * interval_minutes / expected_bars)
                if expected_bars
                else np.nan,
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows).set_index("session_date")
    diagnostics.index = pd.DatetimeIndex(diagnostics.index)
    return rows, diagnostics


def aggregate_realized_variance(
    controls: Mapping[str, pd.DataFrame],
    calendar: pd.DataFrame | None = None,
    config: RealizedVarianceConfig = RealizedVarianceConfig(),
) -> RealizedVarianceResult:
    """Aggregate aligned minute-return controls into daily five-minute IVAR.

    Each accepted interval is the sum of exactly ``interval_minutes`` returns
    inside one session.  Partial intervals and intervals containing gaps are
    excluded from every control so that comparisons use identical observations.
    """

    index, columns = _validate_controls(controls)
    rules = _calendar_rules(calendar)
    groups, interval_diagnostics = _valid_interval_groups(
        index,
        rules,
        config.interval_minutes,
    )
    interval_returns: dict[str, list[np.ndarray]] = {name: [] for name in controls}
    interval_index: list[tuple[pd.Timestamp, int]] = []
    for session, interval_number, positions in groups:
        interval_index.append((session, interval_number))
        for name, frame in controls.items():
            interval_returns[name].append(
                frame.iloc[positions].sum(axis=0).to_numpy(dtype=float)
            )
    multi_index = pd.MultiIndex.from_tuples(
        interval_index,
        names=["session_date", "interval_number"],
    )
    interval_frames = {
        name: pd.DataFrame(values, index=multi_index, columns=columns)
        for name, values in interval_returns.items()
    }
    daily_ivar = {
        name: frame.pow(2).groupby(level="session_date").sum()
        for name, frame in interval_frames.items()
    }
    for frame in daily_ivar.values():
        frame.index = pd.DatetimeIndex(frame.index)
        frame.index.name = "session_date"
    log_daily_ivar: dict[str, pd.DataFrame] = {}
    floor_rows: list[dict[str, object]] = []
    for name, frame in daily_ivar.items():
        adjusted = frame.clip(lower=config.variance_floor)
        log_daily_ivar[name] = np.log(adjusted)
        material_threshold = config.variance_floor * config.material_floor_multiplier
        for stock in frame.columns:
            ivar = frame[stock].to_numpy(dtype=float)
            floor_rows.append(
                {
                    "control": name,
                    "stock": stock,
                    "variance_floor": config.variance_floor,
                    "material_floor_threshold": material_threshold,
                    "n_days": int(len(ivar)),
                    "n_floor_applied": int(np.sum(ivar <= config.variance_floor)),
                    "n_material_floor_observations": int(np.sum(ivar <= material_threshold)),
                    "min_ivar": float(np.min(ivar)) if len(ivar) else np.nan,
                    "median_ivar": float(np.median(ivar)) if len(ivar) else np.nan,
                    "max_ivar": float(np.max(ivar)) if len(ivar) else np.nan,
                }
            )
    floor_diagnostics = pd.DataFrame(floor_rows)
    return RealizedVarianceResult(
        interval_returns=interval_frames,
        daily_ivar=daily_ivar,
        log_daily_ivar=log_daily_ivar,
        interval_diagnostics=interval_diagnostics,
        floor_diagnostics=floor_diagnostics,
    )


def log_realized_variance(
    daily_ivar: pd.DataFrame,
    config: RealizedVarianceConfig = RealizedVarianceConfig(),
) -> pd.DataFrame:
    """Apply the fixed documented floor and return log IVAR."""

    validate_panel(daily_ivar, context="Daily realized variance", require_complete=True)
    if (daily_ivar < 0).any().any():
        raise ValueError("Realized variance cannot be negative.")
    return np.log(daily_ivar.clip(lower=config.variance_floor))
