"""Shared data loading, validation, normalization, and output helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntradayNormalization:
    """Result of minute-of-day volatility normalization."""

    normalized_returns: pd.DataFrame
    volatility_by_symbol: pd.DataFrame
    observations_by_minute: pd.Series

    def summary(self) -> pd.DataFrame:
        """Cross-sectional minute-of-day volatility and observation counts."""
        profile = pd.DataFrame(
            {
                "mean_volatility": self.volatility_by_symbol.mean(axis=1),
                "min_volatility": self.volatility_by_symbol.min(axis=1),
                "max_volatility": self.volatility_by_symbol.max(axis=1),
                "observations": self.observations_by_minute,
            }
        )
        profile.index.name = "minute_from_open"
        return profile

    def unit_volatility_error(self) -> float:
        """Maximum deviation from unit volatility at each minute of day."""
        profile = self.normalized_returns.groupby(
            minute_from_open(self.normalized_returns.index)
        ).std(ddof=1)
        return float(np.max(np.abs(profile.to_numpy() - 1)))


def require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str = "Data frame",
) -> list[str]:
    """Validate and return an ordered column list."""

    ordered = list(columns)
    missing = [column for column in ordered if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing columns: {missing}")
    return ordered


def validate_panel(
    panel: pd.DataFrame,
    *,
    context: str,
    require_complete: bool = True,
) -> None:
    """Validate the structural invariants shared by analysis panels."""

    if panel.empty:
        raise ValueError(f"{context} is empty.")
    if panel.index.has_duplicates:
        raise ValueError(f"{context} contains duplicate index values.")
    if not panel.index.is_monotonic_increasing:
        raise ValueError(f"{context} index is not sorted.")
    if require_complete and panel.isna().any().any():
        raise ValueError(f"{context} contains NaN values.")
    if require_complete and not np.isfinite(panel.to_numpy(dtype=float)).all():
        raise ValueError(f"{context} contains non-finite values.")


def load_panel(
    path: Path,
    columns: Sequence[str],
    *,
    context: str,
    drop_incomplete: bool = False,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Load selected columns from Parquet and enforce panel invariants."""

    frame = pd.read_parquet(path)
    ordered = require_columns(frame, columns, context=context)
    panel = frame.loc[:, ordered]
    if drop_incomplete:
        panel = panel.dropna(how="any")
    validate_panel(panel, context=context, require_complete=require_complete)
    return panel


def minute_from_open(index: pd.DatetimeIndex) -> pd.Series:
    """Return the zero-based regular-session minute for every timestamp."""

    return pd.Series(
        index.hour * 60 + index.minute - (9 * 60 + 30),
        index=index,
        name="minute_from_open",
    )


def normalize_intraday_volatility(returns: pd.DataFrame) -> IntradayNormalization:
    """Scale returns by each symbol's full-sample minute volatility."""

    minute = minute_from_open(returns.index)
    volatility = returns.groupby(minute).std(ddof=1)
    observations = minute.value_counts().sort_index()

    scale = volatility.loc[minute.to_numpy()].copy()
    scale.index = returns.index
    if scale.isna().any().any() or (scale <= 0).any().any():
        raise ValueError("Invalid intraday volatility profile.")

    normalized = returns.divide(scale)
    if not np.isfinite(normalized.to_numpy()).all():
        raise ValueError("Intraday-normalized returns contain non-finite values.")

    return IntradayNormalization(
        normalized_returns=normalized,
        volatility_by_symbol=volatility,
        observations_by_minute=observations,
    )


def save_csv_tables(tables: Mapping[str, pd.DataFrame], directory: Path) -> None:
    """Save a named collection of indexed CSV tables."""

    for filename, table in tables.items():
        table.to_csv(directory / filename)


def load_benchmark_panel() -> pd.DataFrame:
    """Load complete, validated CORE and SPY/XLF observations."""
    from config import BENCHMARKS, CORE_UNIVERSE, RETURN_MATRIX_CLEAN_FILE

    return load_panel(
        RETURN_MATRIX_CLEAN_FILE,
        [*CORE_UNIVERSE, *BENCHMARKS],
        context="Complete benchmark panel",
        drop_incomplete=True,
    )
