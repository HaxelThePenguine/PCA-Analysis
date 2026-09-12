"""Reusable routines for preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import BAD_SESSION_DATES, NY_TZ, RAW_SYMBOL_DIR, SYMBOLS, SYSTEMIC_GAP
from utils.data import require_columns


def load_close_series(symbol: str) -> pd.Series:
    """Load one symbol's unique, ordered New York close-price series."""

    path = RAW_SYMBOL_DIR / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data file: {path}")

    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["ny_time"] = frame["timestamp"].dt.tz_convert(NY_TZ)
    return (
        frame.drop_duplicates("ny_time")
        .set_index("ny_time")
        .sort_index()["close"]
        .rename(symbol)
    )


def build_price_matrix() -> pd.DataFrame:
    """Align all symbols to SPY's observed timestamp index."""

    prices = pd.concat(
        {symbol: load_close_series(symbol) for symbol in SYMBOLS},
        axis=1,
        sort=False,
    ).sort_index()
    return prices.reindex(prices["SPY"].dropna().index)


def remove_systemic_gap(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop the one known market-wide gap configured for the dataset."""

    gap_date, gap_start, gap_end = SYSTEMIC_GAP
    in_gap = (
        (prices.index.date == gap_date)
        & (prices.index.time >= gap_start)
        & (prices.index.time <= gap_end)
    )
    return prices.loc[~in_gap].copy()


def compute_intraday_returns(
    prices: pd.DataFrame,
    *,
    require_consecutive: bool = False,
) -> pd.DataFrame:
    """Compute within-session log returns.

    The default retains the historical Stage 3 behavior.  The corrected OOS
    protocol passes ``require_consecutive=True`` so a row cannot silently
    represent a return spanning a missing timestamp (for example, the first
    row after the documented 2023-06-05 feed gap).
    """

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Intraday prices require a DatetimeIndex.")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Intraday prices require a unique sorted index.")
    if (prices <= 0).any().any():
        raise ValueError("Log returns require strictly positive prices.")

    session = pd.Series(prices.index.normalize(), index=prices.index)
    returns = np.log(prices).diff()
    same_session = session.eq(session.shift(1))
    returns.loc[~same_session] = np.nan
    if require_consecutive:
        elapsed = prices.index.to_series().diff()
        consecutive = elapsed.eq(pd.Timedelta(minutes=1))
        returns.loc[~(same_session & consecutive)] = np.nan
    return returns


def compute_strict_intraday_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Return one-minute log returns only for consecutive same-session bars."""

    return compute_intraday_returns(prices, require_consecutive=True)


def build_contaminated_mask(
    missing: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Mark returns at and immediately after every forward-filled candle."""

    aligned = (
        missing.reindex(index=returns.index, columns=returns.columns)
        .fillna(False)
        .astype(bool)
    )
    return aligned | aligned.shift(1, fill_value=False)


def clean_returns(
    returns: pd.DataFrame,
    missing: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the configured session and candle-contamination filters."""

    cleaned_input = returns.copy()
    bad_session = pd.Index(cleaned_input.index.date).isin(BAD_SESSION_DATES)
    cleaned_input.loc[bad_session, :] = np.nan

    contaminated = build_contaminated_mask(missing, cleaned_input)
    cleaned = cleaned_input.mask(contaminated)
    complete = cleaned.dropna(how="any")
    return cleaned, complete, contaminated


def summarize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Return the distribution statistics used by the inspection report."""

    return pd.DataFrame(
        {
            "mean": returns.mean(),
            "std": returns.std(),
            "min": returns.min(),
            "max": returns.max(),
            "p001": returns.quantile(0.001),
            "p999": returns.quantile(0.999),
        }
    )


def largest_absolute_returns(returns: pd.DataFrame, limit: int = 50) -> pd.Series:
    """Return the largest absolute values across timestamp-symbol cells."""

    return returns.stack().abs().sort_values(ascending=False).head(limit)


def build_complete_universe(
    returns: pd.DataFrame,
    symbols: tuple[str, ...],
) -> pd.DataFrame:
    """Select an ordered universe and retain only complete timestamps."""

    columns = require_columns(returns, symbols, context="Clean return matrix")
    return returns.loc[:, columns].dropna(how="any")
