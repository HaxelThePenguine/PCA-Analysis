"""Reusable routines for missing data."""

from __future__ import annotations

import pandas as pd

from config import NY_TZ, RAW_SYMBOL_DIR, SYMBOLS


def normalize_ny_timestamp(value: object) -> pd.Timestamp:
    """Return a timestamp localized or converted to New York time."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(NY_TZ)
    return timestamp.tz_convert(NY_TZ)


def build_expected_index(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    """Expand market sessions into the expected one-minute timestamp index."""

    session_minutes = []
    for row in calendar.itertuples(index=False):
        market_open = normalize_ny_timestamp(row.session_open)
        market_close = normalize_ny_timestamp(row.session_close)
        session_minutes.append(
            pd.date_range(
                start=market_open,
                end=market_close - pd.Timedelta(minutes=1),
                freq="1min",
            )
        )

    expected = session_minutes[0].append(session_minutes[1:])
    return pd.DatetimeIndex(
        pd.to_datetime(expected, utc=True),
        name="ny_time",
    ).tz_convert(NY_TZ)


def load_observed_index(symbol: str) -> pd.DatetimeIndex:
    """Load and normalize the observed timestamps for one symbol."""

    path = RAW_SYMBOL_DIR / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data file: {path}")

    frame = pd.read_parquet(path, columns=["ny_time"])
    observed = pd.DatetimeIndex(pd.to_datetime(frame["ny_time"]))
    if observed.tz is None:
        return observed.tz_localize(NY_TZ)
    return observed.tz_convert(NY_TZ)


def build_missing_matrix(expected_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the timestamp-by-symbol missing-observation matrix."""

    missing = pd.DataFrame(False, index=expected_index, columns=SYMBOLS)
    for symbol in SYMBOLS:
        missing[symbol] = ~expected_index.isin(load_observed_index(symbol))

    missing["n_missing"] = missing.loc[:, list(SYMBOLS)].sum(axis=1)
    missing["missing_pct"] = missing["n_missing"] / len(SYMBOLS) * 100
    return missing


def affected_symbols(row: pd.Series) -> list[str]:
    """Return the symbols marked missing in a matrix row."""

    return [symbol for symbol in SYMBOLS if row[symbol]]


def daily_missing_summary(missing: pd.DataFrame) -> pd.DataFrame:
    """Aggregate missing observations by session."""

    return (
        missing["n_missing"]
        .groupby(missing.index.date)
        .agg(max_symbols_missing="max", total_missing_cells="sum")
        .sort_values("total_missing_cells", ascending=False)
    )
