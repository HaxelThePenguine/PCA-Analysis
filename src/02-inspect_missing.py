"""Inspect gaps in the raw minute-bar panel and save quality reports."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from config import (
    CALENDAR_FILE,
    COMMON_MISSING_GAPS_FILE,
    MISSING_MATRIX_FILE,
    NY_TZ,
    RAW_SYMBOL_DIR,
    SYMBOLS,
    ensure_project_directories,
)


DATES_TO_INSPECT = ("2023-06-05", "2023-03-13", "2025-09-12")
COMMON_GAP_MIN_SYMBOLS = 5
# Preserve the historical report schema, whose benchmark columns are XLF, SPY.
COMMON_GAPS_COLUMNS = (*SYMBOLS[:-2], SYMBOLS[-1], SYMBOLS[-2], "n_missing", "missing_pct")


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


def print_timestamp_rows(rows: pd.DataFrame, *, compact: bool = False) -> None:
    """Print missing symbols for each timestamp in a report slice."""

    for timestamp, row in rows.iterrows():
        affected = ", ".join(affected_symbols(row))
        if compact:
            print(timestamp.strftime("%H:%M"), f"({int(row['n_missing'])})", affected)
        else:
            print(timestamp, f" | missing={row['n_missing']:2.0f}", " | ", affected)


def daily_missing_summary(missing: pd.DataFrame) -> pd.DataFrame:
    """Aggregate missing observations by session."""

    return (
        missing["n_missing"]
        .groupby(missing.index.date)
        .agg(max_symbols_missing="max", total_missing_cells="sum")
        .sort_values("total_missing_cells", ascending=False)
    )


def report_missingness(
    missing: pd.DataFrame,
    common_gaps: pd.DataFrame,
    dates: Iterable[str] = DATES_TO_INSPECT,
) -> None:
    """Print the human-readable diagnostics saved by this stage."""

    print("\n=== COMMON MISSING TIMESTAMPS ===\n")
    print_timestamp_rows(common_gaps)

    print("\n=== WORST 50 MINUTES ===\n")
    print(
        missing[["n_missing", "missing_pct"]]
        .sort_values("n_missing", ascending=False)
        .head(50)
    )

    print("\n=== WORST DAYS ===\n")
    print(daily_missing_summary(missing).head(30))

    for date_to_inspect in dates:
        day = missing.loc[date_to_inspect]
        day = day[day["n_missing"] > 0]
        print(f"\n=== {date_to_inspect} ===\n")
        print_timestamp_rows(day, compact=True)


def main() -> None:
    """Build, report, and save the raw-data missingness matrix."""

    ensure_project_directories()
    calendar = pd.read_csv(
        CALENDAR_FILE,
        parse_dates=["session_open", "session_close"],
    )
    missing = build_missing_matrix(build_expected_index(calendar))
    common_gaps = missing[missing["n_missing"] >= COMMON_GAP_MIN_SYMBOLS].copy()

    report_missingness(missing, common_gaps)
    report_missing = missing.loc[:, list(COMMON_GAPS_COLUMNS)]
    report_missing.to_parquet(MISSING_MATRIX_FILE, compression="zstd")
    common_gaps.loc[:, list(COMMON_GAPS_COLUMNS)].to_csv(COMMON_MISSING_GAPS_FILE)
    print("\nReports saved.")


if __name__ == "__main__":
    main()
