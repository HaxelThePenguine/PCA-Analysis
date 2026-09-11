"""Charts and console output for missing data."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from utils.missing_data import affected_symbols, daily_missing_summary

DATES_TO_INSPECT = ("2023-06-05", "2023-03-13", "2025-09-12")


def print_timestamp_rows(rows: pd.DataFrame, *, compact: bool = False) -> None:
    """Print missing symbols for each timestamp in a report slice."""

    for timestamp, row in rows.iterrows():
        affected = ", ".join(affected_symbols(row))
        if compact:
            print(timestamp.strftime("%H:%M"), f"({int(row['n_missing'])})", affected)
        else:
            print(timestamp, f" | missing={row['n_missing']:2.0f}", " | ", affected)


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
