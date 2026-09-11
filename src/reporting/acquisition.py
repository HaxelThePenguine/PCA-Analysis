"""Charts and console output for acquisition."""

from __future__ import annotations

import pandas as pd

from config import DATASET_DIR, METADATA_DIR, RAW_SYMBOL_DIR, REPORTS_DIR, SYMBOLS

QUALITY_DISPLAY_COLUMNS = [
    "trading_days",
    "expected_bars",
    "observed_bars",
    "missing_bars",
    "total_coverage_pct",
    "worst_daily_coverage_pct",
    "median_volume",
    "median_trade_count",
]


def print_calendar_summary(calendar: pd.DataFrame) -> None:
    """Print session totals and any detected early closes."""

    print("\n=== MARKET CALENDAR ===")
    print(f"Trading days: {len(calendar)}")
    print(f"Expected minute-bars per symbol: {calendar['expected_bars'].sum():,}")

    early_closes = calendar[calendar["expected_bars"] < 390]
    if not early_closes.empty:
        print("\nEarly closes detected:")
        print(
            early_closes[
                ["date", "session_open", "session_close", "expected_bars"]
            ].to_string(index=False)
        )


def print_final_report(summary: pd.DataFrame) -> None:
    """Print the final coverage table and generated output paths."""

    print(
        "\n"
        + summary[QUALITY_DISPLAY_COLUMNS]
        .round(
            {
                "total_coverage_pct": 3,
                "worst_daily_coverage_pct": 2,
                "median_volume": 0,
                "median_trade_count": 0,
            }
        )
        .to_string()
    )
    print("\n====================================")
    print("DOWNLOAD COMPLETE")
    print("====================================")
    print(f"\nDataset: {DATASET_DIR.resolve()}")
    print("\nParquet files by ticker:")
    for symbol in SYMBOLS:
        print(f"  {RAW_SYMBOL_DIR / (symbol + '.parquet')}")
    print("\nReports:")
    print(f"  {REPORTS_DIR / 'data_quality_summary.csv'}")
    print(f"  {REPORTS_DIR / 'daily_data_quality.csv'}")
    print(f"  {METADATA_DIR / 'market_calendar.csv'}")
