import os
import time
from datetime import date, timedelta

import pandas as pd

from alpaca.data.enums import DataFeed, Adjustment
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

from config import (
    CALENDAR_FILE,
    CHUNKS_DIR,
    DATASET_DIR,
    END_DATE,
    MAX_RETRIES,
    METADATA_DIR,
    NY_TZ,
    RAW_SYMBOL_DIR,
    REPORTS_DIR,
    START_DATE,
    SYMBOLS,
    UTC_TZ,
    ensure_project_directories,
)


BASE_DIR = DATASET_DIR
SYMBOLS_DIR = RAW_SYMBOL_DIR
ensure_project_directories()


# ============================================================
# API KEYS
# ============================================================

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError(
        "ALPACA_API_KEY / ALPACA_SECRET_KEY not found."
    )


# ============================================================
# CLIENTS
# ============================================================

data_client = StockHistoricalDataClient(
    API_KEY,
    SECRET_KEY,
)

trading_client = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=True,
)


# ============================================================
# HELPERS
# ============================================================

def to_ny_timestamp(value):
    """
    Convert an Alpaca calendar timestamp
    to America/New_York.
    """

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize(NY_TZ)

    return ts.tz_convert(NY_TZ)


def month_chunks(start_date, end_date):
    """
    Produce intervalli mensili [start, end)
    """

    current = date(
        start_date.year,
        start_date.month,
        1,
    )

    final_exclusive = end_date + timedelta(days=1)

    while current < final_exclusive:

        if current.month == 12:
            next_month = date(
                current.year + 1,
                1,
                1,
            )
        else:
            next_month = date(
                current.year,
                current.month + 1,
                1,
            )

        chunk_start = max(current, start_date)
        chunk_end = min(next_month, final_exclusive)

        label = f"{current.year}-{current.month:02d}"

        yield label, chunk_start, chunk_end

        current = next_month


def date_to_utc(local_date):
    """
    New York midnight -> UTC.

    Avoids DST ambiguity.
    """

    ts = pd.Timestamp(local_date)

    ts = ts.tz_localize(NY_TZ)

    return ts.tz_convert(UTC_TZ).to_pydatetime()


# ============================================================
# MARKET CALENDAR
# ============================================================

print("\n=== MARKET CALENDAR ===")

calendar_request = GetCalendarRequest(
    start=START_DATE,
    end=END_DATE,
)

calendar = trading_client.get_calendar(
    calendar_request
)

calendar_rows = []

for session in calendar:

    session_open = to_ny_timestamp(
        session.open
    )

    session_close = to_ny_timestamp(
        session.close
    )

    open_minute = (
        session_open.hour * 60
        + session_open.minute
    )

    close_minute = (
        session_close.hour * 60
        + session_close.minute
    )

    expected_bars = (
        close_minute - open_minute
    )

    calendar_rows.append(
        {
            "date": session.date,
            "session_open": session_open,
            "session_close": session_close,
            "open_minute": open_minute,
            "close_minute": close_minute,
            "expected_bars": expected_bars,
        }
    )


calendar_df = pd.DataFrame(calendar_rows)

calendar_df.to_csv(
    CALENDAR_FILE,
    index=False,
)

print(
    f"Trading days: {len(calendar_df)}"
)

print(
    f"Expected minute-bars per symbol: "
    f"{calendar_df['expected_bars'].sum():,}"
)

early_closes = calendar_df[
    calendar_df["expected_bars"] < 390
]

if not early_closes.empty:

    print("\nEarly closes detected:")

    print(
        early_closes[
            [
                "date",
                "session_open",
                "session_close",
                "expected_bars",
            ]
        ].to_string(index=False)
    )


# Fast lookup for the intraday filter

calendar_filter = calendar_df[
    [
        "date",
        "open_minute",
        "close_minute",
        "expected_bars",
    ]
].copy()


# ============================================================
# DOWNLOAD MONTH BY MONTH
# ============================================================

print("\n=== DOWNLOAD ===")

for label, chunk_start, chunk_end in month_chunks(
    START_DATE,
    END_DATE,
):

    chunk_dir = CHUNKS_DIR / label

    success_file = chunk_dir / "_SUCCESS"

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    if success_file.exists():

        print(
            f"[SKIP] {label} already complete"
        )

        continue

    chunk_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\n[DOWNLOAD] {label}"
    )

    print(
        f"  {chunk_start} -> {chunk_end}"
    )

    request = StockBarsRequest(
        symbol_or_symbols=list(SYMBOLS),
        timeframe=TimeFrame.Minute,

        start=date_to_utc(
            chunk_start
        ),

        end=date_to_utc(
            chunk_end
        ),

        feed=DataFeed.SIP,

        # Keep the raw feed.
        # This is preferable when volumes are also being studied.
        adjustment=Adjustment.RAW,
    )


    # ========================================================
    # RETRY
    # ========================================================

    bars_response = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            bars_response = (
                data_client.get_stock_bars(
                    request
                )
            )

            break

        except Exception as error:

            print(
                f"  Attempt "
                f"{attempt}/{MAX_RETRIES}: "
                f"failed: {error}"
            )

            if attempt == MAX_RETRIES:
                raise

            wait_seconds = min(
                2 ** attempt,
                60,
            )

            print(
                f"  Retrying in "
                f"{wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )


    # ========================================================
    # DATAFRAME
    # ========================================================

    df = bars_response.df

    if df.empty:

        print(
            "  No data returned."
        )

        continue

    df = (
        df
        .reset_index()
        .copy()
    )


    # ========================================================
    # TIMEZONE
    # ========================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df["ny_time"] = (
        df["timestamp"]
        .dt
        .tz_convert(NY_TZ)
    )

    df["date"] = (
        df["ny_time"]
        .dt
        .date
    )

    df["minute_of_day"] = (
        df["ny_time"].dt.hour * 60
        + df["ny_time"].dt.minute
    )


    # ========================================================
    # MERGE WITH MARKET CALENDAR
    # ========================================================

    df = df.merge(
        calendar_filter,
        on="date",
        how="inner",
    )


    # ========================================================
    # REGULAR MARKET HOURS ONLY
    # ========================================================

    df = df[
        (
            df["minute_of_day"]
            >= df["open_minute"]
        )
        &
        (
            df["minute_of_day"]
            < df["close_minute"]
        )
    ].copy()


    # ========================================================
    # CLEANUP
    # ========================================================

    df = df[
        [
            "symbol",
            "timestamp",
            "ny_time",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
        ]
    ]

    df = (
        df
        .drop_duplicates(
            subset=[
                "symbol",
                "timestamp",
            ]
        )
        .sort_values(
            [
                "symbol",
                "timestamp",
            ]
        )
    )


    # ========================================================
    # SAVE ONE FILE PER SYMBOL / MONTH
    # ========================================================

    print(
                f"  RTH bars: {len(df):,}"
    )

    missing_symbols = []

    for symbol in SYMBOLS:

        symbol_df = df[
            df["symbol"] == symbol
        ].copy()

        if symbol_df.empty:

            print(
                f"  WARNING: {symbol} "
                f"0 bars"
            )

            missing_symbols.append(symbol)
            continue

        output_file = (
            chunk_dir
            / f"{symbol}.parquet"
        )

        symbol_df.to_parquet(
            output_file,
            index=False,
            compression="zstd",
        )

        print(
            f"  {symbol:<5} "
            f"{len(symbol_df):>7,}"
        )


    if missing_symbols:
        raise RuntimeError(
            f"Download incomplete for {label}; "
            "month not marked as complete. "
            f"Symbols without bars: {', '.join(missing_symbols)}"
        )

    # Mark success only after the month is complete for every symbol.

    success_file.touch()

    print(
        f"[OK] {label}"
    )


# ============================================================
# CONSOLIDATE BY SYMBOL
# ============================================================

print("\n=== CONSOLIDATION ===")

for symbol in SYMBOLS:

    print(
        f"\n{symbol}:",
        end=" ",
        flush=True,
    )

    files = sorted(
        CHUNKS_DIR.glob(
            f"*/{symbol}.parquet"
        )
    )

    if not files:

        print(
            "no files found"
        )

        continue

    frames = [
        pd.read_parquet(file)
        for file in files
    ]

    symbol_df = pd.concat(
        frames,
        ignore_index=True,
    )

    symbol_df["timestamp"] = pd.to_datetime(
        symbol_df["timestamp"],
        utc=True,
    )

    symbol_df = (
        symbol_df
        .drop_duplicates(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    output_file = (
        SYMBOLS_DIR
        / f"{symbol}.parquet"
    )

    symbol_df.to_parquet(
        output_file,
        index=False,
        compression="zstd",
    )

    print(
        f"{len(symbol_df):,} bars"
    )


# ============================================================
# QUALITY REPORT
# ============================================================

print("\n=== QUALITY REPORT ===")

expected = (
    calendar_df[
        [
            "date",
            "expected_bars",
        ]
    ]
    .copy()
)

expected["date"] = pd.to_datetime(
    expected["date"]
).dt.date


daily_reports = []


for symbol in SYMBOLS:

    file = (
        SYMBOLS_DIR
        / f"{symbol}.parquet"
    )

    if not file.exists():
        continue

    data = pd.read_parquet(
        file
    )

    data["date"] = pd.to_datetime(
        data["date"]
    ).dt.date

    observed = (
        data.groupby("date")
        .size()
        .rename("observed_bars")
        .reset_index()
    )

    daily = expected.merge(
        observed,
        on="date",
        how="left",
    )

    daily["observed_bars"] = (
        daily["observed_bars"]
        .fillna(0)
        .astype(int)
    )

    daily["missing_bars"] = (
        daily["expected_bars"]
        - daily["observed_bars"]
    )

    daily["coverage"] = (
        daily["observed_bars"]
        / daily["expected_bars"]
    )

    daily["symbol"] = symbol

    daily_reports.append(
        daily
    )


daily_quality = pd.concat(
    daily_reports,
    ignore_index=True,
)


daily_quality.to_csv(
    REPORTS_DIR
    / "daily_data_quality.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

quality_summary = (
    daily_quality
    .groupby("symbol")
    .agg(
        trading_days=(
            "date",
            "nunique",
        ),

        expected_bars=(
            "expected_bars",
            "sum",
        ),

        observed_bars=(
            "observed_bars",
            "sum",
        ),

        missing_bars=(
            "missing_bars",
            "sum",
        ),

        avg_daily_coverage=(
            "coverage",
            "mean",
        ),

        worst_daily_coverage=(
            "coverage",
            "min",
        ),
    )
)


quality_summary[
    "total_coverage"
] = (
    quality_summary[
        "observed_bars"
    ]
    /
    quality_summary[
        "expected_bars"
    ]
)


# ============================================================
# LIQUIDITY STATS
# ============================================================

liquidity_rows = []

for symbol in SYMBOLS:

    file = (
        SYMBOLS_DIR
        / f"{symbol}.parquet"
    )

    if not file.exists():
        continue

    data = pd.read_parquet(
        file,
        columns=[
            "volume",
            "trade_count",
        ],
    )

    liquidity_rows.append(
        {
            "symbol": symbol,

            "median_volume": (
                data["volume"]
                .median()
            ),

            "mean_volume": (
                data["volume"]
                .mean()
            ),

            "median_trade_count": (
                data["trade_count"]
                .median()
            ),

            "mean_trade_count": (
                data["trade_count"]
                .mean()
            ),
        }
    )


liquidity_df = (
    pd.DataFrame(
        liquidity_rows
    )
    .set_index("symbol")
)


quality_summary = (
    quality_summary
    .join(liquidity_df)
)


quality_summary[
    "total_coverage_pct"
] = (
    quality_summary[
        "total_coverage"
    ]
    * 100
)

quality_summary[
    "avg_daily_coverage_pct"
] = (
    quality_summary[
        "avg_daily_coverage"
    ]
    * 100
)

quality_summary[
    "worst_daily_coverage_pct"
] = (
    quality_summary[
        "worst_daily_coverage"
    ]
    * 100
)


quality_summary = (
    quality_summary
    .sort_values(
        "total_coverage_pct",
        ascending=False,
    )
)


quality_summary.to_csv(
    REPORTS_DIR
    / "data_quality_summary.csv"
)


# ============================================================
# DISPLAY FINAL REPORT
# ============================================================

columns = [
    "trading_days",
    "expected_bars",
    "observed_bars",
    "missing_bars",
    "total_coverage_pct",
    "worst_daily_coverage_pct",
    "median_volume",
    "median_trade_count",
]

print(
    "\n"
    + quality_summary[
        columns
    ]
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


# ============================================================
# FINAL INFO
# ============================================================

print(
    "\n===================================="
)

print(
    "DOWNLOAD COMPLETE"
)

print(
    "===================================="
)

print(
    f"\nDataset: {BASE_DIR.resolve()}"
)

print(
    "\nParquet files by ticker:"
)

for symbol in SYMBOLS:

    print(
        f"  {SYMBOLS_DIR / (symbol + '.parquet')}"
    )

print(
    "\nReports:"
)

print(
    f"  {REPORTS_DIR / 'data_quality_summary.csv'}"
)

print(
    f"  {REPORTS_DIR / 'daily_data_quality.csv'}"
)

print(
    f"  {METADATA_DIR / 'market_calendar.csv'}"
)
