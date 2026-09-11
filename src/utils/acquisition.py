"""Reusable routines for acquisition."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

from config import (
    CHUNKS_DIR,
    END_DATE,
    MAX_RETRIES,
    NY_TZ,
    RAW_SYMBOL_DIR,
    START_DATE,
    SYMBOLS,
    UTC_TZ,
)

BAR_COLUMNS = [
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


def read_credentials() -> tuple[str, str]:
    """Read the Alpaca credentials required by the downloader."""

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not found.")
    return api_key, secret_key


def create_clients() -> tuple[StockHistoricalDataClient, TradingClient]:
    """Create authenticated historical-data and paper-trading clients."""

    api_key, secret_key = read_credentials()
    return (
        StockHistoricalDataClient(api_key, secret_key),
        TradingClient(api_key, secret_key, paper=True),
    )


def to_ny_timestamp(value: object) -> pd.Timestamp:
    """Convert an Alpaca calendar timestamp to New York time."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(NY_TZ)
    return timestamp.tz_convert(NY_TZ)


def month_chunks(
    start_date: date,
    end_date: date,
) -> Iterator[tuple[str, date, date]]:
    """Yield labeled monthly half-open intervals covering the date range."""

    current = date(start_date.year, start_date.month, 1)
    final_exclusive = end_date + timedelta(days=1)
    while current < final_exclusive:
        next_month = (
            date(current.year + 1, 1, 1)
            if current.month == 12
            else date(current.year, current.month + 1, 1)
        )
        yield (
            f"{current.year}-{current.month:02d}",
            max(current, start_date),
            min(next_month, final_exclusive),
        )
        current = next_month


def date_to_utc(local_date: date) -> datetime:
    """Convert New York midnight to an unambiguous UTC datetime."""

    return (
        pd.Timestamp(local_date).tz_localize(NY_TZ).tz_convert(UTC_TZ).to_pydatetime()
    )


def build_market_calendar(
    trading_client: TradingClient,
    start_date: date = START_DATE,
    end_date: date = END_DATE,
) -> pd.DataFrame:
    """Fetch and structure the market calendar used by every download filter."""

    request = GetCalendarRequest(start=start_date, end=end_date)
    rows = []
    for session in trading_client.get_calendar(request):
        session_open = to_ny_timestamp(session.open)
        session_close = to_ny_timestamp(session.close)
        open_minute = session_open.hour * 60 + session_open.minute
        close_minute = session_close.hour * 60 + session_close.minute
        rows.append(
            {
                "date": session.date,
                "session_open": session_open,
                "session_close": session_close,
                "open_minute": open_minute,
                "close_minute": close_minute,
                "expected_bars": close_minute - open_minute,
            }
        )
    return pd.DataFrame(rows)


def request_bars_with_retry(
    data_client: StockHistoricalDataClient,
    request: StockBarsRequest,
) -> Any:
    """Fetch one chunk with bounded exponential backoff."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return data_client.get_stock_bars(request)
        except Exception as error:
            print(f"  Attempt {attempt}/{MAX_RETRIES}: failed: {error}")
            if attempt == MAX_RETRIES:
                raise
            wait_seconds = min(2**attempt, 60)
            print(f"  Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)
    raise RuntimeError("Unreachable retry state.")


def regular_session_bars(
    bars: pd.DataFrame,
    calendar_filter: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize timestamps and retain unique regular-session minute bars."""

    frame = bars.reset_index().copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["ny_time"] = frame["timestamp"].dt.tz_convert(NY_TZ)
    frame["date"] = frame["ny_time"].dt.date
    frame["minute_of_day"] = frame["ny_time"].dt.hour * 60 + frame["ny_time"].dt.minute
    frame = frame.merge(calendar_filter, on="date", how="inner")
    in_session = (frame["minute_of_day"] >= frame["open_minute"]) & (
        frame["minute_of_day"] < frame["close_minute"]
    )
    return (
        frame.loc[in_session, BAR_COLUMNS]
        .drop_duplicates(subset=["symbol", "timestamp"])
        .sort_values(["symbol", "timestamp"])
    )


def save_chunk_by_symbol(
    bars: pd.DataFrame,
    chunk_dir: Path,
    symbols: Sequence[str] = SYMBOLS,
) -> None:
    """Persist one monthly Parquet file per symbol or fail atomically."""

    print(f"  RTH bars: {len(bars):,}")
    missing_symbols = []
    for symbol in symbols:
        symbol_bars = bars[bars["symbol"] == symbol].copy()
        if symbol_bars.empty:
            print(f"  WARNING: {symbol} 0 bars")
            missing_symbols.append(symbol)
            continue
        symbol_bars.to_parquet(
            chunk_dir / f"{symbol}.parquet",
            index=False,
            compression="zstd",
        )
        print(f"  {symbol:<5} {len(symbol_bars):>7,}")

    if missing_symbols:
        raise RuntimeError(
            "Download incomplete; month not marked as complete. "
            f"Symbols without bars: {', '.join(missing_symbols)}"
        )


def download_chunks(
    data_client: StockHistoricalDataClient,
    calendar: pd.DataFrame,
) -> None:
    """Download every incomplete monthly chunk and mark complete chunks."""

    calendar_filter = calendar[
        ["date", "open_minute", "close_minute", "expected_bars"]
    ].copy()
    print("\n=== DOWNLOAD ===")

    for label, chunk_start, chunk_end in month_chunks(START_DATE, END_DATE):
        chunk_dir = CHUNKS_DIR / label
        success_file = chunk_dir / "_SUCCESS"
        if success_file.exists():
            print(f"[SKIP] {label} already complete")
            continue

        chunk_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[DOWNLOAD] {label}")
        print(f"  {chunk_start} -> {chunk_end}")
        request = StockBarsRequest(
            symbol_or_symbols=list(SYMBOLS),
            timeframe=TimeFrame.Minute,
            start=date_to_utc(chunk_start),
            end=date_to_utc(chunk_end),
            feed=DataFeed.SIP,
            adjustment=Adjustment.RAW,
        )
        response = request_bars_with_retry(data_client, request)
        if response.df.empty:
            print("  No data returned.")
            continue

        bars = regular_session_bars(response.df, calendar_filter)
        try:
            save_chunk_by_symbol(bars, chunk_dir)
        except RuntimeError as error:
            raise RuntimeError(f"Download incomplete for {label}; {error}") from error
        success_file.touch()
        print(f"[OK] {label}")


def consolidate_symbol(symbol: str) -> pd.DataFrame | None:
    """Combine all monthly files for one symbol into a deduplicated history."""

    files = sorted(CHUNKS_DIR.glob(f"*/{symbol}.parquet"))
    if not files:
        return None

    frame = pd.concat(
        [pd.read_parquet(path) for path in files],
        ignore_index=True,
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return (
        frame.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def consolidate_all_symbols(symbols: Sequence[str] = SYMBOLS) -> None:
    """Write one complete Parquet history per symbol."""

    print("\n=== CONSOLIDATION ===")
    for symbol in symbols:
        print(f"\n{symbol}:", end=" ", flush=True)
        frame = consolidate_symbol(symbol)
        if frame is None:
            print("no files found")
            continue
        frame.to_parquet(
            RAW_SYMBOL_DIR / f"{symbol}.parquet",
            index=False,
            compression="zstd",
        )
        print(f"{len(frame):,} bars")


def build_daily_quality(
    calendar: pd.DataFrame,
    symbols: Sequence[str] = SYMBOLS,
) -> pd.DataFrame:
    """Measure expected versus observed bars for every symbol and session."""

    expected = calendar[["date", "expected_bars"]].copy()
    expected["date"] = pd.to_datetime(expected["date"]).dt.date
    reports = []

    for symbol in symbols:
        path = RAW_SYMBOL_DIR / f"{symbol}.parquet"
        if not path.exists():
            continue
        data = pd.read_parquet(path)
        data["date"] = pd.to_datetime(data["date"]).dt.date
        observed = data.groupby("date").size().rename("observed_bars").reset_index()
        daily = expected.merge(observed, on="date", how="left")
        daily["observed_bars"] = daily["observed_bars"].fillna(0).astype(int)
        daily["missing_bars"] = daily["expected_bars"] - daily["observed_bars"]
        daily["coverage"] = daily["observed_bars"] / daily["expected_bars"]
        daily["symbol"] = symbol
        reports.append(daily)

    if not reports:
        raise RuntimeError("No consolidated symbol data is available.")
    return pd.concat(reports, ignore_index=True)


def build_liquidity_summary(
    symbols: Sequence[str] = SYMBOLS,
) -> pd.DataFrame:
    """Aggregate minute-bar volume and trade-count statistics by symbol."""

    rows = []
    for symbol in symbols:
        path = RAW_SYMBOL_DIR / f"{symbol}.parquet"
        if not path.exists():
            continue
        data = pd.read_parquet(path, columns=["volume", "trade_count"])
        rows.append(
            {
                "symbol": symbol,
                "median_volume": data["volume"].median(),
                "mean_volume": data["volume"].mean(),
                "median_trade_count": data["trade_count"].median(),
                "mean_trade_count": data["trade_count"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("symbol")


def build_quality_summary(daily_quality: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily coverage and join per-symbol liquidity statistics."""

    summary = daily_quality.groupby("symbol").agg(
        trading_days=("date", "nunique"),
        expected_bars=("expected_bars", "sum"),
        observed_bars=("observed_bars", "sum"),
        missing_bars=("missing_bars", "sum"),
        avg_daily_coverage=("coverage", "mean"),
        worst_daily_coverage=("coverage", "min"),
    )
    summary["total_coverage"] = summary["observed_bars"] / summary["expected_bars"]
    summary = summary.join(build_liquidity_summary())
    for source, target in (
        ("total_coverage", "total_coverage_pct"),
        ("avg_daily_coverage", "avg_daily_coverage_pct"),
        ("worst_daily_coverage", "worst_daily_coverage_pct"),
    ):
        summary[target] = summary[source] * 100
    return summary.sort_values("total_coverage_pct", ascending=False)
