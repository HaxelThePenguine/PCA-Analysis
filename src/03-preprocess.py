"""Build synchronized close-price and intraday log-return matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CLOSE_MATRIX_FILE,
    MISSING_MASK_FILE,
    NY_TZ,
    RAW_SYMBOL_DIR,
    RETURN_MATRIX_FILE,
    SYMBOLS,
    SYSTEMIC_GAP,
    ensure_project_directories,
)


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


def compute_intraday_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns without turning overnight moves into minute returns."""

    session = pd.Series(prices.index.date, index=prices.index)
    returns = np.log(prices).diff()
    returns.loc[session != session.shift(1)] = np.nan
    return returns


def main() -> None:
    """Create and persist the synchronized preprocessing outputs."""

    ensure_project_directories()
    prices = build_price_matrix()
    print("\n=== RAW MATRIX ===")
    print("Rows:", len(prices))
    print("Missing values:")
    print(prices.isna().sum())

    prices = remove_systemic_gap(prices)
    missing_mask = prices.isna()
    prices_clean = prices.ffill()
    returns = compute_intraday_returns(prices_clean)

    print("\n=== CLEAN MATRIX ===")
    print("Missing prices:", int(prices_clean.isna().sum().sum()))
    print("Missing returns:", int(returns.isna().sum().sum()))
    print("Imputed observations:", int(missing_mask.sum().sum()))

    prices_clean.to_parquet(CLOSE_MATRIX_FILE, compression="zstd")
    returns.to_parquet(RETURN_MATRIX_FILE, compression="zstd")
    missing_mask.to_parquet(MISSING_MASK_FILE, compression="zstd")

    print("\nSaved:")
    for path in (CLOSE_MATRIX_FILE, RETURN_MATRIX_FILE, MISSING_MASK_FILE):
        print(path)


if __name__ == "__main__":
    main()
