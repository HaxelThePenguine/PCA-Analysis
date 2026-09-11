"""Build synchronized close-price and intraday log-return matrices."""

from __future__ import annotations

from config import (
    CLOSE_MATRIX_FILE,
    MISSING_MASK_FILE,
    RETURN_MATRIX_FILE,
    ensure_project_directories,
)
from reporting.preprocessing import print_clean_prices, print_raw_prices
from utils.preprocessing import (
    build_price_matrix,
    compute_intraday_returns,
    remove_systemic_gap,
)


def main() -> None:
    """Create and persist the synchronized preprocessing outputs."""

    ensure_project_directories()
    prices = build_price_matrix()
    print_raw_prices(prices)

    prices = remove_systemic_gap(prices)
    missing_mask = prices.isna()
    prices_clean = prices.ffill()
    returns = compute_intraday_returns(prices_clean)

    print_clean_prices(prices_clean, returns, missing_mask)

    prices_clean.to_parquet(CLOSE_MATRIX_FILE, compression="zstd")
    returns.to_parquet(RETURN_MATRIX_FILE, compression="zstd")
    missing_mask.to_parquet(MISSING_MASK_FILE, compression="zstd")

    print("\nSaved:")
    for path in (CLOSE_MATRIX_FILE, RETURN_MATRIX_FILE, MISSING_MASK_FILE):
        print(path)


if __name__ == "__main__":
    main()
