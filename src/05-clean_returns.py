"""Remove known bad sessions and returns contaminated by missing candles."""

from __future__ import annotations

import pandas as pd

from config import (
    CONTAMINATED_MASK_FILE,
    MISSING_MASK_FILE,
    RETURN_MATRIX_CLEAN_FILE,
    RETURN_MATRIX_COMPLETE_FILE,
    RETURN_MATRIX_FILE,
    ensure_project_directories,
)
from reporting.preprocessing import print_cleaning_report
from utils.preprocessing import clean_returns


def main() -> None:
    """Clean the raw return matrix and persist all masks and panels."""

    ensure_project_directories()
    returns = pd.read_parquet(RETURN_MATRIX_FILE)
    missing = pd.read_parquet(MISSING_MASK_FILE)
    cleaned, complete, contaminated = clean_returns(returns, missing)

    print_cleaning_report(returns, complete)
    cleaned.to_parquet(RETURN_MATRIX_CLEAN_FILE, compression="zstd")
    complete.to_parquet(RETURN_MATRIX_COMPLETE_FILE, compression="zstd")
    contaminated.to_parquet(CONTAMINATED_MASK_FILE, compression="zstd")

    print("\nSaved:")
    print("return_matrix_clean.parquet")
    print("return_matrix_complete.parquet")
    print("contaminated_mask.parquet")


if __name__ == "__main__":
    main()
