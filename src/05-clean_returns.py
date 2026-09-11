"""Remove known bad sessions and returns contaminated by missing candles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    BAD_SESSION_DATES,
    CONTAMINATED_MASK_FILE,
    MISSING_MASK_FILE,
    RETURN_MATRIX_CLEAN_FILE,
    RETURN_MATRIX_COMPLETE_FILE,
    RETURN_MATRIX_FILE,
    ensure_project_directories,
)


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


def print_cleaning_report(original: pd.DataFrame, complete: pd.DataFrame) -> None:
    """Print row retention and completeness diagnostics."""

    removed = len(original) - len(complete)
    print("\n=== CLEANING REPORT ===\n")
    print(f"Original rows:       {len(original):,}")
    print(f"Clean complete rows: {len(complete):,}")
    print(f"Removed rows:        {removed:,}")
    print(f"Retained:            {100 * len(complete) / len(original):.2f}%")
    print("\nMissing values after cleaning:")
    print(complete.isna().sum())


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
