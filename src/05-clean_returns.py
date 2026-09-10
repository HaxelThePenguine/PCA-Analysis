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

ensure_project_directories()

returns = pd.read_parquet(RETURN_MATRIX_FILE)
missing = pd.read_parquet(MISSING_MASK_FILE)


# ============================================================
# 1. REMOVE KNOWN BAD MARKET-DATA DAY
# ============================================================

bad_session = pd.Index(returns.index.date).isin(BAD_SESSION_DATES)
returns.loc[bad_session, :] = np.nan


# ============================================================
# 2. REMOVE RETURNS AFFECTED BY MISSING CANDLES
# ============================================================

# If candle t is missing:
#
# t     -> synthetic 0 return after forward fill
# t+1   -> contains movement accumulated since last real price
#
# Therefore both returns are contaminated.

missing = (
    missing
    .reindex(index=returns.index, columns=returns.columns)
    .fillna(False)
    .astype(bool)
)
contaminated = missing | missing.shift(1, fill_value=False)

returns_clean = returns.mask(contaminated)


# ============================================================
# 3. COMPLETE CROSS-SECTIONAL PANEL
# ============================================================

# PCA requires all securities observed at the same timestamp.
#
# We simply discard timestamps containing any NaN.

returns_complete = returns_clean.dropna(how="any")


# ============================================================
# REPORT
# ============================================================

print("\n=== CLEANING REPORT ===\n")

print(f"Original rows:       {len(returns):,}")
print(f"Clean complete rows: {len(returns_complete):,}")

removed = len(returns) - len(returns_complete)

print(f"Removed rows:        {removed:,}")
print(
    f"Retained:            "
    f"{100 * len(returns_complete) / len(returns):.2f}%"
)

print("\nMissing values after cleaning:")
print(returns_complete.isna().sum())


# ============================================================
# SAVE
# ============================================================

returns_clean.to_parquet(
    RETURN_MATRIX_CLEAN_FILE,
    compression="zstd",
)

returns_complete.to_parquet(
    RETURN_MATRIX_COMPLETE_FILE,
    compression="zstd",
)

contaminated.to_parquet(
    CONTAMINATED_MASK_FILE,
    compression="zstd",
)

print("\nSaved:")
print("return_matrix_clean.parquet")
print("return_matrix_complete.parquet")
print("contaminated_mask.parquet")
