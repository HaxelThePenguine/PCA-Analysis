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


ensure_project_directories()


# ============================================================
# LOAD CLOSE PRICES
# ============================================================

prices = {}

for symbol in SYMBOLS:

    input_file = RAW_SYMBOL_DIR / f"{symbol}.parquet"
    if not input_file.exists():
        raise FileNotFoundError(f"Missing raw data file: {input_file}")

    df = pd.read_parquet(
        input_file
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df["ny_time"] = (
        df["timestamp"]
        .dt
        .tz_convert(NY_TZ)
    )

    df = (
        df
        .drop_duplicates("ny_time")
        .set_index("ny_time")
        .sort_index()
    )

    prices[symbol] = df["close"]


# ============================================================
# BUILD COMMON MATRIX
# ============================================================

prices = pd.concat(
    prices,
    axis=1,
    sort=False,
).sort_index()

master_index = prices["SPY"].dropna().index
prices = prices.reindex(master_index)

print("\n=== RAW MATRIX ===")
print("Rows:", len(prices))
print("Missing values:")
print(prices.isna().sum())


# ============================================================
# REMOVE SYSTEMIC GAP
# ============================================================

gap_date, gap_start, gap_end = SYSTEMIC_GAP
systemic_gap = (
    (prices.index.date == gap_date)
    & (prices.index.time >= gap_start)
    & (prices.index.time <= gap_end)
)

prices = prices.loc[~systemic_gap].copy()


# ============================================================
# FORWARD FILL MISSING PRICES
# ============================================================

missing_mask = prices.isna()

prices_clean = prices.ffill()


# ============================================================
# RETURNS
# ============================================================

# Identifica la sessione NY
session = pd.Series(
    prices_clean.index.date,
    index=prices_clean.index,
)

log_prices = np.log(prices_clean)

returns = log_prices.diff()


# Non vogliamo trasformare overnight returns
# in falsi "1-minute returns".
new_session = session != session.shift(1)

returns.loc[new_session] = np.nan


# ============================================================
# FINAL CHECK
# ============================================================

print("\n=== CLEAN MATRIX ===")

print(
    "Missing prices:",
    int(prices_clean.isna().sum().sum())
)

print(
    "Missing returns:",
    int(returns.isna().sum().sum())
)

print(
    "Imputed observations:",
    int(missing_mask.sum().sum())
)


# ============================================================
# SAVE
# ============================================================

prices_clean.to_parquet(
    CLOSE_MATRIX_FILE,
    compression="zstd",
)

returns.to_parquet(
    RETURN_MATRIX_FILE,
    compression="zstd",
)

missing_mask.to_parquet(
    MISSING_MASK_FILE,
    compression="zstd",
)


print("\nSaved:")

print(
    CLOSE_MATRIX_FILE
)

print(
    RETURN_MATRIX_FILE
)

print(
    MISSING_MASK_FILE
)
