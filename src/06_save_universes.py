import pandas as pd

from config import (
    CORE_UNIVERSE,
    FULL_UNIVERSE,
    RETURN_CORE_FILE,
    RETURN_FULL_FILE,
    RETURN_MATRIX_CLEAN_FILE,
    ensure_project_directories,
)


ensure_project_directories()


# ============================================================
# LOAD
# ============================================================

returns = pd.read_parquet(RETURN_MATRIX_CLEAN_FILE)


# ============================================================
# BUILD COMPLETE PANELS
# ============================================================

core = returns[list(CORE_UNIVERSE)].dropna(how="any")
full = returns[list(FULL_UNIVERSE)].dropna(how="any")


# ============================================================
# SAVE
# ============================================================

core.to_parquet(
    RETURN_CORE_FILE,
    compression="zstd",
)

full.to_parquet(
    RETURN_FULL_FILE,
    compression="zstd",
)


# ============================================================
# REPORT
# ============================================================

print("\n=== DATASETS SAVED ===\n")

print("CORE")
print(f"Shape: {core.shape}")
print(f"Rows:  {len(core):,}")
print(f"NaN:   {int(core.isna().sum().sum())}")

print("\nFULL")
print(f"Shape: {full.shape}")
print(f"Rows:  {len(full):,}")
print(f"NaN:   {int(full.isna().sum().sum())}")

print("\nFiles:")
print(RETURN_CORE_FILE)
print(RETURN_FULL_FILE)
