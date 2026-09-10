"""Shared project configuration.

All paths are resolved from the project root so the scripts can be run from
the repository root, an IDE, or another working directory.
"""

from datetime import date, time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "alpaca_us_banks_1m"

CHUNKS_DIR = DATASET_DIR / "chunks"
RAW_DIR = DATASET_DIR / "raw"
RAW_SYMBOL_DIR = RAW_DIR / "by_symbol"
INTERMEDIATE_DIR = DATASET_DIR / "intermediate"
PROCESSED_DIR = DATASET_DIR / "processed"
REPORTS_DIR = DATASET_DIR / "reports"
METADATA_DIR = DATASET_DIR / "metadata"

BANKS = (
    "BAC", "COF", "JPM", "HBAN", "WFC", "USB",
    "RF", "SCHW", "C", "AXP", "TFC", "FITB",
    "KEY", "GS", "MS", "FHN", "CFG", "SYF",
)

BENCHMARKS = ("XLF", "SPY")
SYMBOLS = BANKS + BENCHMARKS

CORE_UNIVERSE = (
    "JPM", "BAC", "WFC", "C", "USB", "TFC",
    "KEY", "RF", "FITB", "CFG", "HBAN", "MS",
)
FULL_UNIVERSE = BANKS

START_DATE = date(2023, 1, 1)
END_DATE = date(2026, 9, 9)
BAD_SESSION_DATES = (date(2023, 1, 24),)
SYSTEMIC_GAP = (date(2023, 6, 5), time(9, 52), time(9, 55))

NY_TZ = "America/New_York"
UTC_TZ = "UTC"
MAX_RETRIES = 6


# Intermediate preprocessing outputs.
CLOSE_MATRIX_FILE = INTERMEDIATE_DIR / "close_matrix.parquet"
RETURN_MATRIX_FILE = INTERMEDIATE_DIR / "return_matrix.parquet"
MISSING_MASK_FILE = INTERMEDIATE_DIR / "missing_mask.parquet"

# Clean analysis outputs.
RETURN_MATRIX_CLEAN_FILE = PROCESSED_DIR / "return_matrix_clean.parquet"
RETURN_MATRIX_COMPLETE_FILE = PROCESSED_DIR / "return_matrix_complete.parquet"
CONTAMINATED_MASK_FILE = PROCESSED_DIR / "contaminated_mask.parquet"
RETURN_CORE_FILE = PROCESSED_DIR / "return_core.parquet"
RETURN_FULL_FILE = PROCESSED_DIR / "return_full.parquet"

# Data-quality reports.
CALENDAR_FILE = METADATA_DIR / "market_calendar.csv"
MISSING_MATRIX_FILE = REPORTS_DIR / "missing_matrix.parquet"
COMMON_MISSING_GAPS_FILE = REPORTS_DIR / "common_missing_gaps.csv"


def ensure_project_directories():
    """Create only the project folders needed by the pipeline."""

    for directory in (
        DATASET_DIR,
        CHUNKS_DIR,
        RAW_SYMBOL_DIR,
        INTERMEDIATE_DIR,
        PROCESSED_DIR,
        REPORTS_DIR,
        METADATA_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
