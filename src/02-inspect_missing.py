"""Inspect gaps in the raw minute-bar panel and save quality reports."""

from __future__ import annotations

import pandas as pd

from config import (
    CALENDAR_FILE,
    COMMON_MISSING_GAPS_FILE,
    MISSING_MATRIX_FILE,
    SYMBOLS,
    ensure_project_directories,
)
from reporting.missing_data import report_missingness
from utils.missing_data import build_expected_index, build_missing_matrix

COMMON_GAP_MIN_SYMBOLS = 5


COMMON_GAPS_COLUMNS = (
    *SYMBOLS[:-2],
    SYMBOLS[-1],
    SYMBOLS[-2],
    "n_missing",
    "missing_pct",
)


def main() -> None:
    """Build, report, and save the raw-data missingness matrix."""

    ensure_project_directories()
    calendar = pd.read_csv(
        CALENDAR_FILE,
        parse_dates=["session_open", "session_close"],
    )
    missing = build_missing_matrix(build_expected_index(calendar))
    common_gaps = missing[missing["n_missing"] >= COMMON_GAP_MIN_SYMBOLS].copy()

    report_missingness(missing, common_gaps)
    report_missing = missing.loc[:, list(COMMON_GAPS_COLUMNS)]
    report_missing.to_parquet(MISSING_MATRIX_FILE, compression="zstd")
    common_gaps.loc[:, list(COMMON_GAPS_COLUMNS)].to_csv(COMMON_MISSING_GAPS_FILE)
    print("\nReports saved.")


if __name__ == "__main__":
    main()
