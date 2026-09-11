"""Create complete CORE and FULL analysis universes."""

from __future__ import annotations

import pandas as pd

from config import (
    CORE_UNIVERSE,
    FULL_UNIVERSE,
    RETURN_CORE_FILE,
    RETURN_FULL_FILE,
    RETURN_MATRIX_CLEAN_FILE,
    ensure_project_directories,
)
from reporting.preprocessing import print_universe_summary
from utils.preprocessing import build_complete_universe


def main() -> None:
    """Build and save both configured analysis universes."""

    ensure_project_directories()
    returns = pd.read_parquet(RETURN_MATRIX_CLEAN_FILE)
    core = build_complete_universe(returns, CORE_UNIVERSE)
    full = build_complete_universe(returns, FULL_UNIVERSE)

    core.to_parquet(RETURN_CORE_FILE, compression="zstd")
    full.to_parquet(RETURN_FULL_FILE, compression="zstd")

    print("\n=== DATASETS SAVED ===")
    print_universe_summary("CORE", core)
    print_universe_summary("FULL", full)
    print("\nFiles:")
    print(RETURN_CORE_FILE)
    print(RETURN_FULL_FILE)


if __name__ == "__main__":
    main()
