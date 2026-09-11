"""Console presentation of preprocessing and return-quality results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def print_raw_prices(prices: pd.DataFrame) -> None:
    print("\n=== RAW MATRIX ===")
    print("Rows:", len(prices))
    print("Missing values:")
    print(prices.isna().sum())


def print_clean_prices(prices, returns, missing) -> None:
    print("\n=== CLEAN MATRIX ===")
    print("Missing prices:", int(prices.isna().sum().sum()))
    print("Missing returns:", int(returns.isna().sum().sum()))
    print("Imputed observations:", int(missing.sum().sum()))


def print_return_diagnostics(returns, summary, extremes) -> None:
    print("\n=== RETURN SUMMARY ===\n")
    print(summary)
    print("\n=== LARGEST ABSOLUTE RETURNS ===\n")
    for (timestamp, symbol), _ in extremes.items():
        value = returns.loc[timestamp, symbol]
        print(
            timestamp,
            symbol,
            f"return={value:.6f}",
            f"pct={100 * (np.exp(value) - 1):.3f}%",
        )


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


def print_universe_summary(name: str, panel: pd.DataFrame) -> None:
    """Print shape and completeness for one saved universe."""

    print(f"\n{name}")
    print(f"Shape: {panel.shape}")
    print(f"Rows:  {len(panel):,}")
    print(f"NaN:   {int(panel.isna().sum().sum())}")
