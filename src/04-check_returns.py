"""Print distribution diagnostics and the largest absolute minute returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import RETURN_MATRIX_FILE


def summarize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Return the distribution statistics used by the inspection report."""

    return pd.DataFrame(
        {
            "mean": returns.mean(),
            "std": returns.std(),
            "min": returns.min(),
            "max": returns.max(),
            "p001": returns.quantile(0.001),
            "p999": returns.quantile(0.999),
        }
    )


def largest_absolute_returns(returns: pd.DataFrame, limit: int = 50) -> pd.Series:
    """Return the largest absolute values across timestamp-symbol cells."""

    return returns.stack().abs().sort_values(ascending=False).head(limit)


def main() -> None:
    """Load the return matrix and print its diagnostics."""

    returns = pd.read_parquet(RETURN_MATRIX_FILE)
    print("\n=== RETURN SUMMARY ===\n")
    print(summarize_returns(returns))
    print("\n=== LARGEST ABSOLUTE RETURNS ===\n")

    for (timestamp, symbol), _ in largest_absolute_returns(returns).items():
        value = returns.loc[timestamp, symbol]
        print(
            timestamp,
            symbol,
            f"return={value:.6f}",
            f"pct={100 * (np.exp(value) - 1):.3f}%",
        )


if __name__ == "__main__":
    main()
