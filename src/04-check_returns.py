"""Print distribution diagnostics and the largest absolute minute returns."""

from __future__ import annotations

import pandas as pd

from config import RETURN_MATRIX_FILE
from reporting.preprocessing import print_return_diagnostics
from utils.preprocessing import largest_absolute_returns, summarize_returns


def main() -> None:
    """Load the return matrix and print its diagnostics."""

    returns = pd.read_parquet(RETURN_MATRIX_FILE)
    print_return_diagnostics(
        returns, summarize_returns(returns), largest_absolute_returns(returns)
    )


if __name__ == "__main__":
    main()
