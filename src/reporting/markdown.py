"""Shared Markdown rendering for report tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def markdown_table(
    frame: pd.DataFrame, *, max_rows: int | None = None,
    float_digits: int | None = None, empty: str = "", missing: str = "",
) -> str:
    """Render a compact, pipe-safe Markdown table."""
    if frame is None or frame.empty:
        return empty
    value = frame.head(max_rows) if max_rows is not None else frame

    def cell(item: object) -> str:
        if pd.isna(item):
            return missing
        if float_digits is not None and isinstance(item, (float, np.floating)):
            return f"{float(item):.{float_digits}f}"
        return str(item).replace("|", "\\|")

    rows = [[str(column) for column in value.columns], ["---"] * len(value.columns)]
    rows.extend([cell(item) for item in record] for record in value.itertuples(index=False, name=None))
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)
