"""Session-based rolling-window helpers shared by numbered stages."""

from __future__ import annotations

import numpy as np
import pandas as pd


def session_index(index: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Return sorted normalized sessions and an integer session code per row."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Session indexing requires a DatetimeIndex.")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Session indexing requires a unique sorted index.")

    normalized = index.normalize()
    sessions = pd.DatetimeIndex(pd.unique(normalized)).sort_values()
    sessions.name = "session_date"
    codes = sessions.get_indexer(normalized)
    if (codes < 0).any():
        raise ValueError("Could not map every observation to a trading session.")
    return sessions, codes


def rolling_starts(
    n_sessions: int,
    window_size: int,
    step: int = 5,
) -> list[int]:
    """Return trailing-window starts and always include the final window."""

    if n_sessions < 0:
        raise ValueError("n_sessions must be non-negative.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")

    last_start = n_sessions - window_size
    if last_start < 0:
        return []
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def session_window_positions(
    codes: np.ndarray,
    start: int,
    window_size: int,
) -> np.ndarray:
    """Return row positions belonging to one inclusive trailing session window."""

    if start < 0 or window_size <= 0:
        raise ValueError("start must be non-negative and window_size must be positive.")
    end = start + window_size - 1
    positions = np.flatnonzero((codes >= start) & (codes <= end))
    if not len(positions):
        raise ValueError("The requested session window contains no observations.")
    return positions
