"""Shared clocks, hashing, calendar, and persistence helpers for OOS stages."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from config import BAD_SESSION_DATES, NY_TZ
from utils.preprocessing import clean_returns, compute_strict_intraday_returns


def utc_now_iso() -> str:
    """Return an explicit UTC timestamp without fractional seconds."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def stable_hash(value: object) -> str:
    """Hash JSON-serializable protocol content deterministically."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        default=_json_default,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_prefix_fingerprint(
    path: Path,
    prefix_bytes: int = 1_048_576,
) -> dict[str, object]:
    """Fingerprint a large file from metadata and a fixed-size prefix."""

    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(prefix_bytes))
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "prefix_bytes_hashed": int(prefix_bytes),
        "prefix_sha256": digest.hexdigest(),
    }


def file_sha256(path: Path) -> str:
    """Return the complete SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_hash(frame: pd.DataFrame | pd.Series) -> str:
    """Hash labels, dtypes, and values of a pandas object."""

    value = frame.to_frame() if isinstance(frame, pd.Series) else frame
    payload = {
        "columns": [str(column) for column in value.columns],
        "index": [str(item) for item in value.index],
        "dtypes": [str(dtype) for dtype in value.dtypes],
        "values_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(value, index=True)
            .to_numpy(dtype=np.uint64)
            .tobytes()
        ).hexdigest(),
    }
    return stable_hash(payload)


def session_date_index(values: Iterable[object]) -> pd.DatetimeIndex:
    """Normalize mixed ISO values to timezone-naive exchange dates."""

    dates = pd.DatetimeIndex(pd.to_datetime(values, format="mixed"))
    if dates.tz is not None:
        dates = dates.tz_convert(NY_TZ).tz_localize(None)
    else:
        dates = dates.tz_localize(None)
    return dates.normalize()


def _exchange_timestamps(values: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(values)
    except (TypeError, ValueError):
        # Pandas 3 rejects a Series mixing EST and EDT offsets. Parsing through
        # UTC preserves the represented instant across daylight-saving changes.
        parsed = pd.to_datetime(values, utc=True)
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(NY_TZ)
    return parsed.dt.tz_convert(NY_TZ)


def normalize_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Validate a market calendar and preserve exchange timestamps."""

    required = {
        "date",
        "session_open",
        "session_close",
        "open_minute",
        "close_minute",
        "expected_bars",
    }
    missing = required.difference(calendar.columns)
    if missing:
        raise ValueError(f"Market calendar is missing columns: {sorted(missing)}")
    result = calendar.copy()
    result["session_date"] = session_date_index(result["date"])
    result = result[~result["session_date"].isin(pd.DatetimeIndex(BAD_SESSION_DATES))]
    result = result.sort_values("session_date").drop_duplicates(
        "session_date",
        keep="first",
    )
    result["session_open"] = _exchange_timestamps(result["session_open"])
    result["session_close"] = _exchange_timestamps(result["session_close"])
    return result.reset_index(drop=True)


def strict_clean_return_panel(
    close_matrix: pd.DataFrame,
    missing_mask: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build strict one-minute returns and retain contamination masks."""

    returns = compute_strict_intraday_returns(close_matrix)
    aligned_missing = missing_mask.reindex(
        index=close_matrix.index,
        columns=close_matrix.columns,
    )
    if aligned_missing.isna().any().any():
        raise ValueError("The missingness mask does not align with the close matrix.")
    return clean_returns(returns, aligned_missing)


def training_bounds(
    row_number: int,
    har_window: str | int,
) -> tuple[int, int]:
    """Return the causal HAR slice ending immediately before an origin."""

    if har_window == "expanding":
        return 0, row_number
    train_length = int(har_window)
    if train_length <= 0:
        raise ValueError("har_window must be positive or 'expanding'.")
    return max(0, row_number - train_length), row_number


def calendar_time_maps(
    calendar: pd.DataFrame,
) -> tuple[dict[pd.Timestamp, pd.Timestamp], dict[pd.Timestamp, pd.Timestamp]]:
    """Map exchange dates to their session opens and closes."""

    value = normalize_calendar(calendar)
    opens = dict(zip(value["session_date"], value["session_open"]))
    closes = dict(zip(value["session_date"], value["session_close"]))
    return opens, closes


def safe_positive_variance(
    predicted_log_variance: float,
    smearing_factor: float,
    floor: float,
) -> tuple[float, bool]:
    """Back-transform a log forecast with bounded exponentiation."""

    clipped = float(np.clip(predicted_log_variance, -40.0, 40.0))
    value = max(float(np.exp(clipped) * smearing_factor), floor)
    return value, bool(clipped != predicted_log_variance)


def write_json(path: Path, value: Mapping[str, object]) -> None:
    """Write deterministic, human-readable UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, default=_json_default)
    path.write_text(content + "\n", encoding="utf-8")
