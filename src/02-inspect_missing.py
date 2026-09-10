import pandas as pd

from config import (
    CALENDAR_FILE,
    COMMON_MISSING_GAPS_FILE,
    MISSING_MATRIX_FILE,
    NY_TZ,
    RAW_SYMBOL_DIR,
    SYMBOLS,
    ensure_project_directories,
)


ensure_project_directories()


# ============================================================
# CALENDAR
# ============================================================

calendar = pd.read_csv(
    CALENDAR_FILE,
    parse_dates=["session_open", "session_close"],
)

expected_timestamps = []


for _, row in calendar.iterrows():

    market_open = pd.Timestamp(row["session_open"])
    market_close = pd.Timestamp(row["session_close"])

    if market_open.tzinfo is None:
        market_open = market_open.tz_localize(NY_TZ)

    if market_close.tzinfo is None:
        market_close = market_close.tz_localize(NY_TZ)

    # Last bar = close - 1 minute.
    minutes = pd.date_range(
        start=market_open,
        end=market_close - pd.Timedelta(minutes=1),
        freq="1min",
    )

    expected_timestamps.extend(minutes)


expected_index = (
    pd.DatetimeIndex(
        pd.to_datetime(expected_timestamps, utc=True)
    )
    .tz_convert(NY_TZ)
    .rename("ny_time")
)


# ============================================================
# OBSERVED / MISSING MATRIX
# ============================================================

missing = pd.DataFrame(
    False,
    index=expected_index,
    columns=SYMBOLS,
)


for symbol in SYMBOLS:

    file = RAW_SYMBOL_DIR / f"{symbol}.parquet"

    if not file.exists():
        raise FileNotFoundError(f"Missing raw data file: {file}")

    df = pd.read_parquet(
        file,
        columns=["ny_time"],
    )

    observed = pd.DatetimeIndex(
        pd.to_datetime(df["ny_time"])
    )

    if observed.tz is None:
        observed = observed.tz_localize(NY_TZ)
    else:
        observed = observed.tz_convert(NY_TZ)

    missing[symbol] = ~missing.index.isin(observed)


# ============================================================
# NUMBER OF SYMBOLS MISSING EACH MINUTE
# ============================================================

missing["n_missing"] = (
    missing[list(SYMBOLS)]
    .sum(axis=1)
)

missing["missing_pct"] = (
    missing["n_missing"]
    / len(SYMBOLS)
    * 100
)


# ============================================================
# COMMON GAPS
# ============================================================

common_gaps = missing[
    missing["n_missing"] >= 5
].copy()

print("\n=== COMMON MISSING TIMESTAMPS ===\n")

for timestamp, row in common_gaps.iterrows():

    affected = [
        symbol
        for symbol in SYMBOLS
        if row[symbol]
    ]

    print(
        timestamp,
        f" | missing={row['n_missing']:2.0f}",
        " | ",
        ", ".join(affected),
    )


# ============================================================
# WORST TIMESTAMPS
# ============================================================

print("\n=== WORST 50 MINUTES ===\n")

print(
    missing[
        ["n_missing", "missing_pct"]
    ]
    .sort_values(
        "n_missing",
        ascending=False,
    )
    .head(50)
)


# ============================================================
# DAILY COMMON GAPS
# ============================================================

daily = (
    missing["n_missing"]
    .groupby(missing.index.date)
    .agg(
        max_symbols_missing="max",
        total_missing_cells="sum",
    )
    .sort_values(
        "total_missing_cells",
        ascending=False,
    )
)

print("\n=== WORST DAYS ===\n")

print(
    daily.head(30)
)


# ============================================================
# SPECIFIC DATE INSPECTION
# ============================================================

DATES_TO_INSPECT = [
    "2023-06-05",
    "2023-03-13",
    "2025-09-12",
]

for d in DATES_TO_INSPECT:

    day = missing.loc[d]

    day = day[
        day["n_missing"] > 0
    ]

    print(
        f"\n=== {d} ===\n"
    )

    for timestamp, row in day.iterrows():

        affected = [
            symbol
            for symbol in SYMBOLS
            if row[symbol]
        ]

        print(
            timestamp.strftime("%H:%M"),
            f"({int(row['n_missing'])})",
            ", ".join(affected),
        )


# ============================================================
# SAVE
# ============================================================

missing.to_parquet(
    MISSING_MATRIX_FILE,
    compression="zstd",
)

common_gaps.to_csv(
    COMMON_MISSING_GAPS_FILE,
)

print("\nReports saved.")
