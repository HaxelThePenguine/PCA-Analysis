# Intraday Statistical Factor Extraction in US Financials

Research project on 1-minute SIP data for U.S. financial stocks. The research context and methodology are documented in [the long-form research note](<README.md — Intraday Statistical Factor Extraction in US Financials.md>).

## Current data layout

```text
alpaca_us_banks_1m/
├── raw/by_symbol/       immutable downloaded Parquet files
├── chunks/              monthly download chunks
├── metadata/            market calendar
├── reports/             data-quality reports
├── intermediate/        synchronized prices, returns, and missing mask
└── processed/           cleaned panels and CORE/FULL universes
```

The raw data is never overwritten by preprocessing. Generated Parquet files are also excluded from future Git commits by `.gitignore`.

## Pipeline order

Run these scripts from the project root:

```text
01-crwal.py             download/resume raw SIP data and build reports
02-inspect_missing.py   inspect the minute-by-symbol missing matrix
03-preprocess.py       build synchronized prices and within-session returns
04-check_returns.py    inspect return distributions and extremes
05-clean_returns.py    remove bad sessions and contaminated returns
06_save_universes.py   save the CORE and FULL research panels
```

The downloader requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. The remaining steps operate on files already present locally.

All shared paths, symbols, dates, and research universes live in `src/config.py`; individual scripts should not add their own copies of those settings.

## Next research step

The cleaned primary input for the next stage is:

```text
alpaca_us_banks_1m/processed/return_core.parquet
```

The next analysis is the transparent covariance-PCA versus correlation-PCA baseline described in the research note.
