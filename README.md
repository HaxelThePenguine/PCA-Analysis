# Intraday Statistical Factor Extraction in US Financials

## Why this project exists

Most PCA examples begin with a perfectly clean matrix, draw a scree plot, and stop there. This project is interested in what comes before and after that picture.

We start from one-minute SIP market data for a group of large U.S. financial stocks. The first job is to build a synchronized return panel without hiding the difficult parts of the data: missing bars, feed gaps, early closes, trading halts, and extreme market moves. Once that foundation is reliable, we use PCA and related methods to understand how much of the cross-sectional movement is genuinely common across the stocks.

The longer-term question is whether anything left after removing the broad common factors has a stable structure. If residual movements show persistence, mean reversion, or lead-lag relationships, they may deserve further research. That signal would still need to survive walk-forward testing, realistic transaction costs, and an honest out-of-sample evaluation before it could be considered useful.

## Questions we want to answer

The project is organized around a few practical research questions:

1. When these financial stocks move together, how much of that movement can be explained by one dominant common factor?
2. Does the factor structure become more concentrated during periods of market stress, such as the regional-bank crisis in March 2023?
3. Are the results materially different when we use covariance PCA, which preserves volatility differences, versus correlation PCA, which puts the securities on a comparable scale?
4. After removing broad market and financial-sector exposure through SPY and XLF, is there still a meaningful internal structure among the financial stocks?
5. Do the residual components contain any repeatable short-horizon behavior, or do they look like noise once data quality and multiple testing are taken seriously?

The project is descriptive first and predictive later. No residual pattern will be treated as an alpha signal until it has been tested out of sample and after trading frictions have been included.

## Data scope

Historical data comes from Alpaca Market Data using the SIP feed.

| Item | Choice |
| --- | --- |
| Frequency | 1-minute bars |
| Fields | timestamp, open, high, low, close, volume, trade_count, vwap |
| Period | 2023-01-01 through 2026-09-09 |
| Sessions | 924 regular U.S. trading sessions |
| Normal session | 09:30 to 16:00 America/New_York |
| Expected full-session bars | 390 per symbol, adjusted for early closes |

The market calendar supplied by Alpaca is used to handle early-close sessions rather than assuming that every day has exactly 390 observations.

### Instruments

The downloaded universe contains 18 financial stocks and two broad benchmarks:

```text
Financial stocks:
BAC, COF, JPM, HBAN, WFC, USB, RF, SCHW, C, AXP,
TFC, FITB, KEY, GS, MS, FHN, CFG, SYF

Benchmarks:
XLF, SPY
```

The analysis uses two explicit universes:

| Universe | Constituents | Purpose |
| --- | ---: | --- |
| CORE | 12 | Primary PCA specification with high retention and a coherent bank/financial cross-section |
| FULL | 18 | Robustness check that includes all downloaded financial stocks |

CORE contains `JPM`, `BAC`, `WFC`, `C`, `USB`, `TFC`, `KEY`, `RF`, `FITB`, `CFG`, `HBAN`, and `MS`.

FULL contains all 18 financial stocks listed above. SPY and XLF remain available as benchmarks and residualization factors, but are not part of either stock universe.

## Data-quality decisions

Data treatment is part of the research, not a detail to hide. The following decisions are recorded explicitly so that later PCA results can be reproduced and challenged.

### Common feed gap

Between 09:52 and 09:55 ET on 2023-06-05, 18 of the 20 instruments were missing at the same time. This is treated as a systemic market-data gap. Those timestamps are removed globally rather than forward-filled across the panel.

### Missing bar does not necessarily mean no trading

A security can have no OHLC bar for a minute even though transactions were present in the raw SIP data. This was verified for GS on 2025-09-12 at 09:50 ET, where many odd-lot trades were present.

Previous-tick sampling is used only to construct a synchronized price series. Returns directly affected by an imputed price are marked as contaminated and excluded from the complete PCA panel.

If a price is missing at time `t`, both the synthetic return at `t` and the return at `t+1` are excluded. The second return may contain the whole move accumulated since the last observed price.

### Known bad session

The full session on 2023-01-24 is removed from return analysis because of the NYSE opening-auction malfunction and subsequent trade cancellations. The large early-session moves from that day are not treated as ordinary observations.

### Stress periods and extreme returns

The March 2023 regional-bank crisis remains in the research data because it is economically meaningful for the questions we are asking. Large returns are not clipped or winsorized automatically. They are investigated first to determine whether they reflect real market events, data problems, or a security-specific halt.

## Data layout

Raw observations are stored as compressed Parquet files and are treated as immutable.

```text
alpaca_us_banks_1m/
├── raw/
│   └── by_symbol/              raw one-minute files, one per instrument
├── chunks/                     monthly download chunks and completion markers
├── metadata/
│   └── market_calendar.csv
├── reports/
│   ├── data_quality_summary.csv
│   ├── daily_data_quality.csv
│   ├── missing_matrix.parquet
│   └── common_missing_gaps.csv
├── intermediate/
│   ├── close_matrix.parquet
│   ├── return_matrix.parquet
│   └── missing_mask.parquet
└── processed/
    ├── return_matrix_clean.parquet
    ├── return_matrix_complete.parquet
    ├── contaminated_mask.parquet
    ├── return_core.parquet
    └── return_full.parquet
```

The raw and generated data folders are excluded from Git through `.gitignore`. This keeps the code repository small and avoids treating a large local research dataset as source code.

## Preprocessing pipeline

The pipeline follows a simple sequence:

```text
Raw SIP minute bars
        ↓
Market-calendar alignment
        ↓
Systemic-gap removal
        ↓
Missing-bar inspection
        ↓
Cross-sectional price synchronization
        ↓
Within-session log returns
        ↓
Contamination masking
        ↓
Complete CORE and FULL panels
        ↓
Baseline PCA
```

The scripts deliberately keep the stages separate. Each step reads the previous stage's output and writes a named artifact that can be inspected or reused later.

### One-minute returns

For security `i` and minute `t`, the return is:

$$
r_{i,t} = \log(P_{i,t}) - \log(P_{i,t-1})
$$

Returns are calculated within each trading session. The first minute of every session is set to missing so that the overnight close-to-open move is not mixed into a one-minute intraday return.

## PCA methodology

Let `r_t` be the vector of stock returns at minute `t`. PCA starts from the covariance matrix:

$$
\Sigma = Cov(r_t)
$$

$$
\Sigma v_k = \lambda_k v_k
$$

Here, `v_k` is the loading vector for component `k`, and `lambda_k` is the amount of variance associated with that component. The corresponding statistical factor return is:

$$
f_{k,t} = v_k' r_t
$$

### Covariance PCA

Covariance PCA works with centered returns and preserves the original volatility scale of each security. Higher-volatility stocks therefore have more influence on the estimated covariance structure.

### Correlation PCA

Correlation PCA first standardizes each security:

$$
z_{i,t} = \frac{r_{i,t} - \mu_i}{\sigma_i}
$$

This gives each stock comparable marginal volatility and makes the result more focused on co-movement than on differences in individual volatility. It is the main baseline specification because volatility levels differ materially across the universe.

The first baseline comparison will save eigenvalues, explained-variance ratios, cumulative explained variance, loadings for PC1 through PC3, the correlation matrix, and the associated plots.

## Market and sector residualization

After the raw-return baseline, broad market and financial-sector exposure will be removed using SPY and XLF:

$$
r_{i,t} = \alpha_i + \beta_{i,M} r_{SPY,t} + \beta_{i,F} r_{XLF,t} + \epsilon_{i,t}
$$

The residuals are the part of each stock's return that is not explained by those two benchmark returns. PCA on the residual matrix will then be compared with PCA on the original returns.

This comparison is intended to answer a specific question: does the financial-stock universe contain internal structure that is hidden by broad market and sector exposure?

## Research roadmap

### 1. Baseline PCA

Run covariance and correlation PCA on `return_core.parquet` before applying any intraday normalization. This gives us a transparent reference point and lets us understand the raw geometry of the data before adding more choices.

### 2. Intraday seasonality

Volatility is usually higher near the open and close. We will estimate volatility by minute of day for each security and repeat PCA on returns scaled by that intraday profile. The goal is to see whether the apparent factor structure is being driven mainly by the market clock.

### 3. Market and sector residuals

We will remove SPY and XLF exposure and compare the residual eigenstructure with the raw-return result. Any interpretation of a residual component will be based on its actual loadings and constituent composition, not on intuition alone.

### 4. Rolling and stress-regime analysis

Rolling PCA will track eigenvalues, PC1 explained variance, loading stability, and eigenvector rotation through time. Stress windows such as March 2023 will be examined separately to test whether correlation rises and effective cross-sectional dimensionality falls during crisis conditions.

When comparing eigenvectors across windows, the sign is normalized through an absolute dot product because the sign of an eigenvector is arbitrary.

$$
S_k(t,t+\Delta) = |v_k(t)' v_k(t+\Delta)|
$$

### 5. Covariance estimation and random-matrix diagnostics

Sample covariance will be compared with shrinkage estimators such as Ledoit-Wolf, and potentially with exponentially weighted covariance. Random Matrix Theory will be used as a diagnostic benchmark for separating strong empirical components from noise. The Marchenko-Pastur distribution will not be treated as literal truth because the returns are not iid Gaussian observations.

### 6. Residual dynamics and out-of-sample testing

Only after the factor structure is understood will we study autocorrelation, mean reversion, stationarity, lead-lag relationships, and short-horizon forecasting. Any apparent signal must be evaluated walk-forward, out of sample, and with realistic transaction costs.

## Robustness checks

The main comparisons are:

```text
CORE universe             vs FULL universe
Covariance PCA             vs Correlation PCA
Raw returns                vs Intraday-normalized returns
Raw returns                vs SPY/XLF residual returns
1-minute data              vs 5-minute aggregation
Sample covariance          vs Shrinkage covariance
All periods                vs Stress periods excluded
Full-quality names         vs Lower-coverage names such as GS
```

The purpose of these checks is not to accumulate techniques. It is to understand which conclusions survive reasonable changes in data treatment and model specification.

## Current status

Completed:

- Historical SIP data acquisition
- Official session-calendar handling, including early closes
- Monthly download chunking and resume support
- Per-symbol and per-day data-quality reports
- Minute-level missing-data matrix and systemic-gap detection
- Initial classification of security-specific missing bars and stress intervals
- Within-session log-return construction
- Contaminated-return masking
- Complete-panel construction
- CORE and FULL research datasets
- Baseline covariance and correlation PCA on the CORE universe
- Intraday-volatility profile and normalized PCA robustness check
- SPY/XLF residualization and PCA of the remaining CORE structure
- Initial code cleanup and local Git versioning

Next:

- Review the remaining outliers and invalid-value checks
- Compare residual structure across rolling and stress windows

Later:

- Rolling PCA and stress-regime comparison
- Shrinkage covariance
- Random-matrix diagnostics
- Residual dynamics
- Walk-forward testing and transaction-cost analysis

## Reproducibility

The downloader reads credentials from environment variables:

```text
ALPACA_API_KEY
ALPACA_SECRET_KEY
```

Install the Python dependencies listed in `requirements.txt`. The current scripts can then be run from the project root in this order:

```text
01-crwal.py             download or resume raw SIP data and build reports
02-inspect_missing.py   inspect the minute-by-symbol missing matrix
03-preprocess.py       build synchronized prices and returns
04-check_returns.py    inspect return distributions and extremes
05-clean_returns.py    remove bad sessions and contaminated returns
06_save_universes.py   save the CORE and FULL research panels
07_baseline_pca.py     run covariance and correlation PCA with baseline plots
08_intraday_normalization.py  estimate intraday volatility and repeat PCA
09_benchmark_residualization.py remove SPY/XLF exposure and run residual PCA
```

The downloader filename contains a historical typo (`crwal`). It is kept for compatibility with the existing workflow and can be renamed once any external run commands have been updated.

## Research principles

- Never overwrite raw Alpaca Parquet files during analysis.
- Do not interpret a missing OHLC bar as proof that no trades occurred.
- Do not mix overnight returns with one-minute intraday returns.
- Do not clip extreme returns without investigating their economic or data origin.
- Avoid look-ahead bias in any future predictive experiment.
- Remember that PCA eigenvector signs are arbitrary.
- Support economic labels with actual loadings and constituent composition.
- Treat the CORE universe as the primary specification and the FULL universe as a robustness check.
