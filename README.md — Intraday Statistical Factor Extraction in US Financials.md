# Intraday Statistical Factor Extraction & Residual Dynamics in US Financials

## Overview

This project studies the intraday statistical structure of large U.S. financial stocks using 1-minute market data.

The main objective is to extract latent common factors with **Principal Component Analysis (PCA)** and related spectral methods, then study whether the residual components show useful structure such as:

- mean reversion,
- persistence,
- lead-lag effects,
- regime dependence,
- changing correlation patterns,
- and short-horizon predictability.

The project focuses heavily on **data quality, synchronization, covariance estimation, and out-of-sample robustness**.

---

## Research Idea

At each minute $t$, the vector of financial-stock returns is

$$
\mathbf{r}_t =
\begin{bmatrix}
r_{1,t} \\
r_{2,t} \\
\vdots \\
r_{N,t}
\end{bmatrix}
\in \mathbb{R}^{N}
$$

The covariance matrix of returns is

$$
\Sigma = \operatorname{Cov}(\mathbf{r}_t)
$$

PCA solves the eigenvalue problem

$$
\Sigma \mathbf{v}_k = \lambda_k \mathbf{v}_k
$$

where:

- $\mathbf{v}_k$ is the eigenvector associated with the $k$-th principal component;
- $\lambda_k$ is the corresponding eigenvalue;
- $\lambda_k$ measures the variance explained by that component.

The return of the $k$-th statistical factor is defined as

$$
f_{k,t} = \mathbf{v}_k^\top \mathbf{r}_t
$$

The analysis will study both the dominant factors and the residual dynamics left after removing them.

---

# Dataset

Historical market data is retrieved from **Alpaca Market Data** using the SIP feed.

Frequency:

```text
1-minute OHLCV
```

Available fields:

```text
timestamp
open
high
low
close
volume
trade_count
vwap
```

Period:

```text
2023-01-01
to
2026-09-09
```

Only regular U.S. trading sessions are used.

Normal trading hours:

```text
09:30 – 16:00 America/New_York
```

Early-close sessions are handled through the actual market calendar.

The downloaded dataset contains:

```text
924 trading sessions
~358,920 expected minute observations per symbol
```

---

# Universe

Current financial universe:

```text
BAC
COF
JPM
HBAN
WFC
USB
RF
SCHW
C
AXP
TFC
FITB
KEY
GS
MS
FHN
CFG
SYF
```

Benchmarks:

```text
XLF
SPY
```

The final PCA universe may be narrower depending on economic comparability and data quality.

---

# Data Quality

Before estimating any covariance matrix, the full dataset was checked for missing one-minute bars.

Coverage is defined as

$$
C_i =
\frac{N_i^{\mathrm{observed}}}
{N_i^{\mathrm{expected}}}
$$

Most securities show extremely high coverage.

Representative results:

| Symbol | Coverage |
| ------ | -------: |
| BAC    | 99.999% |
| JPM    | 99.998% |
| USB    | 99.994% |
| C      | 99.990% |
| WFC    | 99.983% |
| MS     | 99.934% |
| CFG    | 99.866% |
| COF    | 99.435% |
| AXP    | 99.429% |
| GS     | 98.413% |

The temporal structure of missing observations is also analyzed, since the same coverage percentage can come from very different market situations.

---

# Missing-Bar Analysis

A minute-by-symbol missing-data matrix was created:

$$
M_{i,t} =
\begin{cases}
1, & \text{if the bar is missing} \\
0, & \text{if the bar is observed}
\end{cases}
$$

This allows missing observations to be separated into three main categories.

## Systemic Data Gaps

On **2023-06-05 between 09:52 and 09:55 ET**, 18 instruments were simultaneously missing, including SPY, XLF, JPM, and BAC.

These timestamps are treated as common market-data gaps and removed from the full panel.

## Security-Specific Gaps

Some securities have isolated missing bars while the rest of the market is trading normally.

This appears more often in names such as:

```text
GS
AXP
COF
FHN
SYF
```

A missing OHLC bar can occur even when transactions happened during that minute, depending on SIP trade conditions and bar-construction rules.

Short isolated gaps can therefore be handled through previous-tick price sampling while keeping an explicit imputation flag.

## Trading Halts and Stress Events

Long contiguous gaps are handled separately.

A clear example appears on **2023-03-13**, during the regional banking crisis, with missing intervals affecting names such as:

```text
SCHW
RF
FHN
KEY
```

These periods are treated as special market events and tracked with dedicated flags.

---

# Data Structure

Raw observations are stored as compressed Parquet files and kept unchanged.

```text
alpaca_us_banks_1m/
│
├── raw/
│   └── by_symbol/
├── chunks/
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

The raw Parquet files are immutable. Intermediate files are rebuilt by the
preprocessing step, while the processed files are the cleaned research inputs.

---

# Preprocessing Pipeline

The planned preprocessing pipeline is:

```text
Raw SIP minute bars
        ↓
Market-calendar alignment
        ↓
Systemic-gap removal
        ↓
Missing-bar classification
        ↓
Corporate-action handling
        ↓
Cross-sectional synchronization
        ↓
Short-gap treatment
        ↓
1-minute log returns
        ↓
Intraday seasonality normalization
        ↓
SPY / XLF residualization
        ↓
PCA
```

One-minute log returns are defined as

$$
r_{i,t}
=
\log P_{i,t}
-
\log P_{i,t-1}
=
\log\left(
\frac{P_{i,t}}{P_{i,t-1}}
\right)
$$

Returns are calculated within each trading session so that overnight moves do not become artificial one-minute observations.

---

# Intraday Normalization

Intraday volatility changes significantly during the session, with higher activity near the market open and close.

Minute-of-day volatility is estimated as

$$
\sigma_i(m)
=
\operatorname{StdDev}
\left(
r_{i,t}
\mid
m_t = m
\right)
$$

where $m_t$ denotes the minute of the trading session associated with observation $t$.

Normalized returns can then be constructed as

$$
\tilde{r}_{i,t}
=
\frac{r_{i,t}}
{\sigma_i(m_t)}
$$

This helps prevent highly volatile parts of the session from dominating covariance estimation.

---

# Market and Sector Residualization

A second specification removes broad market and sector exposure through the regression

$$
r_{i,t}
=
\alpha_i
+
\beta_{i,M} r_{\mathrm{SPY},t}
+
\beta_{i,F} r_{\mathrm{XLF},t}
+
\varepsilon_{i,t}
$$

The estimated residual return is

$$
\hat{\varepsilon}_{i,t}
=
r_{i,t}
-
\hat{\alpha}_i
-
\hat{\beta}_{i,M} r_{\mathrm{SPY},t}
-
\hat{\beta}_{i,F} r_{\mathrm{XLF},t}
$$

PCA will then be compared across:

```text
raw returns
vs
SPY/XLF residual returns
```

---

# PCA and Extensions

Given the cleaned return matrix

$$
R =
\begin{bmatrix}
\mathbf{r}_1^\top \\
\mathbf{r}_2^\top \\
\vdots \\
\mathbf{r}_T^\top
\end{bmatrix}
\in \mathbb{R}^{T \times N}
$$

the sample covariance matrix is

$$
\Sigma
=
\frac{1}{T-1}
R^\top R
$$

assuming that the return matrix has been centered.

The covariance matrix is decomposed as

$$
\Sigma
=
V \Lambda V^\top
$$

where:

- $V$ contains the eigenvectors;
- $\Lambda$ is the diagonal matrix of eigenvalues.

The project will study:

- eigenvalue spectrum,
- explained variance,
- eigenvector loadings,
- eigenportfolio returns,
- rolling PCA,
- eigenvector stability,
- covariance shrinkage,
- random-matrix noise diagnostics,
- residual mean reversion,
- regime changes,
- lead-lag relationships.

Stress periods such as the March 2023 banking crisis will also be analyzed separately.

---

# Robustness Tests

Results will be compared across different specifications:

```text
Core high-coverage universe
vs
Full universe

Raw returns
vs
Intraday-normalized returns

Raw returns
vs
SPY/XLF residual returns

Sample covariance
vs
Shrinkage covariance

1-minute frequency
vs
5-minute aggregation
```

These comparisons help measure how sensitive the PCA structure is to data treatment and model specification.

---

# Current Status

Completed:

- [x] Historical SIP data download
- [x] 2023–2026 dataset construction
- [x] Market-calendar and early-close handling
- [x] Monthly chunking and resume support
- [x] Per-symbol and per-day data-quality reports
- [x] Missing-data matrix
- [x] Systemic-gap detection
- [x] Initial classification of security-specific gaps
- [x] Synchronized price and return matrices
- [x] Contamination masking and complete-panel construction
- [x] CORE and FULL research universes

In progress:

- [ ] Baseline covariance and correlation PCA
- [ ] Outlier and invalid-value review

Planned:

- [ ] Static and rolling PCA
- [ ] Eigenportfolio analysis
- [ ] Eigenvalue and eigenvector stability
- [ ] Shrinkage covariance
- [ ] Random Matrix Theory diagnostics
- [ ] Residual dynamics
- [ ] Walk-forward testing
- [ ] Transaction-cost analysis

---

# Reproducibility

API credentials are read from environment variables:

```text
ALPACA_API_KEY
ALPACA_SECRET_KEY
```

Suggested `.gitignore`:

```text
.env
.venv/
__pycache__/

alpaca_us_banks_1m/chunks/
alpaca_us_banks_1m/raw/
alpaca_us_banks_1m/intermediate/
alpaca_us_banks_1m/processed/

*.parquet
```

Main technologies:

```text
Python
pandas
NumPy
SciPy
scikit-learn
statsmodels
PyArrow
Alpaca Market Data
```

---

# Goal

The final objective is to understand the latent structure of intraday financial-stock returns and test whether the information left outside the dominant common factors contains stable predictive structure.

The main questions are:

1. **Which common statistical factors dominate intraday U.S. financial stocks?**
2. **How stable are these factors across time and stress regimes?**
3. **Do residual components contain exploitable short-horizon structure?**
