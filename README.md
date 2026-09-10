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
│   ├── common_missing_gaps.csv
│   ├── rolling_pca/
│   └── internal_factor_isolation/
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
        ↓
        SPY/XLF residualization
        ↓
        Variance decomposition
        ↓
        Internal factor isolation
        ↓
        Rolling PCA
```

The scripts deliberately keep the stages separate. Each step reads the previous stage's output and writes a named artifact that can be inspected or reused later.

### One-minute returns

For security `i` and minute `t`, the return is:

$$
r_{i,t} = \log(P_{i,t}) - \log(P_{i,t-1})
$$

Returns are calculated within each trading session. The first minute of every session is set to missing so that the overnight close-to-open move is not mixed into a one-minute intraday return.

## PCA methodology

Let `X` be the centered return matrix. For correlation PCA, `Z` is the same matrix after dividing each column by its sample standard deviation. PCA starts from the appropriate cross-sectional matrix:

$$
\Sigma = \frac{1}{n-1}X^\top X
$$

or, for correlation PCA,

$$
R = \frac{1}{n-1}Z^\top Z
$$

The eigendecomposition is:

$$
R v_k = \lambda_k v_k
$$

The vector `v_k` is the normalized eigenvector used as a set of component weights. The factor scores are calculated afterwards as:

$$
f_k = Z v_k
$$

There are two related quantities that are often both called loadings. The files produced by the baseline scripts preserve the eigenvector weights `v_k` for backward compatibility. The conventional technical loadings are:

$$
L_{i,k} = \sqrt{\lambda_k}\,v_{i,k}
$$

For correlation PCA, `L_{i,k}` is the correlation between stock `i` and component `k`. The new reusable PCA module exposes both representations so that interpretation and projection do not get mixed.

### Covariance PCA

Covariance PCA works with centered returns and preserves the original volatility scale of each security. Higher-volatility stocks therefore have more influence on the estimated covariance structure.

### Correlation PCA

Correlation PCA first standardizes each security:

$$
z_{i,t} = \frac{r_{i,t} - \mu_i}{\sigma_i}
$$

This gives each stock comparable marginal volatility and makes the result more focused on co-movement than on differences in individual volatility. It is the main baseline specification because volatility levels differ materially across the universe.

The first baseline comparison saves eigenvalues, explained-variance ratios, cumulative explained variance, eigenvector weights for PC1 through PC3, technical loadings, the correlation matrix, and the associated plots.

## Market and sector residualization

After the raw-return baseline, broad market and financial-sector exposure will be removed using SPY and XLF:

$$
r_{i,t} = \alpha_i + \beta_{i,M} r_{SPY,t} + \beta_{i,F} r_{XLF,t} + \epsilon_{i,t}
$$

The residuals are the part of each stock's return that is not explained by those two benchmark returns. PCA on the residual matrix will then be compared with PCA on the original returns.

This comparison is intended to answer a specific question: does the financial-stock universe contain internal structure that is hidden by broad market and sector exposure?

### Variance decomposition

The residual analysis also keeps an explicit ledger of where the original variance goes. For each stock, the fitted SPY/XLF part and the residual part are additive:

$$
Var(r_i) = Var(\hat{r}_i) + Var(\epsilon_i)
$$

The residual covariance is then decomposed into its principal components. If `v_{i,k}` is the loading of stock `i` on residual component `k`, that component contributes:

$$
c_{i,k} = \lambda_k v_{i,k}^2
$$

This lets us report, for every stock, the share explained by the combined SPY/XLF fit, residual PC1, residual PC2-PC3, and residual PC4-PC12. The benchmark share is kept combined because separating SPY and XLF into two squared terms would double-count their covariance.

## Internal factor isolation

Once standard PCA has shown where the common variation lies, the next question is whether the factors can be made easier to interpret without pretending that the statistical components are economic causes. Script `12_internal_factor_isolation.py` compares three views of the same correlation-scaled stock panel:

- standard PCA, which remains the variance-maximizing reference;
- Varimax, which rotates the first three-component subspace toward a simpler loading pattern;
- Elastic-Net Sparse PCA, which penalizes the component weights so that some become exactly zero while correlated stocks are less likely to be selected arbitrarily.

### Varimax rotation

Let `L` contain the conventional loadings of the first `K` PCA components. Varimax searches for an orthogonal rotation `Q`:

$$
L_{\mathrm{rot}} = LQ,
\qquad Q^\top Q = I
$$

The rotation maximizes a simple-structure criterion:

$$
\max_Q\;\sum_{k=1}^{K}
\left[
\frac{1}{p}\sum_{i=1}^{p}L_{\mathrm{rot},i,k}^{4}
-
\left(\frac{1}{p}\sum_{i=1}^{p}L_{\mathrm{rot},i,k}^{2}\right)^2
\right]
$$

Varimax does not remove a stock. It changes the coordinate system inside the selected factor subspace, so it is a useful interpretive lens when several components may be rotated without changing the represented subspace. The original criterion is due to [Kaiser (1958), *The Varimax Criterion for Analytic Rotation in Factor Analysis*](https://doi.org/10.1007/BF02289233).

### Elastic-Net Sparse PCA

Sparse PCA modifies the PCA reconstruction problem by adding an L1 penalty and an L2 penalty. In the implementation, `B` contains the sparse component weights and `A` keeps the auxiliary factors orthonormal:

$$
\min_{A,B}\;
\lVert Z-ZBA^\top\rVert_F^2
+\lambda_2\lVert B\rVert_F^2
+\sum_{j=1}^{K}\lambda_{1,j}\lVert b_j\rVert_1,
\qquad A^\top A=I
$$

The two penalties have different jobs:

- `L1` (`lambda_1`) creates exact zero weights and controls sparsity;
- `L2` (`lambda_2`) stabilizes correlated groups and reduces the tendency to keep one arbitrary representative.

The method follows the regression formulation introduced by [Zou, Hastie, and Tibshirani (2006), *Sparse Principal Component Analysis*](https://doi.org/10.1198/106186006X113430). The Elastic-Net grouping motivation comes from [Zou and Hastie (2005), *Regularization and Variable Selection Via the Elastic Net*](https://doi.org/10.1111/j.1467-9868.2005.00503.x). The earlier direct L1-constrained approach, SCoTLASS, is documented by [Jolliffe, Trendafilov, and Uddin (2003), *A Modified Principal Component Technique Based on the LASSO*](https://doi.org/10.1198/1061860032148).

The least-squares reconstruction used by the script is only an internal diagnostic of how much information the sparse score space retains. It is not an additional economic regression model. Sparse components can also lose the exact orthogonality and variance-ordering properties of ordinary PCA, so their reconstruction percentage is reported separately from ordinary PCA explained-variance shares. This distinction is also emphasized in the finance-oriented discussion by [Despois (2023), *Identifying and Interpreting the Factors in Factor Models via Sparsity: Different Approaches*](https://doi.org/10.1002/jae.2967).

The project therefore uses the following hierarchy:

```text
Correlation PCA       official variance reference
Varimax               interpretable rotation of the same subspace
Elastic-Net Sparse PCA sparse exploratory factor composition
Pure Lasso             sensitivity comparison, not the main result
```

The current penalty path records reconstruction and non-zero-weight counts for several `lambda_1` values, both with and without the `lambda_2` grouping term. A penalty will be considered useful only if the selected stocks are reasonably stable and the information loss is transparent; a visually sparse result alone is not sufficient evidence of a real factor.

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

#### Rolling-window design and literature rationale

The rolling analysis will use trading sessions as the time unit. This keeps a window tied to an economically meaningful number of market days and avoids treating weekends, holidays, and early closes as ordinary observations. The window length is a design choice rather than a universal constant: shorter windows react faster to regime changes but produce noisier covariance and PCA estimates, while longer windows are more stable but can average together distinct regimes.

The primary specification will use a **20-session window** with a **5-session step**. Twenty sessions are approximately one month of trading and contain about 7,800 scheduled one-minute slots per stock; in this cleaned sample, early closes and data-quality exclusions leave roughly 7,500-7,800 usable observations. This is a reasonable compromise for a 12-stock cross-section: it is short enough to detect changes around stress periods, while high-frequency sampling supplies many observations inside the calendar window. The count is not an effective independent sample size because intraday returns can be dependent and affected by microstructure noise. High-frequency factor research explicitly motivates short rolling intervals for this reason, including one-month windows for time-varying betas.

The analysis will repeat every comparison with a **60-session window**, also evaluated every five sessions. Sixty sessions are approximately one quarter and provide a slower, more stable view of the same structure. Sixty-day rolling correlations are also a familiar diagnostic in the co-movement literature. The 60-session result is therefore a robustness and persistence check, not a claim that three months is intrinsically optimal.

We will not use a one-week window as the main estimate. High-frequency PCA papers show that very short horizons can be informative when the cross-section and estimator are designed for that setting, but our first implementation uses a simple rolling sample covariance/correlation matrix on a small 12-stock universe. A one-week estimate would be more sensitive to individual events, missing observations, microstructure effects, and estimation noise. It remains a useful stress-sensitivity experiment later.

The five-session step is chosen for readable weekly monitoring. Because neighbouring rolling windows overlap, the resulting observations are not independent. We will not interpret a smooth rolling chart as a sequence of independent tests; confidence bands or block-bootstrap checks will be added before making formal claims about changes in loadings or eigenvalues.

The first rolling implementation will run both correlation PCA and covariance PCA on:

- raw returns;
- intraday-normalized returns;
- SPY/XLF residual returns.

`11_rolling_pca.py` implements this first descriptive pass. The intraday-normalized series uses the same fixed full-sample minute-of-day profile as the earlier robustness check; this is acceptable for descriptive comparison, but it must be replaced with a past-only profile before any forecasting or walk-forward test.

For each window we will record PC1 explained variance, cumulative variance explained by the first three components, effective dimension, loading similarity to the previous window, benchmark variance removed, and residual PC1 strength. This makes the interpretation explicit:

- a change in raw PCA that disappears after SPY/XLF residualization is benchmark-associated;
- a change that remains in residual PCA is evidence of internal financial-stock structure;
- a change in covariance PCA without a comparable change in correlation PCA is primarily a volatility-scale effect;
- a short-lived isolated spike is a regime or event candidate, not proof of a persistent factor.

Rolling PCA can locate timing, persistence, and structural changes. It cannot by itself establish that an external variable caused them. To investigate external explanations, the rolling results must later be aligned with independent series and event dates, such as market volatility, interest rates, credit conditions, or documented market events. SPY and XLF are treated as benchmark-associated controls, not as exogenous causal instruments, especially because XLF contains financial stocks.

This design is grounded in the following literature:

- [Aït-Sahalia and Xiu (2019), *Principal Component Analysis of High-Frequency Data*](https://doi.org/10.1080/01621459.2017.1401542) study time-varying high-frequency principal components and show that their explanatory power can change materially during stress periods.
- [Aït-Sahalia, Kalnina, and Xiu (2020), *High-Frequency Factor Models and Regressions*](https://doi.org/10.1016/j.jeconom.2020.01.007) discuss short rolling intervals such as one month, the time variation of betas, and the ability of high-frequency observations to reduce estimation noise.
- [Pelger (2019), *Large-dimensional factor modeling based on high-frequency observations*](https://doi.org/10.1016/j.jeconom.2018.09.004) shows why high-frequency factor methods can study short horizons, while also making clear that the appropriate horizon depends on the estimator and the cross-section.
- [Gospodinov (2017), *Asset Co-movements: Features and Challenges*](https://fraser.stlouisfed.org/title/working-papers-federal-reserve-bank-atlanta-8586/asset-co-movements-657145/content/fulltext/frbatl_wp_2017-11) uses 60- and 120-day rolling co-movement diagnostics and warns that apparent time variation can also arise from finite-sample uncertainty and overlapping windows.
- [Zhang and Tong (2022), *Asymptotic Theory of Principal Component Analysis for Time Series Data with Cautionary Comments*](https://doi.org/10.1111/rssa.12793) show why time-series dependence matters for inference on PCA loadings and motivate bootstrap-based uncertainty checks.

### 5. Internal factor composition

The next analysis will compare standard correlation PCA with Varimax and Elastic-Net Sparse PCA. The purpose is not to force an economic story onto PC1, PC2, or PC3. It is to ask whether the statistical structure can be described using a smaller, more stable set of titles and whether that description survives benchmark residualization.

The first pass uses three components and a transparent penalty path. Pure L1 and Elastic Net are both reported so that the grouping effect can be seen rather than assumed. A later version can add rolling sparse-factor stability and block bootstrap intervals once the static structure is understood.

### 6. Covariance estimation and random-matrix diagnostics

Sample covariance will be compared with shrinkage estimators such as Ledoit-Wolf, and potentially with exponentially weighted covariance. Random Matrix Theory will be used as a diagnostic benchmark for separating strong empirical components from noise. The Marchenko-Pastur distribution will not be treated as literal truth because the returns are not iid Gaussian observations.

### 7. Residual dynamics and out-of-sample testing

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
- Variance ledger showing how benchmark and residual PCA components add back to raw variance
- Initial code cleanup and local Git versioning
- Descriptive rolling covariance/correlation PCA with 20- and 60-session windows
- Reusable PCA and benchmark-residualization modules
- Internal factor-isolation script with Varimax and Elastic-Net Sparse PCA

Next:

- Inspect the Varimax loading maps and the L1/L2 sparsity path
- Compare sparse-factor support across raw and benchmark-residualized returns
- Review the rolling PCA diagnostics and identify stress-window candidates
- Add block-bootstrap confidence bands before making formal rolling-inference claims

Later:

- Rolling Sparse PCA and stress-regime comparison
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
10_variance_decomposition.py  reconcile raw variance with benchmark and residual PCA parts
11_rolling_pca.py             run descriptive 20/60-session rolling PCA diagnostics
12_internal_factor_isolation.py  compare PCA, Varimax, and Elastic-Net Sparse PCA
```

The reusable numerical helpers live in `src/pca_utils.py`; benchmark projection and residualization are shared through `src/benchmark_utils.py`. The numbered scripts call these modules instead of maintaining separate PCA implementations.

The downloader filename contains a historical typo (`crwal`). It is kept for compatibility with the existing workflow and can be renamed once any external run commands have been updated.

## Research principles

- Never overwrite raw Alpaca Parquet files during analysis.
- Do not interpret a missing OHLC bar as proof that no trades occurred.
- Do not mix overnight returns with one-minute intraday returns.
- Do not clip extreme returns without investigating their economic or data origin.
- Avoid look-ahead bias in any future predictive experiment.
- Remember that PCA eigenvector signs are arbitrary.
- Support economic labels with actual loadings and constituent composition.
- Treat Varimax and Sparse PCA as interpretive tools, not causal identification.
- Do not interpret sparse zero weights as proof that a stock has no economic exposure.
- Treat the CORE universe as the primary specification and the FULL universe as a robustness check.
