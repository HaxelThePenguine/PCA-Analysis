# Intraday Statistical Factor Extraction in U.S. Financials

## Research objective

I started this project to understand what moves U.S. banks together at intraday frequency, and what remains after removing SPY and XLF exposure. The pipeline goes from one-minute SIP bars to PCA, sparse factor identification, and next-session volatility forecasts. The main questions are whether the internal banking factors are stable and interpretable, and whether they help predict volatility out of sample.

So far, the aggregate banking factor gives the clearest forecasting result. Sparse rotation identifies persistent MS/C, large-bank, and regional-bank patterns, but the localized histories do not improve on the aggregate factor. After removing the complete factor space, the remaining volatility network is weak and unstable.

Detailed historical results are in [RESULTS_TO_DATE.md](RESULTS_TO_DATE.md). [OOS_PROTOCOL.md](OOS_PROTOCOL.md) collects the Stage 16–17 targets, timing, models, and evaluation rules.

## Data and research universe

Historical observations come from the Alpaca SIP feed. The sample runs from 1 January 2023 through 9 September 2026 and contains 924 regular U.S. trading sessions. Prices are sampled at one-minute frequency during the official regular session, including exchange-calendar early closes.

| Item | Production choice |
| --- | --- |
| Frequency | One-minute SIP bars |
| Fields | OHLC, volume, trade count, and VWAP |
| Downloaded universe | 18 financial stocks, SPY, and XLF |
| Primary universe | 12-stock CORE panel |
| CORE stocks | JPM, BAC, WFC, C, USB, TFC, KEY, RF, FITB, CFG, HBAN, MS |
| Robustness universe | All 18 downloaded financial stocks |
| Broad controls | SPY and XLF |
| Primary scaling | Correlation PCA |

FULL also contains COF, SCHW, AXP, GS, FHN, and SYF and serves as a coverage robustness check.

## Data construction

Prices are aligned to the exchange calendar, so a missing SPY bar does not erase the minute from every other stock. Previous-tick sampling stays within each session; the return at an imputed minute and the following return are marked as contaminated. Only consecutive one-minute returns are retained. Overnight movements and returns spanning a gap are excluded.

The systemic SIP gap on 5 June 2023 is removed globally. The 24 January 2023 session is excluded because of the NYSE opening-auction malfunction and cancellations. The March 2023 regional-bank crisis stays in the sample. Extreme returns are investigated rather than mechanically winsorized.

For stock $i$ at minute $t$,

$$
r_{i,t}=\log P_{i,t}-\log P_{i,t-1}.
$$

The pipeline is:

```text
SIP minute bars
    -> exchange-calendar alignment and quality diagnostics
    -> synchronized prices and contamination masks
    -> within-session log returns
    -> CORE and FULL research panels
    -> SPY/XLF residualization
    -> PCA, sparse rotation, and rolling stability
    -> realized-volatility construction
    -> target-free walk-forward issuance and delayed scoring
```

Raw and generated data remain under `alpaca_us_banks_1m/` and are excluded from Git. Source code, protocol documents, and tests remain versioned.

### Universe selection and temporal splits

CORE and FULL select stocks; they are not train/test datasets. Stage 6 drops incomplete rows separately for each universe. The OOS stages select CORE plus SPY/XLF before cleaning, so missing observations in stocks outside CORE do not reduce the primary sample.

Time splits use whole trading sessions. The Kalman comparison uses 48 training and 12 holdout sessions within each 60-session window, with benchmark coefficients, scaling, and PCA fitted on training only. Stages 16 and 17 use walk-forward estimation: training labels must be observable at the forecast origin, tuning stays within the training history, and target outcomes are joined only during scoring. Missing sessions stay on the exchange calendar when constructing HAR lags.

## Statistical design

### PCA and benchmark residualization

Let $X$ denote the centered return matrix. Correlation PCA standardizes each stock by its sample standard deviation, producing $Z$, and solves

$$
R=\frac{1}{n-1}Z^{\top}Z,
\qquad Rv_k=\lambda_kv_k.
$$

The score is $f_k=Zv_k$; the loading is $L_{ik}=\sqrt{\lambda_k}v_{ik}$. Both are kept explicitly. Correlation PCA is primary because stocks have different volatility scales; covariance PCA provides a comparison.

Broad market and sector exposure is removed stock by stock through

$$
r_{i,t}=\alpha_i+\beta_{i,M}r_{SPY,t}+\beta_{i,F}r_{XLF,t}+e_{i,t}.
$$

The variance ledger reconciles benchmark and residual PCA contributions. SPY and XLF are controls, so their residuals are not structural shocks.

### Sparse localization inside the PCA space

Elastic-Net Sparse PCA provides an exploratory loading map. The main localization stage uses the L1-rotation criterion of [Freyaldenhoven (2026)](https://doi.org/10.3982/QE2369). Starting from the first $K$ PCA eigenvectors,

$$
\Lambda_0=\sqrt{p}[v_1,\ldots,v_K],
\qquad
Q(q)=\|\Lambda_0q\|_1,
\qquad \|q\|_2=1.
$$

A multistart search selects $K$ independent directions and forms the generally oblique rotation $\Lambda_{\mathrm{rot}}=\Lambda_0R$. Least-squares scores are

$$
\widehat F
=Z\Lambda_{\mathrm{rot}}
\left(\Lambda_{\mathrm{rot}}^{\top}\Lambda_{\mathrm{rot}}\right)^{-1}.
$$

The rotation preserves the retained PCA projector:

$$
\widehat F\Lambda_{\mathrm{rot}}^{\top}=ZV_KV_K^{\top}.
$$

The local-factor diagnostic counts loadings below $h_p=1/\log p$. For $p=12$, $h_p=0.4024$ and the five-percent critical count is $\gamma_p=7$. Checks include whole-session bootstrap alignment, $K=2,\ldots,5$, 60/120-session rolling fits, and 200/500-start optimization sensitivity.

### Rolling identification

Each rolling window re-estimates benchmarks, scaling, PCA, and L1 rotation from trailing observations. The primary window is 60 sessions, with 120 sessions as a persistence check. Both advance by five sessions and include the final available window.

Factor columns are aligned by absolute loading cosine. Past-only alignment measures causal stability; full-sample alignment is descriptive. Diagnostics track support Jaccard, loading cosine, local-factor status, conditioning, score correlations, and optimization sensitivity. Overlapping windows are not independent observations.

## Volatility forecasting

Stage 16 tests cross-bank predictability after removing SPY/XLF and the full $K=3$ banking subspace. Stage 17 removes only SPY/XLF from the target and asks whether banking-factor histories improve on Own-HAR. The aggregate specification uses total variance in the retained subspace; local factors use its sparse coordinates.

Both stages forecast next-session realized variance from daily, weekly, and monthly histories. They use chronological estimation, matched forecast keys, QLIKE loss, and date-level inference. Issuance is separate from scoring. The target formulas, model comparisons, and frozen settings are in [OOS_PROTOCOL.md](OOS_PROTOCOL.md).

## Main empirical results

These are the recorded September 2026 results. They have not been rerun on market data after the October preprocessing and CORE-filtering fixes. [RESULTS_TO_DATE.md](RESULTS_TO_DATE.md) records their provenance and limits.

| Specification | PC1 variance | First three components |
| --- | ---: | ---: |
| Raw correlation PCA | 63.77% | 75.85% |
| Intraday-normalized correlation PCA | 64.47% | 75.81% |
| SPY/XLF residual correlation PCA | 35.71% | 56.11% |

SPY/XLF account for 43.59% of total CORE variance. The full-sample $K=3$ L1 rotation produces small-loading counts of 9, 8, and 4. LF1 is concentrated on MS, C, and WFC; LF2 on JPM, BAC, WFC, and an unstable C/MS boundary; LF3 on JPM and the seven regional-bank names. LF1 and LF2 pass the local-factor diagnostic. LF3 does not, but its direction and support are exceptionally stable, so it is best described as a broad regional-bank cluster.

The corrected Stage 16 network gain is small. Under the primary 120-session expanding specification, QLIKE changes from 0.240384 for Own-HAR to 0.240250 for Network HAR, a relative improvement of 0.056%. No primary directed edge reaches the frozen 70% stability threshold.

Stage 17 produces the stronger predictive result.

| Specification | Own-HAR | Aggregate factor | Local factors | Aggregate improvement |
| --- | ---: | ---: | ---: | ---: |
| 120-session factor, expanding HAR | 0.210206 | 0.199614 | 0.206196 | 5.039% |
| 120-session factor, rolling-252 HAR | 0.213218 | 0.202685 | 0.216944 | 4.940% |
| 60-session factor, expanding HAR | 0.214540 | 0.204428 | 0.209091 | 4.713% |

The aggregate banking factor has the strongest historical predictive result. Sparse factors help interpretation but add no forecast value beyond the aggregate. This does not establish causality, return alpha, or live performance.

## Architecture and execution

Numbered scripts in `src/` run each stage. Calculations live in `src/utils/`, reporting in `src/reporting/`, and shared OOS helpers in `src/utils/oos_common.py`. Every stage exposes `main()` and can be imported without running the pipeline.

| Layer | Role |
| --- | --- |
| `src/config.py` | Paths, universes, dates, and cleaning rules |
| Numbered scripts in `src/` | Stage entry points and saved outputs |
| `src/utils/` | Preprocessing, factors, rolling estimation, and forecasting |
| `src/reporting/` | Tables, plots, and summaries |
| `tests/` | Numerical, timing, and pipeline regression checks |

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

To download SIP data, define `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in your environment. If the raw data are already available, start at Stage 2. Run the stages below in order with `python src/<filename>`; Stages 16 and 17 also require a mode.

```text
01-crwal.py                           acquire/resume SIP data and quality reports
02-inspect_missing.py                inspect the minute missingness matrix
03-preprocess.py                     synchronize prices and construct returns
04-check_returns.py                  inspect return distributions and extremes
05-clean_returns.py                  remove invalid sessions and contaminated returns
06_save_universes.py                 persist CORE and FULL panels
07_baseline_pca.py                   estimate covariance and correlation PCA
08_intraday_normalization.py         test minute-of-day volatility normalization
09_benchmark_residualization.py      remove SPY/XLF exposure
10_variance_decomposition.py         reconcile benchmark and PCA variance
11_rolling_pca.py                    run descriptive 20/60-session PCA
12_internal_factor_isolation.py      compare PCA, Varimax, and Sparse PCA
13_l1_local_factor_identification.py identify and bootstrap local factors
13_bis_kalman_dynamic_factors.py     compare rolling and state-space factors
14_dynamic_local_factor_regimes.py   estimate rolling L1 stability
15_factor_adjusted_residual_network.py retain the legacy network comparison
16_oos_har_network_validation.py     run the corrected residual-network protocol
17_oos_factor_augmented_har.py       test factor volatility predictability
```

For a short execution check after preprocessing:

```text
python src/16_oos_har_network_validation.py --mode smoke --no-figures
python src/17_oos_factor_augmented_har.py --mode smoke
```

Smoke mode uses the last 180 sessions and lighter settings; it still requires the local dataset. Tests cover reconstruction, variance reconciliation, preprocessing, CORE selection, holdout isolation, rolling causality, import safety, parallel equivalence, checkpointing, and scoring. Generated artifacts stay under `alpaca_us_banks_1m/reports/` and are excluded from Git.

The downloader filename `01-crwal.py` contains a historical typo and remains unchanged for command compatibility.

## Next step

Rerun the corrected pipeline on the real dataset, then accumulate the frozen 252-session prospective comparison of Aggregate-Factor HAR against Own-HAR. Factor count, features, eligibility, and tuning rules stay fixed before evaluation. Stage 16 Network HAR remains a secondary comparison.

## Interpretation discipline

PCA signs are arbitrary, and sparse weights do not prove economic inclusion or exclusion. L1 factors depend on the retained dimension and rotation criterion. Residualization does not create exogenous shocks; directed network edges describe conditional predictability, not contagion. Historical walk-forward results remain pseudo-OOS when the same history informed model discovery. The main claim still needs prospective confirmation.
