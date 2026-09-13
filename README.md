# Intraday Statistical Factor Extraction in U.S. Financials

## Research objective

This project studies the cross-sectional structure and predictability of intraday volatility among U.S. banks and diversified financial institutions. It begins with raw one-minute SIP bars, constructs a synchronized and auditable return panel, separates broad SPY and XLF exposure from internal banking variation, and then asks two increasingly demanding questions. First, can the residual loading space be represented by stable and economically legible sparse directions? Second, do those directions, or the common subspace that they span, contain information about next-session volatility in a strictly ordered historical walk-forward experiment?

The current answer is deliberately asymmetric. Sparse rotation reveals three persistent loading patterns associated with MS/C, large banks, and regional banks, although only two satisfy the formal local-factor criterion under the retained three-dimensional specification. In forecasting, the rotation-invariant aggregate banking factor contains economically meaningful predictive information, while the three localized factor histories do not improve upon that aggregate representation. Once the complete factor space is removed, the remaining directed volatility network is sparse, weak, and unstable.

The numerical evidence and its interpretation are recorded in [`RESULTS_TO_DATE.md`](RESULTS_TO_DATE.md). The frozen OOS designs are specified separately in [`OOS_PROTOCOL.md`](OOS_PROTOCOL.md), [`OOS_AUDIT.md`](OOS_AUDIT.md), and [`FACTOR_HAR_PROTOCOL.md`](FACTOR_HAR_PROTOCOL.md).

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

The FULL universe additionally contains COF, SCHW, AXP, GS, FHN, and SYF. It is retained as a coverage robustness sample rather than allowed to determine the primary geometry.

## Data construction

Data quality is treated as part of the empirical design. A missing OHLC bar is not interpreted as evidence that no trading occurred. Previous-tick sampling is used only to synchronize prices; the return at the imputed minute and the following return are marked as contaminated and excluded. Returns are computed within each session, so overnight close-to-open movements never enter the intraday panel.

The four-minute systemic SIP gap on 5 June 2023 is removed globally. The complete session on 24 January 2023 is excluded because the NYSE opening-auction malfunction and subsequent cancellations make its early observations unsuitable for ordinary return analysis. The March 2023 regional-bank crisis remains in the sample because it is economically central to the research question. Extreme returns are investigated rather than mechanically winsorized.

For stock (i) at minute (t), the return is

$$
r_{i,t}=\log P_{i,t}-\log P_{i,t-1}.
$$

The resulting sequence is auditable from raw data to model input:

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

## Statistical design

### PCA and benchmark residualization

Let $X$ denote the centered return matrix. Correlation PCA standardizes each stock by its sample standard deviation, producing $Z$, and solves

$$
R=\frac{1}{n-1}Z^\top Z,
\qquad Rv_k=\lambda_kv_k.
$$

The eigenvector $v_k$ defines the component score $f_k=Zv_k$, while the conventional technical loading is $L_{ik}=\sqrt{\lambda_k}v_{ik}$. The implementation preserves both objects so that projection weights are not confused with stock–component correlations. Covariance PCA is retained as a scale-sensitive comparison; correlation PCA is primary because marginal volatility differs materially across stocks.

Broad market and sector exposure is removed stock by stock through

$$
r_{i,t}=\alpha_i+\beta_{i,M}r_{SPY,t}+\beta_{i,F}r_{XLF,t}+e_{i,t}.
$$

The variance ledger reconciles the fitted benchmark component with residual PCA contributions. SPY and XLF are controls, not exogenous instruments, and the resulting residuals must not be interpreted as structural shocks.

### Sparse localization inside the PCA space

Elastic-Net Sparse PCA is used as an exploratory loading map, while the main localization stage uses the L1-rotation criterion of [Freyaldenhoven (2026)](https://doi.org/10.3982/QE2369). Starting from the first $K$ PCA eigenvectors,

$$
\Lambda_0=\sqrt{p}[v_1,\ldots,v_K],
\qquad
Q(q)=\|\Lambda_0q\|_1,
\qquad \|q\|_2=1.
$$

A multistart search selects $K$ linearly independent directions and forms the generally oblique rotation $\Lambda^*=\Lambda_0R$. Scores are recovered by least squares,

$$
\widehat F=Z\Lambda^*(\Lambda^{*\top}\Lambda^*)^{-1}.
$$

Because $R$ is nonsingular, the rotation changes the coordinate system but preserves the retained PCA projector:

$$
\widehat F\Lambda^{*\top}=ZV_KV_K^\top.
$$

The reference local-factor diagnostic counts loadings below $h_p=1/\log p$. With $p=12$, $h_p=0.4024$ and the five-percent critical count is $\gamma_p=7$. A sparse-looking loading pattern is not enough: the result is also evaluated through whole-session bootstrap alignment, $K=2,\ldots,5$ sensitivity, 60- and 120-session rolling fits, and 200-versus-500-start optimization checks.

### Rolling identification

Every rolling window re-estimates benchmark coefficients, standardization, PCA, and L1 rotation from its own trailing observations. Sixty-session windows provide the primary responsive view; 120-session windows provide a slower persistence check; both advance by five trading sessions and include the final available window.

Factor columns are aligned by maximum absolute loading cosine rather than raw column number. Past-only alignment supplies the causal stability diagnostic, whereas full-sample alignment is retained only for ex-post interpretation. Support Jaccard, loading cosine, local-factor status, rotation conditioning, factor-score correlations, and optimization sensitivity are recorded for every window. Overlapping windows remain dependent observations and are never treated as independent hypothesis tests.

## Volatility forecasting

### Corrected residual-network experiment

Stage 16 asks whether cross-bank information predicts volatility after removing SPY/XLF and the complete rolling $K=3$ banking subspace. Valid one-minute residual returns are aggregated into non-overlapping five-minute returns and then into daily realized variance. Each target bank receives unpenalized own daily, weekly, and monthly HAR terms. The same histories from the other eleven banks are partialled with respect to the own-HAR block and estimated by chronological lasso, with the one-standard-error penalty as the frozen primary rule.

The historical replay is pseudo-OOS because the sample had already influenced research design. The separately frozen prospective mode issues forecasts without reading the target outcome, records the exchange-clock information set, and joins realized variance only after it becomes observable.

### Factor-predictability experiment

Stage 17 targets next-session realized variance after removing only SPY and XLF. This leaves the internal banking component in the response and directly tests whether the discovered factor state predicts future volatility. For bank (i),

$$
y^{B}_{i,d+1}
=\alpha_i+\beta_i^\top H_{i,d}+\gamma_i^\top F_d+\varepsilon_{i,d+1},
$$

where $H_{i,d}$ contains the bank's own daily, weekly, and monthly log-realized-variance state. The localized specification places the daily, weekly, and monthly histories of all three sparse factors in $F_d$. The aggregate specification instead uses the corresponding histories of total variance in the retained banking subspace and is invariant to rotations within that subspace.

Own-HAR, Aggregate-Factor HAR, Local-Factor HAR, Network HAR, Hybrid HAR, and persistence share identical forecast keys and a common estimation mask. QLIKE is the primary loss. Inference aggregates the stock panel by target date, applies Bartlett-HAC standard errors, and corrects the predeclared comparisons for multiplicity.

## Main empirical results

The baseline and benchmark-residual geometry is summarized below.

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

The common banking subspace therefore contains stable historical predictive information. The sparse decomposition remains useful for economic interpretation, but it does not add forecast value beyond the aggregate factor. This is evidence about volatility-state predictability, not causal identification, return alpha, or live performance.

## Code organization and reproducibility

The numbered scripts in `src/` are thin research drivers. Reusable numerical work lives in `src/utils/`; presentation code lives in `src/reporting/`. Shared OOS clocks, hashing, strict-return construction, and persistence are isolated in `src/utils/oos_common.py`. Every numbered stage exposes a `main()` entry point and can be imported without executing the pipeline.

After installing `requirements.txt` and defining `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`, the complete sequence is:

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

The complete regression suite is run with:

```text
python -m unittest discover -s tests -v
```

The suite currently contains 59 tests covering reconstruction, variance reconciliation, session resampling, Kalman holdout isolation, rolling-window causality, import safety, serial-versus-parallel equivalence, target-free checkpointing, scoring, and matched-key Stage 17 issuance. Generated CSV, Parquet, Markdown, and figure artifacts remain under `alpaca_us_banks_1m/reports/` and are intentionally excluded from Git.

The downloader filename `01-crwal.py` contains a historical typo and remains unchanged for command compatibility.

## Future research objectives

### Prospective confirmation

The primary next objective is to accumulate the frozen 252-session prospective cohort for Aggregate-Factor HAR versus Own-HAR. No newly observed outcome may alter factor count, feature construction, eligibility rules, or interpretation before its forecast is issued. Stage 16 Network HAR remains in the ledger as a secondary comparison, but it is no longer the principal empirical claim.

### Robust volatility measurement

After the frozen confirmation begins, a separately labelled robustness branch should compare five-minute realized variance with jump-robust and microstructure-robust targets, including bipower variation and a realized-kernel or pre-averaging estimator. These alternatives may explain whether the aggregate factor predicts continuous volatility, jumps, or measurement noise; they must not retroactively modify the frozen target.

### Parsimonious factor dynamics

The next model extension should preserve the aggregate factor's low dimensionality. A regularized comparison among Aggregate-Factor HAR, HARQ-style measurement-error controls, and a small factor-innovation model is preferable to adding many correlated local-factor regressors. Any tuning must remain nested inside the walk-forward origin and must be judged by matched QLIKE rather than in-sample fit.

### State dependence and stress

Predictive gains should be decomposed across calm, high-volatility, and jump-intensive states using thresholds fixed before evaluation. The March–May 2023 episode can support descriptive interpretation, but a formal stress claim requires an evaluation design whose burn-in leaves valid event-period forecasts or a future stress episode observed after the protocol freeze.

### Economic value

A trading or risk-management claim requires a separately frozen mapping from variance forecasts to positions or capital allocation. That stage must specify exposure constraints, execution delay, turnover, transaction costs, and a benchmark policy. Until then, the project establishes statistical volatility predictability rather than tradable alpha.

## Interpretation discipline

PCA directions are statistical coordinates whose signs are arbitrary. Sparse weights and zeros do not prove economic inclusion or exclusion. L1 factors remain conditional on the retained dimension and rotation criterion. SPY/XLF residualization does not create exogenous shocks. Penalized directed edges describe conditional Granger-predictive structure, not contagion or structural causality. Historical walk-forward evidence is stronger than in-sample fit but remains pseudo-OOS when the same history influenced model discovery. The project's strongest claim will therefore remain provisional until it survives the frozen prospective cohort.
