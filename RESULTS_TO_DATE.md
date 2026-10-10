# Results to Date

Recomputed on 10 October 2026 using Alpaca SIP data from 1 January 2023 through 9 September 2026. The calendar contains 924 sessions; excluding the 24 January 2023 auction session leaves 923 usable sessions and 354,329 complete CORE minute returns. These results use the current CORE filtering, calendar alignment, and training-only Kalman fits.

The sample informed factor discovery and model design, so the forecasting results remain historical pseudo-OOS. Prospective confirmation is still pending.

## Forecasting result

Aggregate-Factor HAR reduces historical pooled QLIKE by 2.916–3.784% relative to Own-HAR across the three specifications. Separate local-factor histories do not improve on the aggregate representation. After removing the full banking factor space, Network HAR improves primary QLIKE by 0.030%. No primary directed HAR term reaches the 70% stability threshold.

## Common variation and local factors

SPY/XLF account for 43.59% of raw CORE variance.

| Specification | PC1 variance | First three components |
| --- | --- | --- |
| Raw correlation PCA | 63.77% | 75.85% |
| Intraday-normalized covariance PCA | 64.47% | 75.81% |
| SPY/XLF residual correlation PCA | 35.71% | 56.11% |

Minute-of-day normalization changes the leading variance shares only slightly. Its PCA uses the covariance matrix of normalized returns. Removing SPY/XLF reduces common variation but leaves a banking subspace.

The full-sample $K=3$ L1 rotation preserves that subspace. With $h_p=1/\log(12)=0.4024$, a direction needs more than seven small loadings to pass the reference local-factor rule.

| Direction | Active support | Small loadings | Local-factor rule |
| --- | --- | --- | --- |
| LF1 | WFC, C, MS | 9 | Passes |
| LF2 | JPM, BAC, WFC, MS | 8 | Passes |
| LF3 | JPM, USB, TFC, KEY, RF, FITB, CFG, HBAN | 4 | Does not pass |

LF1 concentrates on MS/C, LF2 on large banks with an unstable C/MS boundary, and LF3 on the regional-bank cluster. Whole-session bootstrap median loading cosines are 0.9991, 0.9977, and 0.9999. LF2's fifth percentile is 0.6897; the other two exceed 0.996. The broad regional direction is the most stable even though it does not pass the local-factor rule.

## Rolling stability

The 60- and 120-session fits use 174 and 162 windows, advancing by five sessions. These overlapping windows are descriptive observations.

| Window | Factor | Mean past-only cosine | Mean support Jaccard | Local-factor rate |
| --- | --- | --- | --- | --- |
| 60 | LF1 | 0.960 | 0.656 | 99.4% |
| 60 | LF2 | 0.880 | 0.763 | 92.5% |
| 60 | LF3 | 0.995 | 0.932 | 0.0% |
| 120 | LF1 | 0.973 | 0.738 | 100.0% |
| 120 | LF2 | 0.871 | 0.775 | 98.1% |
| 120 | LF3 | 0.997 | 0.932 | 0.0% |

LF2 has the weakest direction stability. The 500-start sensitivity check flags disagreements in 75 of 311 refitted windows. $K=3$ is an interpretive choice; the eigenvalue-ratio diagnostic selects one dominant factor. The rotated factors are oblique and their scores can be correlated.

## Kalman comparison

The corrected comparison uses 45 windows with 48 training and 12 holdout sessions. Benchmark coefficients, PCA scaling, and state parameters are estimated on training data only.

| Method | Median holdout reconstruction MSE |
| --- | --- |
| Rolling static PCA + L1 | 0.386056 |
| Kalman filtered factors + L1 | 0.397248 |
| Kalman one-step predictions | 1.093252 |

Errors are measured in training-standardized units. Static and filtered reconstructions use the current holdout observation; one-step predictions use only preceding observations. The filtered reconstruction has a higher median error than static PCA in this run. This comparison does not show a forecasting advantage for Kalman factors.

## Stage 16: network after factor removal

Stage 16 forecasts realized variance after removing SPY/XLF and the full banking factor space. Lower QLIKE is better.

| Specification | Own-HAR QLIKE | Network-HAR 1SE QLIKE | Improvement |
| --- | --- | --- | --- |
| 120-session factor, expanding HAR | 0.203963 | 0.203903 | 0.030% |
| 120-session factor, rolling-252 HAR | 0.203854 | 0.202081 | 0.870% |
| 60-session factor, expanding HAR | 0.199703 | 0.200429 | -0.364% |

The primary HAC p-values are 0.5535 and 0.5464 at lags 5 and 20. No primary directed HAR term reaches the 70% stability threshold. The 60-session sensitivity has eight stable terms but a higher QLIKE loss. Stable selection alone does not imply a forecast gain.

Scoring applies the 90% valid-bar rule after forecast issuance:

| Specification | Issued target dates | Scored target dates | Coverage |
| --- | --- | --- | --- |
| 120-session factor, expanding HAR | 529 | 478 | 90.4% |
| 120-session factor, rolling-252 HAR | 529 | 478 | 90.4% |
| 60-session factor, expanding HAR | 589 | 520 | 88.3% |

Stage 15 remains an earlier network comparison. The results above use Stage 16's forecast timing and delayed scoring rules.

## Stage 17: factor predictability

The target retains the banking component after SPY/XLF removal. Models use the same forecast keys and training mask.

| Specification | Own-HAR | Aggregate-Factor HAR | Local-Factor HAR | Aggregate improvement |
| --- | --- | --- | --- | --- |
| 120-session factor, expanding HAR | 0.171383 | 0.164923 | 0.167543 | 3.770% |
| 120-session factor, rolling-252 HAR | 0.170997 | 0.166010 | 0.173207 | 2.916% |
| 60-session factor, expanding HAR | 0.171903 | 0.165398 | 0.168505 | 3.784% |

Aggregate-Factor HAR improves 11 of 12 banks in the primary comparison. Local-Factor HAR changes QLIKE relative to the aggregate model by +1.588%, +4.336%, +1.878% in the table's order; positive values mean a higher loss.

The predeclared aggregate-versus-own comparisons give:

| Specification | HAC lag | p-value | Holm-adjusted p-value |
| --- | --- | --- | --- |
| 120-session factor, expanding HAR | 5 | 0.005767 | 0.0346 |
| 120-session factor, expanding HAR | 20 | 0.007889 | 0.04734 |
| 120-session factor, rolling-252 HAR | 5 | 0.05354 | 0.2141 |
| 120-session factor, rolling-252 HAR | 20 | 0.06038 | 0.2415 |
| 60-session factor, expanding HAR | 5 | 0.0009528 | 0.005717 |
| 60-session factor, expanding HAR | 20 | 0.0009009 | 0.005405 |

Holm adjustment covers the six declared comparisons within each specification and HAC lag. Historical model selection still limits the interpretation.

## Pending validation

The historical pipeline has been rerun on market data. The next empirical step is the frozen 252-session prospective comparison. These results concern volatility predictability and loading patterns; they do not establish causal shocks, contagion, or tradable return alpha.

See [README.md](README.md) for execution and [OOS_PROTOCOL.md](OOS_PROTOCOL.md) for forecast definitions and evaluation. Detailed run tables and reports are saved locally under `alpaca_us_banks_1m/reports/`.
