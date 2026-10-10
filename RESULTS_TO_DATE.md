# Results to Date

Results recorded on 13 September 2026, using market data through 9 September 2026.

These runs predate the October changes to CORE filtering, calendar alignment, and Kalman training fits. The figures below still need to be recomputed on market data with those changes. Synthetic tests check the code but cannot validate the empirical results.

## Forecasting result

The aggregate banking factor is the strongest historical forecasting signal. In Stage 17, it improves pooled QLIKE by roughly 5% over Own-HAR across the three specifications. Sparse local factors help interpret the loading space but do not outperform the aggregate representation. After removing the complete factor space, Stage 16 finds little stable network signal.

All results are historical pseudo-OOS: the 2023–2026 sample informed factor discovery and model design. Prospective confirmation is pending.

## Common variation and local factors

The recorded sample contains 924 regular trading sessions and the twelve CORE stocks. SPY/XLF account for 43.59% of raw CORE variance.

| Correlation PCA | PC1 explained variance | First three components |
| --- | ---: | ---: |
| Raw returns | 63.77% | 75.85% |
| Intraday-normalized returns | 64.47% | 75.81% |
| SPY/XLF residuals | 35.71% | 56.11% |

Intraday normalization changes little. Benchmark removal reduces common variation but leaves a structured banking subspace.

The full-sample $K=3$ L1 rotation preserves that subspace and its reconstruction. With $h_p=1/\log(12)=0.4024$, a direction needs more than seven small loadings to pass the reference local-factor rule.

| Direction | Active support at the reference threshold | Small loadings | Interpretation |
| --- | --- | ---: | --- |
| LF1 | WFC, C, MS | 9 | Formally local; concentrated on MS/C |
| LF2 | JPM, BAC, WFC, MS | 8 | Formally local; C/MS boundary is unstable |
| LF3 | JPM, USB, TFC, KEY, RF, FITB, CFG, HBAN | 4 | Broad regional-bank cluster, not formally local |

Whole-session bootstrap median loading cosines are 0.9991, 0.9977, and 0.9999. LF2's fifth percentile falls to 0.6891, while LF1 and LF3 remain above 0.996. The broad regional direction is the most stable even though it is not formally local.

## Rolling stability

Stage 14 uses 174 trailing 60-session windows and 162 trailing 120-session windows, advancing by five sessions. These are overlapping descriptive windows, not independent observations.

| Window | Factor | Mean past-only cosine | Mean support Jaccard | Local-factor rate |
| ---: | --- | ---: | ---: | ---: |
| 60 | LF1 | 0.960 | 0.656 | 99.4% |
| 60 | LF2 | 0.881 | 0.767 | 92.5% |
| 60 | LF3 | 0.995 | 0.932 | 0.0% |
| 120 | LF1 | 0.973 | 0.738 | 100.0% |
| 120 | LF2 | 0.871 | 0.775 | 98.1% |
| 120 | LF3 | 0.997 | 0.932 | 0.0% |

LF2 is the least stable identification: 77 of 305 windows refitted with 500 starts disagree with the 200-start fit. $K=3$ is an interpretive choice; the eigenvalue-ratio diagnostic selects one dominant factor. The factors are oblique and their scores can be correlated.

The earlier Kalman holdout comparison used full-sample benchmark coefficients. It needs to be rerun with training-only coefficients before comparing model performance.

## Stage 16: network after factor removal

Stage 16 forecasts realized variance after removing SPY/XLF and the full banking factor space.

| Specification | Own-HAR QLIKE | Network-HAR 1SE QLIKE | Improvement |
| --- | ---: | ---: | ---: |
| 120-session factor, expanding HAR | 0.240384 | 0.240250 | 0.056% |
| 120-session factor, rolling-252 HAR | 0.242541 | 0.239269 | 1.349% |
| 60-session factor, expanding HAR | 0.240483 | 0.240246 | 0.098% |

The primary gain is small: HAC p-values are 0.138 and 0.061 at lags 5 and 20. The rolling-252 gain is larger but remains near 0.10 in HAC/bootstrap inference. The 60-session gain is statistically detectable, with HAC p-values 0.0145/0.0051, but economically small.

No primary directed edge reaches the 70% stability threshold. The earlier Stage 15 gain of roughly 3.5–4.6% is superseded by this corrected experiment.

The 90% valid-bar rule leaves 199 of 529 forecast dates for the 120-session designs and 205 of 589 for the 60-session design. This 35–38% coverage limits the interpretation and must be reassessed after the CORE-filtering fix.

## Stage 17: factor predictability

The target retains the banking component after SPY/XLF removal. All models use the same forecast keys and training mask.

| Specification | Own-HAR | Aggregate-Factor HAR | Local-Factor HAR | Aggregate improvement |
| --- | ---: | ---: | ---: | ---: |
| 120-session factor, expanding HAR | 0.210206 | 0.199614 | 0.206196 | 5.039% |
| 120-session factor, rolling-252 HAR | 0.213218 | 0.202685 | 0.216944 | 4.940% |
| 60-session factor, expanding HAR | 0.214540 | 0.204428 | 0.209091 | 4.713% |

Aggregate-Factor HAR improves ten of twelve banks in the primary and 60-session designs. Citigroup is the consistent exception; WFC is roughly neutral to negative. The pooled daily win rate is 51–54%, so the gain mainly reflects fewer severe errors rather than wins on almost every date.

Holm-adjusted HAC p-values for the primary comparison are 0.134 at lag 5 and 0.047 at lag 20. The 60-session sensitivity survives correction at both lags, with 0.009/0.004. Rolling-252 does not survive multiplicity correction.

Local-Factor HAR is worse than Aggregate-Factor HAR by 3.297%, 7.035%, and 2.281% across the three designs. Its nine factor-history regressors also create more collinearity: median condition numbers range from about 906 to 2,425. The hybrid reaches about 27,703–316,426. The aggregate signal is more parsimonious and stable.

## Pending validation

The next empirical step is the frozen 252-session prospective comparison of Aggregate-Factor HAR against Own-HAR, after rerunning the corrected pipeline on real data. The current record supports volatility predictability and descriptive loading patterns; it does not establish causal shocks, contagion, or tradable return alpha.

See [README.md](README.md) for execution and [OOS_PROTOCOL.md](OOS_PROTOCOL.md) for forecast definitions and evaluation. Run tables and reports are saved locally under `alpaca_us_banks_1m/reports/`.
