# OOS Volatility Forecasting: Stages 16 and 17

This document collects the forecasting rules for both OOS stages. The code and each run's saved configuration define the executable specification. Results are reported separately in [RESULTS_TO_DATE.md](RESULTS_TO_DATE.md).

The historical sample already informed factor discovery and model design, so the replay is **pseudo-OOS**. Prospective confirmation requires a protocol frozen before the evaluated outcomes become observable.

## What each stage tests

| Stage | Target | Main comparison | Protocol version |
| --- | --- | --- | --- |
| 16 | Next-session RV after removing SPY/XLF and the full banking factor space | Network HAR versus Own-HAR | `oos-har-network-v2.0.0` |
| 17 | Next-session RV after removing only SPY/XLF | Aggregate-Factor HAR versus Own-HAR | `oos-factor-har-v1.0.0` |

Stage 16 tests the network left after factor removal. Stage 17 keeps the banking component in the target and tests its predictive value. The two targets answer different questions and their loss levels are not directly comparable.

The fixed CORE universe is JPM, BAC, WFC, C, USB, TFC, KEY, RF, FITB, CFG, HBAN, MS. SPY and XLF are controls. CORE plus the controls are selected before complete-case cleaning; missing stocks outside CORE do not reduce this sample. This is a fixed research universe, not a point-in-time universe backtest.

## Architecture and artifacts

| Component | Responsibility |
| --- | --- |
| `src/16_oos_har_network_validation.py`, `src/17_oos_factor_augmented_har.py` | Run modes, configuration, and artifact persistence |
| `src/utils/oos_common.py` | Exchange clocks, strict returns, calendar alignment, hashes |
| `src/utils/factor_adjusted_residuals.py` | Training-only benchmark/PCA fits and application to later blocks |
| `src/utils/oos_har_network.py` | Factor/RV vintages, Stage 16 issuance, delayed scoring |
| `src/utils/factor_har_oos.py`, `src/utils/network_har.py` | Matched Stage 17 models, HAR estimation, chronological Lasso tuning |
| `src/reporting/` | Tables, figures, and run summaries |

Generated artifacts live under `alpaca_us_banks_1m/reports/` and stay outside Git. Each run saves its configuration and provenance, factor vintages, coefficients, tuning diagnostics, and forecast ledger. Realized outcomes and losses belong to the later score ledger. The saved forecast hash protects issued predictions from accidental rewriting.

## Information set and preprocessing

At session $d$'s close, predictors may use accepted observations through that close. Training labels must already be observable; the forecast targets the next exchange session. Session logic uses `America/New_York`, including daylight saving and early closes.

$$
\max_{s\in\mathcal T_d}\operatorname{available\_at}(y_s)
\leq \operatorname{as\_of}_d
< \operatorname{open}_{d+1}.
$$

Prospective issuance must also occur after the origin close and before the target open, with the target open strictly after the recorded UTC freeze. Already opened targets are not backfilled as prospective forecasts.

Only consecutive, same-session one-minute returns are accepted. Overnight moves, gap-spanning returns, and the returns at and after imputed candles are excluded. Prices are forward-filled within sessions only. The 24 January 2023 session and the configured 5 June 2023 feed gap are excluded; genuine extreme returns are not mechanically clipped.

Factor fits use the preceding 120 sessions, with 60 sessions as a robustness check, and update every five sessions. Benchmark coefficients, centering, scaling, and the $K=3$ PCA basis are fitted on training data and held fixed for the later scoring block.

## PCA projector and realized variance

Let $V_3$ contain the three retained orthonormal eigenvectors. For a column vector of standardized benchmark residuals $z_t$, with training standard deviations in the diagonal matrix $D$,

$$
P_3=V_3V_3^{\top},
\qquad
u_t=D\,(I-P_3)z_t.
$$

This defines the Stage 16 factor-adjusted residual. In the code's row-matrix convention the same operation is $U=Z(I-P_3)D$. L1 rotation changes the coordinates within the retained space; it does not change this projector. A rotation failure is recorded separately from the PCA residual target.

Residuals are summed into non-overlapping five-minute bins and squared within each session:

$$
RV_{i,d}=\sum_{b\in d}\left(\sum_{t\in b}u_{i,t}\right)^2,
\qquad
y_{i,d}=\log\!\left(\max(RV_{i,d},10^{-16})\right).
$$

Each bin requires all five consecutive valid minute returns. Scoring requires at least one valid bin and at least 90% valid-bar coverage. These rules determine eligibility after realization; they do not use future quality to decide whether to issue a forecast. Missing, non-positive, and ineligible outcomes retain an explicit status. QLIKE uses raw positive RV, not a clipped proxy.

## HAR models and tuning

Daily, weekly, and monthly features are means of log variance on exchange-session positions:

$$
H_d(x)=\left(
x_d,\quad \frac{1}{5}\sum_{j=0}^{4}x_{d-j},\quad
\frac{1}{22}\sum_{j=0}^{21}x_{d-j}
\right).
$$

A missing session stays on the calendar and invalidates each weekly or monthly lookback crossing it. Own-HAR uses only the target bank's history. Network HAR adds 33 terms from the other eleven banks. Own terms and the intercept are unpenalized; cross-bank terms are partialled out and penalized.

The primary specification combines a 120-session factor window with expanding HAR and at least 252 valid training equations. Robustness specifications use rolling-252 HAR or a 60-session factor window with expanding HAR.

Lasso uses three chronological training folds, refreshed every 20 issued forecasts. Candidates are $\{0.01,0.03,0.10,0.30,1.00\}\alpha_{\max}$, with standardization and $\alpha_{\max}$ recomputed inside each fold. The primary rule chooses the strongest penalty within one standard error of the minimum validation log-MSE. Stage 16 also records the minimum-loss model and persistence. Fallbacks, convergence, conditioning, and clipping are recorded explicitly; log-to-variance smearing uses training residuals only.

## Stage 17 factor predictors

Write $RV^B$ for benchmark-residual variance, before banking-factor removal. Using the same column-vector convention for $z_t$, frozen PCA basis $V_d$, and L1 loading matrix $\Lambda_d$, scores are

$$
f^{PC}_{t,d}=V_d^{\top}z_t,
\qquad
f^{LF}_{t,d}=(\Lambda_d^{\top}\Lambda_d)^{-1}\Lambda_d^{\top}z_t.
$$

Local columns are aligned through time using permutation and sign. The factor-variance measures are

$$
FV_{k,d}=\sum_{b\in d}\left(\sum_{t\in b}f_{k,t,d}\right)^2,
\qquad
CFV_d=\sum_{k=1}^{3}FV^{PC}_{k,d}.
$$

$CFV_d$ is invariant to orthogonal rotations of the PCA basis. Local-factor variances depend on the chosen sparse coordinates.

| Model | Predictors beyond own-bank HAR |
| --- | --- |
| Aggregate-Factor HAR | Daily, weekly, monthly $\log CFV$ |
| Local-Factor HAR | Daily, weekly, monthly log-variance of each of the three local factors |
| Network HAR | Penalized histories of the other banks |
| Hybrid HAR | Unpenalized local-factor histories plus penalized other-bank histories |
| Persistence | Origin variance carried forward, without a fitted HAR equation |

Together with Own-HAR, these six models share a common finite training mask and identical forecast keys. The prospective primary comparison is Aggregate-Factor HAR versus Own-HAR; localization and network comparisons remain diagnostics.

## Evaluation and execution

The primary loss is

$$
\operatorname{QLIKE}(RV,\widehat{RV})=
\frac{RV}{\widehat{RV}}
-\log\!\left(\frac{RV}{\widehat{RV}}\right)-1.
$$

Loss differences are model A minus model B: negative values favor A. Inference averages stocks within target date before applying Bartlett-HAC at lags 5 and 20. Stage 17 uses Holm adjustment across predeclared comparisons. Stage 16 adds a paired moving-block bootstrap with 2,000 replications and block lengths 20/5; its optional edge bootstrap uses 200 replications, conditional on generated features and fixed penalties. Directed edges represent conditional predictability, not causal contagion.

Both scripts accept `--mode smoke`, `audit`, `freeze`, `prospective`, and `score`. Stage 16 also supports `report`. Run modes are:

| Mode | Purpose |
| --- | --- |
| `smoke` | Last 180 sessions with lighter settings; still requires local market data |
| `audit` | Historical pseudo-OOS replay; `--resume` continues compatible checkpoints |
| `freeze` | Save the prospective protocol and actual UTC freeze timestamp |
| `prospective` | Issue eligible forecasts without reading target outcomes |
| `score --run-id RUN_ID` | Join outcomes once observable, preserving the forecast ledger |

Use `python src/<stage>.py --help` for arguments. Checkpoint compatibility follows model settings, data/calendar prefixes, and run identity; worker count and checkpoint cadence are operational settings. The prospective horizon is 252 sessions, with descriptive checks at 63 and 126, not optional stopping rules. A frozen configuration must not be changed retrospectively to improve evaluated outcomes.
