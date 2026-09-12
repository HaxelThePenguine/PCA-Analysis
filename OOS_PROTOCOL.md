# Stage 16 OOS HAR and Network HAR Protocol

This document registers the corrected Stage 16 procedure implemented in
`src/16_oos_har_network_validation.py`. The protocol version is
`oos-har-network-v2.0.0`. Historical execution is labelled
`historical_pseudo_oos`; it is a replay of an already explored sample and is
not an untouched holdout. The prospective track is frozen at the recorded UTC
timestamp and is the only track that can produce genuinely new confirmation
evidence.

## Research question and universe

The question is whether lagged cross-stock realized-variance information
improves one-session-ahead forecasts of factor-adjusted realized variance
beyond each stock's own HAR dynamics. The confirmation universe is frozen at
the twelve CORE stocks, in this order:

| Position | Stock |
| ---: | --- |
| 1 | JPM |
| 2 | BAC |
| 3 | WFC |
| 4 | C |
| 5 | USB |
| 6 | TFC |
| 7 | KEY |
| 8 | RF |
| 9 | FITB |
| 10 | CFG |
| 11 | HBAN |
| 12 | MS |

SPY and XLF are benchmark controls and are not forecast targets. This is a
fixed historical research universe, not a point-in-time universe backtest.

## Information clock

All timestamps are represented in `America/New_York` for session logic. At the
close of exchange session (d), the procedure may use accepted observations
through that close to issue a forecast for session (d+1). The ledger records
`as_of`, the next-session open, the target outcome availability timestamp, the
training-label cutoff, and the last training-label availability timestamp.
Every issued row satisfies

\[
\max\{\operatorname{available\_at}(y_s):s\in\mathcal T_d\}
\leq \operatorname{as\_of}_d
< \operatorname{open}_{d+1}.
\]

The target row is absent from the issuance function. The historical path is a
causal replay because the current target is not read while its forecast is
constructed; it is still marked pseudo-OOS because the sample informed prior
research choices. In prospective mode, a target session is eligible only when
its open is strictly after the actual recorded protocol freeze and strictly
after the recorded forecast-issuance timestamp. An already opened or already
observed target is never reconstructed retrospectively and labelled
prospective.

## Preprocessing and target construction

The input prices are the synchronized repository close matrix together with
its missingness mask. One-minute log returns are accepted only when both price
endpoints are consecutive one-minute observations in the same exchange
session. The first return of each session is unavailable, so the overnight
close-to-open move is never included. Missing intervals remain unavailable;
there is no interpolation, backward fill, cross-gap differencing, or clipping
of genuine extreme returns. The contamination mask from the existing pipeline
is retained and applied after strict differencing. The known bad session
2023-01-24 is excluded, and the configured 2023-06-05 systemic gap cannot
create a return spanning the removed timestamps.

For each factor vintage, SPY/XLF residualization, centering, scaling, PCA and
the retained (K=3) subspace are fitted only on completed preceding sessions.
The fitted transformation receives a factor-fit identifier and is applied only
to its later score block. The L1 rotation is retained for interpretation and
loading reports. Because it is a nonsingular rotation of the retained PCA
space, residual construction uses the PCA projector directly:

\[
P_3=V_3V_3^\top,
\qquad
u_t=D\,(I-P_3)z_t.
\]

The implementation records projector discrepancy, residual orthogonality,
rotation status, optimizer success rate and condition diagnostics. A rotation
failure therefore affects interpretation metadata, not the definition of the
full-subspace residual target.

Residual returns are grouped into non-overlapping five-minute intervals. An
interval is valid only when it contains exactly five consecutive valid
one-minute returns within one session. For stock (i) and session (d), the
realized target is

\[
RV_{i,d}=\sum_{b\in d}\left(\sum_{t\in b}u_{i,t}\right)^2,
\qquad
y_{i,d}=\log\left(\max(RV_{i,d},\varepsilon)\right),
\]

with the repository's variance floor

\[
\varepsilon=10^{-16}.
\]

The raw positive realized variance is retained for scoring. The floor is a
diagnostic and is never used to turn a non-positive observed target into a
positive target silently.

Target quality is determined before confirmation by a fixed rule: at least one
valid five-minute interval and a valid-bar fraction of at least 0.90. This
rule governs scoring eligibility after realization. It cannot prevent a
forecast from being issued at the preceding origin. Missing or low-quality
targets remain in the score ledger with an explicit status.

## HAR information set

Daily observations are first reindexed to the exchange-session calendar. Thus,
a missing trading session remains a missing calendar position rather than
collapsing the next-session target onto the next available row. The features
use means of log variance, not the logarithm of an average variance:

\[
x_{i,d,D}=y_{i,d},
\qquad
x_{i,d,W}=\frac{1}{5}\sum_{k=0}^{4}y_{i,d-k},
\qquad
x_{i,d,M}=\frac{1}{22}\sum_{k=0}^{21}y_{i,d-k}.
\]

The ranges follow exchange-session positions and include both endpoints. The
own-HAR equation is

\[
y_{i,d+1}=a_i+\sum_{h\in\{D,W,M\}}b_{i,h}x_{i,d,h}+e_{i,d+1},
\]

and the Network HAR extension is

\[
y_{i,d+1}=a_i+\sum_hb_{i,h}x_{i,d,h}
 +\sum_{j\neq i}\sum_h\gamma_{i,j,h}x_{j,d,h}+e_{i,d+1}.
\]

The intercept and the three own terms are unpenalized. The 33 cross-stock
terms are penalized after partialling out the intercept and own-HAR terms. A
selected directed edge (j\to i) is a conditional predictive relationship;
it is not a structural causal or contagion claim.

The primary HAR fit is expanding with at least 252 usable equations. The
retained robustness specifications are a 252-session rolling HAR window with
the 120-session factor vintage and an expanding HAR window with the
60-session factor vintage. Each specification is compared against its own
factor-adjusted target and own-HAR benchmark.

## Lasso tuning and numerical controls

Penalty tuning occurs every 20 eligible origins according to the fixed
schedule. The candidates are

\[
\alpha\in\{0.01,0.03,0.10,0.30,1.00\}\alpha_{\max},
\qquad
\alpha_{\max}=\frac{1}{n}
\max_k\left|X_{\mathrm{std},k}^{\top}y_{\mathrm{res}}\right|.
\]

The cross design is standardized inside each chronological training fold and
again on the full training sample. The absolute alpha value, not only the
fraction, is recomputed on the corresponding sample. Three chronological inner
folds are retained. Minimum validation log-MSE is a tuning objective; final
forecast evaluation remains QLIKE. The primary penalty is the strongest
candidate within one standard error of the minimum validation log-MSE. The
minimum-loss candidate is a secondary model.

The coordinate-descent solver records convergence, iteration count and maximum
KKT violation. Reaching `max_iter` is not treated as convergence. If no valid
chronological validation loss exists, the strongest candidate is selected with
an explicit fallback status and reason. Training-only residual smearing is
used to map the log forecast to variance units:

\[
\widehat{RV}_{i,d+1}=\exp(\widehat y_{i,d+1})
\,\overline{\exp(e_{i,\mathcal T_d})}.
\]

Exponent clipping and smearing clipping counts are recorded separately.

## Evaluation and inference

The primary loss is

\[
QLIKE(RV,\widehat{RV})=
\frac{RV}{\widehat{RV}}-\log\left(\frac{RV}{\widehat{RV}}\right)-1.
\]

It is evaluated only for positive, quality-eligible observed variance, using
the raw observed variance. The primary comparison is Network HAR 1-SE minus
own HAR on identical `(specification, target date, stock)` keys. The date-level
differential is

\[
\delta_d=\frac{1}{N_d}\sum_i
\left[QLIKE_{i,d}^{Network}-QLIKE_{i,d}^{Own}\right].
\]

Inference is performed over dates after averaging the stock panel within each
date. Twelve stocks on one day are not treated as twelve independent
replications. Bartlett HAC uses lag 5 primarily and lag 20 as a fixed
sensitivity. A paired circular moving-block bootstrap uses 2,000 replications,
block lengths 20 and 5, and the same date-index vector for every stock in a
replication. Percentile intervals describe the observed loss series. The
reported bootstrap p-value, when present, is computed from the centered null
distribution.

Per-stock inference is secondary and is reported with Benjamini–Hochberg and
Holm adjustments. The conditional edge bootstrap uses 200 replications at a
small predeclared set of checkpoints, holds the observed penalty fixed and
resamples training rows jointly across target equations. It is uncertainty
conditional on generated features and fixed penalties, not a full re-estimation
bootstrap of factors, tuning and graph selection.

These choices draw on the HAR motivation in [Corsi (2009)](https://academic.oup.com/jfec/article-abstract/7/2/174/856522), the volatility-proxy and loss-function discussion in [Patton (2011)](https://doi.org/10.1016/j.jeconom.2010.03.034), and the predictive-procedure comparison framework of [Giacomini and White (2006)](https://doi.org/10.1111/j.1468-0262.2006.00718.x). Their assumptions do not turn this nested, expanding, penalized experiment into a guarantee of textbook Diebold–Mariano or conditional-predictive-ability validity.

## Artifacts, freeze and execution

Every run writes a machine-readable manifest containing protocol and
configuration hashes, code hashes, freeze timestamp, last data examined,
universe, target convention, preprocessing rules, schedules, candidate
penalties, evaluation endpoint, data provenance, origin dates and run status.
Forecast ledgers contain no realized target or loss columns and are protected
by a stored SHA-256 digest. Score ledgers are created only by the later scoring
mode, which rebuilds eligible realized outcomes from the then-current market
data without rewriting the forecast ledger. Factor-fit metadata, RV-vintage
metadata, coefficients, edge histories, tuning outcomes and failure
diagnostics are written beside model, stock, period, HAC, bootstrap and graph
summaries. Generated artifacts remain outside Git.

On Windows PowerShell, the main commands are:

```powershell
& .\.venv\Scripts\python.exe src\16_oos_har_network_validation.py --mode smoke --no-figures --n-jobs 2
& .\.venv\Scripts\python.exe src\16_oos_har_network_validation.py --mode audit --no-figures --n-jobs 2 --run-id historical_oos_v2
& .\.venv\Scripts\python.exe src\16_oos_har_network_validation.py --mode freeze
& .\.venv\Scripts\python.exe src\16_oos_har_network_validation.py --mode prospective --no-figures --n-jobs 2
& .\.venv\Scripts\python.exe src\16_oos_har_network_validation.py --mode score --run-id PROSPECTIVE_RUN_ID
& .\.venv\Scripts\python.exe src\16_oos_har_network_validation.py --mode report --run-id RUN_ID
```

The equivalent POSIX invocation replaces the interpreter path with
`./.venv/bin/python` and uses forward slashes. The audit run may resume with
`--resume`: existing compatible specification checkpoints are resumed, while
specifications not yet started begin normally. Compatibility is determined by
the forecast-generating statistical settings and the data and calendar
prefixes. Operational changes such as worker count and checkpoint cadence do
not invalidate a checkpoint, whereas model, protocol, data, calendar or run
identity changes do. A completed run is never silently recomputed from an
incompatible checkpoint.

The confirmation endpoint is 252 exchange sessions, with descriptive checks
scheduled at 63 and 126 sessions. These are design checkpoints, not optional
stopping rules and not a power calculation. If no eligible post-freeze target
exists, the prospective status remains `ready / awaiting_future_data` or
`awaiting_future_data`.
