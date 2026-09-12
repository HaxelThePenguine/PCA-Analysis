# Stage 17 Factor-Augmented HAR Protocol

## Research question

Stage 17 tests whether the three previously identified banking factors contain
incremental information about next-session bank volatility in a genuinely
out-of-sample forecasting design. This is deliberately narrower than asking
whether the factors explain contemporaneous returns. A factor may explain a
large fraction of today’s covariance and nevertheless add no predictive
content once a bank’s own heterogeneous volatility history is known.

The executable boundary follows the same structure as the frozen Stage 16
protocol. `src/17_oos_factor_augmented_har.py` orchestrates protocol states and
artifacts, `src/utils/factor_har_oos.py` defines the matched causal forecasting
design, `src/utils/oos_common.py` supplies the shared clock, calendar, strict
preprocessing, hashing, and persistence contracts, and
`src/reporting/factor_har.py` owns summaries, HAC comparison tables, and the
run report. Presentation code therefore does not participate in model fitting.

The response for bank (i) is benchmark-residual realized variance. At every
minute (t), the return is first residualized on SPY and XLF using coefficients
estimated from sessions strictly preceding the scoring block. Five-minute
benchmark-residual returns are then squared and summed within session (d),
producing (RV^B_{i,d}). The forecast response is

\[
y^B_{i,d+1}=\log RV^B_{i,d+1}.
\]

The target deliberately retains the internal banking factor component. It is
therefore suitable for testing whether the factors forecast bank volatility;
the factor-adjusted target used in Stages 15 and 16 would remove precisely the
variation under investigation.

## Causal factor vintages

At an update origin (d), benchmark residuals from the preceding (W\in
\{120,60\}) exchange sessions are standardized using training-window moments.
Correlation PCA estimates a three-dimensional subspace with orthonormal basis
(V_d\). The basis is frozen and applied only to later observations in the
five-session scoring block. The PCA scores are

\[
f^{PC}_{t,d}=z_tV_d.
\]

The sparse local loading matrix (\Lambda_d\) is the L1 rotation of that same
subspace. Its columns are aligned through time by permutation and sign, and the
corresponding oblique coordinates are

\[
f^{LF}_{t,d}=z_t\Lambda_d(\Lambda_d^\top\Lambda_d)^{-1}.
\]

Consequently,

\[
f^{LF}_{t,d}\Lambda_d^\top
=z_t\Lambda_d(\Lambda_d^\top\Lambda_d)^{-1}\Lambda_d^\top,
\]

which is the projection of standardized benchmark residuals onto the retained
banking subspace. No full-sample factor score, loading, mean, scale, or factor
label is used during forecast issuance.

For each factor coordinate (k), intraday scores are aggregated into
five-minute returns and daily realized factor variance,

\[
FV_{k,d}=\sum_{b\in d}\left(\sum_{t\in b}f_{k,t,d}\right)^2.
\]

The rotation-invariant aggregate common-factor measure is

\[
CFV_d=\sum_{k=1}^{3}FV^{PC}_{k,d}.
\]

Because it sums squared coordinates in an orthonormal basis, (CFV_d) is
unchanged by an orthogonal rotation of the retained PCA subspace. The three
local-factor variances are intentionally rotation-dependent: they test whether
the economically localized MS/C, large-bank, and regional-bank coordinates add
information beyond the aggregate subspace.

## Forecast models

For any series (x_d), its heterogeneous autoregressive state at origin (d)
is

\[
H_d(x)=\left(x_d,\frac{1}{5}\sum_{j=0}^{4}x_{d-j},
\frac{1}{22}\sum_{j=0}^{21}x_{d-j}\right).
\]

All averages use exchange-session positions. A missing session is retained as
missing and invalidates every weekly or monthly feature whose lookback crosses
it. The own-HAR benchmark uses only (H_d(\log RV^B_i)). The aggregate model
adds (H_d(\log CFV)). The economically motivated local-factor model is

\[
y^B_{i,d+1}=\alpha_i+\beta_i^\top H_d(\log RV^B_i)
+\sum_{k=1}^{3}\gamma_{ik}^\top H_d(\log FV^{LF}_k)
+\varepsilon_{i,d+1}.
\]

A benchmark-residual Network HAR adds the daily, weekly, and monthly histories
of the other eleven banks through a partialled-out Lasso. A hybrid model keeps
own-bank and local-factor histories unpenalized and applies the same Lasso only
to other-bank histories. The Lasso penalty is selected by chronological
cross-validation under the one-standard-error rule and is refreshed every
twenty issued forecasts. Persistence is retained as a scale-free diagnostic
benchmark.

The six models use one common finite-observation mask at estimation and must
produce identical evaluation keys. This restriction prevents a model from
appearing superior because it silently drops difficult dates or banks. The
primary specification uses a 120-session factor window and an expanding HAR
window with at least 252 valid training observations. A 252-session rolling HAR
window and a 60-session factor window are robustness specifications fixed in
advance.

## Identification of incremental predictive content

Aggregate-Factor HAR versus Own-HAR tests whether the retained common banking
subspace predicts future benchmark-residual variance. Local-Factor HAR versus
Aggregate-Factor HAR tests whether sparse economic localization adds predictive
content beyond the rotation-invariant magnitude of the subspace. Network HAR
versus Own-HAR measures cross-bank predictability before removing the internal
factors. Hybrid versus Local-Factor HAR asks whether a residual cross-bank
network remains useful after the factor histories are observed. Hybrid versus
Network HAR measures whether factor histories improve the best directly
comparable network specification.

The historical audit treats all comparisons as predeclared diagnostics. After
observing that the aggregate-factor specification is materially more stable
than the local decomposition, the prospective primary comparison is frozen as
Aggregate-Factor HAR versus Own-HAR. This selection is based on development
data and is therefore legitimate only for future, not retrospective,
confirmation. The primary loss is QLIKE evaluated on raw positive realized variance,

\[
L(RV,\widehat{RV})=
\frac{RV}{\widehat{RV}}-\log\left(\frac{RV}{\widehat{RV}}\right)-1.
\]

Loss differences are defined as model A minus model B, so a negative mean
favors model A. Inference first averages the twelve-bank panel within target
date and then uses Bartlett HAC standard errors at lags five and twenty. Holm
adjustment is applied across the predeclared model comparisons within each
specification and lag. The existing rule requiring at least ninety percent of
expected valid intraday bars remains unchanged.

## Historical audit and prospective confirmation

The historical walk-forward replay is development evidence because the
2023–2026 sample influenced earlier factor discovery. It can reject an
unpromising model and diagnose where predictive content resides, but it cannot
constitute an untouched confirmation. The prospective protocol must be frozen
at a real UTC timestamp. A forecast is eligible only when the protocol was
frozen before the target session opened, the forecast was issued after the
origin close and before the target open, and no target observation was present
at issuance. The immutable forecast ledger is hashed before later outcomes are
joined.

Predictive success is not equivalent to return alpha. Stage 17 forecasts
variance and defines neither a position rule nor an execution model. Any later
economic-value claim requires a separately frozen portfolio mapping, exposure
constraints, turnover accounting, execution delay, and transaction costs.
