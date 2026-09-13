# Results to Date: Intraday Statistical Factor Extraction in U.S. Financials

> **Snapshot date:** 13 September 2026
>
> **Market-data endpoint:** 9 September 2026
>
> **Current status:** factor discovery and the corrected historical OOS replays are complete; genuinely untouched prospective confirmation remains pending

## Central result

The project identifies a strong common banking-volatility state and a stable, economically legible organization of its residual loading space. After removing SPY and XLF, an oblique L1 rotation of the first three residual principal components repeatedly separates an MS/C direction, a large-bank direction, and a broad regional-bank direction. This localization is descriptively useful, but the predictive evidence belongs to the aggregate subspace rather than to the three labels individually.

In the corrected Stage 17 historical walk-forward replay, Aggregate-Factor HAR reduces pooled QLIKE relative to Own-HAR by 5.039% in the primary 120-session expanding specification. The sign and approximate magnitude survive both the rolling-252 HAR design and the 60-session factor-window sensitivity. Local-Factor HAR never outperforms the rotation-invariant aggregate specification. After the entire three-dimensional factor space is removed, Stage 16 finds only a small and unstable residual-network gain. The coherent conclusion is therefore that common banking volatility is a useful predictive state variable, whereas neither a persistent post-factor network nor three separately predictive local factors is currently supported.

These are historical pseudo-OOS results. The 2023–2026 sample influenced factor discovery and protocol development, so no retrospective procedure can turn it into an untouched holdout.

## Research sample and data integrity

The sample contains 924 regular U.S. trading sessions from 1 January 2023 through 9 September 2026. The primary CORE universe consists of JPM, BAC, WFC, C, USB, TFC, KEY, RF, FITB, CFG, HBAN, and MS. SPY and XLF serve as broad controls; six additional financial stocks remain available in the FULL robustness universe.

| Data decision | Production treatment |
| --- | --- |
| Intraday frequency | One-minute SIP bars during the official regular session |
| Overnight return | Excluded by computing returns within session |
| Missing price | Previous-tick synchronization with both affected returns masked |
| Systemic feed gap | Four minutes on 5 June 2023 removed globally |
| Known invalid session | 24 January 2023 removed in full |
| March 2023 banking stress | Retained as an economically meaningful episode |
| Extreme observations | Investigated rather than mechanically winsorized |

This treatment produces a synchronized return panel without converting missing observations into artificial zero returns. It also preserves a direct audit trail from raw SIP bars to every factor and forecast input.

## Common variation and benchmark removal

The raw CORE panel is dominated by one common direction. Intraday normalization barely changes the result, which rules out the deterministic open-and-close volatility profile as its main explanation.

| Specification | PC1 explained variance | First three cumulative variance |
| --- | ---: | ---: |
| Raw correlation PCA | **63.77%** | **75.85%** |
| Raw covariance PCA | 64.73% | 76.91% |
| Intraday-normalized correlation PCA | **64.47%** | **75.81%** |
| SPY/XLF residual correlation PCA | **35.71%** | **56.11%** |
| SPY/XLF residual covariance PCA | 41.79% | 59.41% |

The joint SPY/XLF fit accounts for 43.59% of raw CORE variance. Residual PC1 contributes another 23.58% of raw variance, residual PC2–PC3 contribute 9.94%, and the remaining residual components contribute 22.90%, completing the variance ledger at 100%. Benchmark removal therefore reduces the common component materially but leaves a structured internal banking space.

The descriptive 20- and 60-session rolling PCA confirms that this structure is time varying. Raw correlation-PC1 ranges from 49.97% to 86.60% in 20-session windows and from 54.93% to 81.77% in 60-session windows. After SPY/XLF removal, the corresponding ranges fall to 24.55–48.03% and 26.50–43.51%. These overlapping windows locate variation; they do not provide independent observations or formal break tests.

## Sparse factor localization

The full-sample local-factor stage begins from the residual correlation-PCA space with $K=3$ and selects a sparse coordinate system by minimizing the total L1 loading norm. Because the rotation is nonsingular, it preserves the selected projector and the 56.1085% residual reconstruction share. The maximum production reconstruction discrepancy is below $3\times10^{-14}$.

With twelve stocks, the reference small-loading threshold is $h_n=1/\log(12)=0.4024$, and a direction must contain more than seven small loadings to pass the five-percent local-factor rule. The estimated small-loading counts are 9, 8, and 4.

| Direction | Active support under the reference threshold | Statistical interpretation |
| --- | --- | --- |
| LF1 | WFC, C, MS | Formally local and strongly concentrated on MS/C |
| LF2 | JPM, BAC, WFC, MS | Formally local, with an unstable C/MS boundary |
| LF3 | JPM, USB, TFC, KEY, RF, FITB, CFG, HBAN | Broad regional-bank cluster; stable but not formally local |

The whole-session bootstrap resamples all 924 sessions, re-estimates PCA and the L1 rotation, and aligns each replication to the reference loading space. LF1 has median cosine 0.9991 and fifth-percentile cosine 0.9964. LF3 has median cosine 0.9999 and fifth percentile 0.9997. LF2 has median cosine 0.9977 but a much lower fifth percentile of 0.6891: JPM, BAC, and WFC remain active in every replication, while the fourth loading alternates mainly between C and MS. The evidence supports two sharply local directions and one exceptionally stable broad cluster, not three independently identified structural shocks.

The eigenvalue-ratio diagnostic selects one dominant factor in both the raw and residual panels. The local-factor test is negative at $K=2$ and positive at $K=3,4,5$. Retaining $K=3$ is therefore an interpretive and continuity decision, not a conclusive estimate of the true latent dimension.

## Rolling stability and the Kalman comparison

Stage 14 re-estimates SPY/XLF residualization, correlation PCA, and the $K=3$ L1 rotation inside 174 trailing 60-session windows and 162 trailing 120-session windows. Both grids advance by five sessions. Past-only alignment provides the causal stability view; full-sample alignment is used only for ex-post labeling.

| Window | Factor | Mean past-only cosine | Fifth-percentile cosine | Mean support Jaccard | Local-factor rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 60 | LF1 | 0.960 | 0.870 | 0.656 | 99.4% |
| 60 | LF2 | 0.881 | 0.707 | 0.767 | 92.5% |
| 60 | LF3 | 0.995 | 0.986 | 0.932 | 0.0% |
| 120 | LF1 | 0.973 | 0.897 | 0.738 | 100.0% |
| 120 | LF2 | 0.871 | 0.664 | 0.775 | 98.1% |
| 120 | LF3 | 0.997 | 0.988 | 0.932 | 0.0% |

LF3 is the most stable direction even though it remains too broad to pass the local-factor rule. LF1 is stable overall. LF2 is the main sensitivity warning: it has the lowest cosine values and the largest number of persistent instability candidates. Of 305 windows receiving a 500-start refit, 77 disagree with the primary 200-start result under the recorded comparison. Factor scores are oblique rather than independent; in 60-session windows, mean off-diagonal correlations are approximately 0.41 for LF1–LF2, 0.25 for LF1–LF3, and 0.42 for LF2–LF3.

The session-level Kalman extension is a useful negative complexity result. Across 45 holdout windows, rolling static PCA/L1 produces median residual variance of 0.378961, compared with 0.390408 for the Kalman-filtered model. The state-space specification is approximately 3.0% worse on this metric. It recovers similar economic loading groups but supplies no demonstrated holdout advantage, so the simpler rolling estimator remains preferred.

## Corrected residual-network evidence

Stage 15 originally suggested a pooled Network-HAR improvement of approximately 3.5–4.6% after factor removal. That result remains an audit artifact rather than the current claim. Stage 16 corrected strict minute-return construction, exchange-session feature alignment, target eligibility, penalty standardization, checkpoint identity, target-free issuance, and delayed outcome scoring under protocol `oos-har-network-v2.0.0`.

The corrected run `historical_oos_v2_20260912` contains 529 forecast dates for each 120-session design and 589 dates for the 60-session design. The frozen rule requiring at least 90% valid intraday bars leaves 199 and 205 eligible dates, respectively. This 35–38% date coverage is explicit and constitutes a material external-validity limitation.

| Stage 16 specification | Own-HAR QLIKE | Network-HAR 1SE QLIKE | Relative improvement |
| --- | ---: | ---: | ---: |
| 120-session factor, expanding HAR | 0.240384 | 0.240250 | 0.056% |
| 120-session factor, rolling-252 HAR | 0.242541 | 0.239269 | 1.349% |
| 60-session factor, expanding HAR | 0.240483 | 0.240246 | 0.098% |

For the primary expanding design, the mean network-minus-own QLIKE differential is -0.000133. HAC p-values are 0.138 and 0.061 at lags 5 and 20; moving-block p-values are 0.151 and 0.067 at block lengths 5 and 20. The rolling-252 improvement is larger, but its HAC and bootstrap p-values remain near 0.10. The 60-session sensitivity is statistically detectable under both dependence choices, with HAC p-values 0.0145 and 0.0051 and moving-block p-values 0.0170 and 0.0075, yet its economic improvement is only 0.098%.

Mean one-standard-error edge density is 3.71% for the primary expanding model, 8.02% for rolling-252, and 4.00% for the 60-session model. No directed edge reaches the frozen 70% stability threshold in any primary specification. The residual network therefore contains weak and time-varying incremental information, not a stable spillover graph.

## Predictive content of the factor state

Stage 17 changes the target to next-session realized variance after removing only SPY and XLF. The rolling factor fit is estimated from preceding sessions and frozen for each later five-session score block. Own-HAR, Aggregate-Factor HAR, Local-Factor HAR, Network HAR, Hybrid HAR, and persistence use a common estimation mask and identical forecast keys under protocol `oos-factor-har-v1.0.0`.

| Stage 17 specification | Own-HAR | Aggregate-Factor HAR | Local-Factor HAR | Aggregate improvement | Local improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| 120-session factor, expanding HAR | 0.210206 | 0.199614 | 0.206196 | 5.039% | 1.908% |
| 120-session factor, rolling-252 HAR | 0.213218 | 0.202685 | 0.216944 | 4.940% | -1.748% |
| 60-session factor, expanding HAR | 0.214540 | 0.204428 | 0.209091 | 4.713% | 2.540% |

Aggregate-Factor HAR improves ten of twelve banks in both the primary and 60-session specifications. Citigroup is the consistent exception, while WFC is approximately neutral to negative. The daily pooled win rate is only 51–54%, indicating that much of the QLIKE gain comes from reducing relatively severe errors rather than winning on nearly every date.

Inference is supportive but not uniformly decisive. In the primary specification, Holm-adjusted HAC p-values are 0.134 at lag 5 and 0.047 at lag 20. The 60-session sensitivity survives correction at both lags, with adjusted p-values 0.009 and 0.004. The rolling-252 comparison does not survive multiplicity correction. The stable sign and magnitude across designs are encouraging, but the evidence is not equivalent to prospective confirmation.

Sparse localization does not improve the forecast. Relative to Aggregate-Factor HAR, Local-Factor HAR is worse by 3.297%, 7.035%, and 2.281% across the three designs. Its nine factor-history regressors also create substantially more collinearity: median condition numbers range from approximately 906 to 2,425, while the hybrid model reaches approximately 27,703 to 316,426. All fits converge, but those coefficients do not support a structural interpretation. The predictive signal belongs to total volatility in the common banking subspace, not to separately forecastable MS/C, large-bank, and regional-bank components.

## Interpretation

The empirical record now supports a disciplined three-layer description. Broad market and financial-sector exposure explains a large part of minute-level variation. A lower-dimensional internal banking space remains and can be localized into economically recognizable loading patterns. The total volatility of that space predicts next-session benchmark-residual bank volatility in the historical walk-forward sample.

The evidence does not identify causal shocks, contagion, or tradable return alpha. L1 rotation selects coordinates inside a retained PCA space; it does not make the factors independent. Sparse zeros do not prove zero economic exposure. The historical replay is not an untouched holdout, and the strict target-quality rule leaves limited date coverage. These limitations do not erase the result, but they determine the strength of the claim.

## Future objectives

### 1. Frozen prospective confirmation

The immediate objective is to issue and accumulate the predeclared 252-session prospective cohort for Aggregate-Factor HAR versus Own-HAR. Factor count, estimation windows, target construction, eligibility, and inference must remain fixed before each outcome becomes observable. Stage 16 Network HAR should remain as a secondary benchmark rather than displacing the factor comparison that currently carries the stronger signal.

### 2. Measurement-robust volatility targets

A separate robustness protocol should determine whether the aggregate factor predicts continuous volatility, jumps, or estimator noise. Five-minute realized variance should be compared with bipower variation and a microstructure-robust realized-kernel or pre-averaging estimator. This branch must be labelled exploratory and must not alter the frozen prospective target.

### 3. Parsimonious forecasting extensions

Future models should preserve the low-dimensional nature of the successful signal. The most useful comparison is among Aggregate-Factor HAR, HARQ-style measurement-error adjustment, and a small model of aggregate-factor innovations. Adding many local-factor or network regressors is not justified unless nested walk-forward tuning produces a reproducible QLIKE gain without unstable conditioning.

### 4. State dependence

The aggregate-factor gain should be evaluated separately in calm, high-volatility, and jump-intensive states using thresholds fixed before scoring. The March–May 2023 crisis remains descriptive because the current burn-in eliminates it from the outer forecast. A formal stress claim requires a redesigned historical experiment with valid event-period origins or, preferably, a future stress episode observed after protocol freeze.

### 5. Economic value

Only after prospective predictive confirmation should the project translate volatility forecasts into positions, hedges, or capital allocation. That extension requires a separately frozen mapping, explicit exposure constraints, execution delay, turnover, transaction costs, and a benchmark policy. Until those elements exist, the correct output is a volatility-forecasting result rather than an alpha claim.

## Reproducibility and evidence

The executable research sequence runs from `src/07_baseline_pca.py` through `src/17_oos_factor_augmented_har.py`; acquisition and preprocessing are implemented in stages 01–06. Generated evidence lives under `alpaca_us_banks_1m/reports/` and remains outside Git because it is reproducible from the local market-data store.

Stage 16 is documented in [`OOS_PROTOCOL.md`](OOS_PROTOCOL.md) and [`OOS_AUDIT.md`](OOS_AUDIT.md). Stage 17 is documented in [`FACTOR_HAR_PROTOCOL.md`](FACTOR_HAR_PROTOCOL.md). Shared exchange clocks, strict-return construction, hashing, and persistence live in [`oos_common.py`](src/utils/oos_common.py). The complete repository suite currently contains 59 passing tests.

For the full mathematical design, execution order, and interpretation rules, see [`README.md`](README.md).
