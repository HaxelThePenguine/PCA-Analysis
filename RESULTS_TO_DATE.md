# Results to Date: Intraday Statistical Factor Extraction in U.S. Financials

> **Snapshot date:** 12 September 2026
> **Market-data endpoint:** 9 September 2026
> **Research status:** factor extraction and leakage-controlled pseudo-out-of-sample residual-volatility forecasting completed; a genuinely untouched future holdout remains pending

## Executive summary

The project now runs from quality-controlled one-minute SIP data through baseline PCA, intraday-normalization checks, SPY/XLF residualization, variance reconciliation, rolling PCA, internal factor isolation, session-level L1 local-factor identification, dynamic rolling local-factor diagnostics, a Kalman dynamic-factor comparison, and leakage-controlled network-HAR forecasting of factor-adjusted realized volatility.

The main conclusions so far are:

1. A strong common component is present in the CORE universe. Correlation PCA assigns **63.77%** of standardized variance to PC1 and **75.85%** to the first three components.
2. The result is not explained by the market clock alone. After applying the intraday volatility profile, PC1 explains **64.47%** and the first three components explain **75.81%**, almost unchanged from the raw correlation-PCA baseline.
3. SPY/XLF exposure is economically important but does not exhaust the cross-section. The combined benchmark fit accounts for **43.59%** of raw CORE variance; the residual panel still has a structured first three-component space.
4. The residual loading space can be localized into three useful working directions. Two directions pass the formal local-factor diagnostic, while the third is a highly stable but broader regional-bank direction.
5. Rolling L1 diagnostics preserve the same broad localization but reveal that LF2 is materially less stable than LF1 and LF3. This warning survives both 60-session and 120-session windows.
6. The most defensible interpretation is therefore **a three-factor working representation with two sharply local directions and one broad, stable cluster direction**, not three independently identified causal factors.
7. Removing SPY/XLF and the complete rolling `K=3` factor space does not eliminate cross-stock volatility predictability. In the primary specification, the network HAR reduces pooled QLIKE from **0.198327** to **0.191449**, an improvement of **3.47%**.
8. The forecasting gain survives both a 252-session rolling HAR window and a 60-session factor-estimation window, but it is heterogeneous across stocks. The strongest gains occur mainly in the regional-bank group, while Citi is worse under the primary expanding network specification.
9. The remaining directed edges are predictive conditional relationships, not structural contagion. Closely overlapping refits make edge-persistence rates optimistic, while the moving-block bootstrap provides more conservative support for a smaller core of links.

The factor-discovery stages remain descriptive. Stage 15 adds pseudo-out-of-sample predictive evidence, but the factor names and directed edges do not identify causal shocks, contagion, tradable alpha, or a uniquely determined true factor count. The historical sample had already been examined while constructing the factor model, so the forecasting exercise is not a pristine final holdout.

## Data and research universe

The input is Alpaca SIP one-minute bar data for 18 U.S. financial stocks plus SPY and XLF:

| Item | Current specification |
| --- | --- |
| Observation period | 1 January 2023 to 9 September 2026 |
| Trading sessions | 924 regular U.S. sessions |
| Frequency | One-minute regular-session bars |
| Instruments | 18 financial stocks, SPY, and XLF |
| Primary universe | 12-stock CORE panel |
| CORE stocks | JPM, BAC, WFC, C, USB, TFC, KEY, RF, FITB, CFG, HBAN, MS |
| Primary PCA scaling | Correlation PCA on centered, standardized returns |
| Benchmarks for residualization | SPY and XLF |

The pipeline removes the known systemic feed gap on 5 June 2023, excludes the 24 January 2023 bad session, calculates returns within sessions only, and masks returns affected by imputed prices. Extreme observations are investigated rather than winsorized automatically. The downloaded FULL universe remains available as a robustness check, while CORE is used for the main factor results because it has materially stronger coverage and a more coherent financial-stock cross-section.

## Results by stage

### 1. Data-quality foundation

The quality-control stage is part of the result, not a preprocessing footnote.

- The 5 June 2023 systemic gap affects 18 of the 20 instruments from 09:52 through 09:55 ET. Those timestamps are removed globally.
- CORE names have near-complete total coverage; the lower-coverage names are retained for the FULL robustness universe but are not allowed to drive the primary PCA geometry.
- The 24 January 2023 NYSE opening-auction malfunction is excluded as a whole session.
- Overnight moves are not mixed into one-minute intraday returns.
- Missing-bar synchronization is separated from the return sample through an explicit contamination mask.

This produces a complete, synchronized CORE return panel on which the subsequent comparisons use the same observations and stock order.

### 2. Baseline PCA: one dominant common direction

The baseline comparison is summarized below. The first three rows use the full-sample CORE panel; the residual rows use the same panel after removing the fitted SPY/XLF exposure.

| Specification | PC1 explained variance | First three cumulative variance |
| --- | ---: | ---: |
| Raw correlation PCA | **63.77%** | **75.85%** |
| Raw covariance PCA | 64.73% | 76.91% |
| Intraday-normalized correlation PCA | **64.47%** | **75.81%** |
| SPY/XLF residual correlation PCA | **35.71%** | **56.11%** |
| SPY/XLF residual covariance PCA | 41.79% | 59.41% |

The raw PC1 technical loadings are positive for all 12 CORE stocks and range from approximately 0.71 to 0.84. This is consistent with a broad common financial-stock direction rather than a component driven by one name.

The covariance and correlation results are close at the aggregate level. This means that the headline common-factor result is not simply an artifact of choosing one of the two standard PCA scalings, although the individual loading maps differ in economically relevant ways.

### 3. Intraday normalization

Volatility is highest around the market open and close, so the project estimates a minute-of-day volatility profile and repeats the PCA after scaling each stock by that profile. The full-sample result changes only marginally:

- PC1 moves from **63.77%** to **64.47%**.
- The first-three cumulative share moves from **75.85%** to **75.81%**.

The common structure therefore survives the first clock-time robustness check. The rolling output still records time variation in the strength of the common direction, so this should not be read as evidence that intraday seasonality is irrelevant in every regime.

### 4. Market and sector residualization

For each CORE stock, the return is decomposed into a fitted SPY/XLF part and a residual part. The decomposition is additive and is reconciled by the variance ledger.

| Component of raw variance | Share of raw variance |
| --- | ---: |
| Fitted SPY/XLF exposure | **43.59%** |
| Residual PC1 | **23.58%** |
| Residual PC2–PC3 | **9.94%** |
| Residual PC4–PC12 | **22.90%** |
| Total | **100.00%** |

The benchmark fit removes between **33.85%** and **57.59%** of individual-stock variance across CORE. The remaining residual variance is not a flat noise floor: residual PCA still produces a first three-component space that is strong enough to support a separate internal-factor analysis.

The residual correlation-PCA eigenvalue ratio selects one dominant residual direction, but the remaining residual components retain meaningful cross-sectional structure. This is why the report treats the first three residual directions as an interpretive working space rather than automatically declaring that exactly three latent economic factors exist.

### 5. Rolling PCA

The descriptive rolling analysis uses overlapping 20-session and 60-session windows with a five-session step. The ranges below refer to correlation PCA across all available windows.

| Transformation | Window | PC1 range | First-three cumulative range |
| --- | ---: | ---: | ---: |
| Raw | 20 sessions | 49.97–86.60% | 65.49–90.97% |
| Raw | 60 sessions | 54.93–81.77% | 68.84–87.33% |
| SPY/XLF residual | 20 sessions | 24.55–48.03% | 47.76–67.07% |
| SPY/XLF residual | 60 sessions | 26.50–43.51% | 49.36–62.29% |

The raw panel is often close to one-dimensional, while the residual panel is materially more diffuse and time-varying. This supports the research design: first remove broad benchmark exposure, then study whether the remaining structure is stable enough to warrant local-factor or network analysis.

The rolling results are descriptive diagnostics. Overlapping windows are not independent observations, and an isolated spike is a regime candidate rather than proof of a persistent factor or an external cause.

### 6. Internal factor isolation

The first three-component space was compared using standard PCA, Varimax, Lasso Sparse PCA, and Elastic-Net Sparse PCA.

For the raw panel, the three-component PCA space reconstructs **75.85%** of standardized variance. The reported Elastic-Net Sparse PCA specification reconstructs **75.67%**, a loss of approximately **0.17 percentage points**, with non-zero counts of **11, 5, and 1** across its three sparse directions.

For the SPY/XLF residual panel, the three-component PCA space reconstructs **56.11%**. The corresponding Elastic-Net Sparse PCA diagnostic reconstructs **55.75%**, a loss of approximately **0.36 percentage points**, with non-zero counts of **8, 4, and 2**. At the reported penalty, the sparse directions point toward:

- a regional-bank group: USB, TFC, KEY, RF, FITB, CFG, and HBAN, with BAC retaining only a near-zero weight;
- a large money-center group: JPM, BAC, WFC, and C;
- a direction dominated by MS, with a nearly zero additional weight on C.

These sparse results are useful as a map of the loading space. They are not identification by themselves: penalization changes the estimated directions, sparse components need not remain orthogonal, and a zero weight does not prove zero economic exposure.

## Three-factor localization after SPY/XLF residualization

### Method

The L1 stage starts from the residual correlation-PCA loading space with `K=3` and searches for an oblique rotation that minimizes the total L1 loading norm. The primary fit uses 1,000 random starting directions. A local loading is counted as small when:

```text
|loading| < h_n,    h_n = 1 / log(12) = 0.4024
```

With 12 stocks, the 5% critical count is `gamma_n = 7`. The implementation therefore requires more than seven small loadings for a direction to pass the formal local-factor diagnostic.

The rotated representation preserves the selected three-dimensional residual PCA subspace. Its reconstruction share is **56.1085%**, and the maximum numerical reconstruction discrepancy is below **3 × 10^-14**.

### Full-sample loading map

The full-sample residual rotation produces small-loading counts of **[9, 8, 4]** for LF1, LF2, and LF3. The active supports and their working interpretations are:

| Direction | Full-sample active support (`|loading| >= 0.4024`) | Working interpretation |
| --- | --- | --- |
| **LF1** | WFC, C, MS | C–MS capital-markets / diversified-bank direction, with WFC as a smaller but active loading |
| **LF2** | JPM, BAC, WFC, MS | Large money-center direction; the fourth active name is close to a C/MS boundary under resampling |
| **LF3** | JPM, USB, TFC, KEY, RF, FITB, CFG, HBAN | Broad regional-bank cluster with a stable JPM anchor |

The signs are conventional and can be reversed without changing the factor. The supports matter more than the signs for this diagnostic.

### Session-bootstrap stability

To avoid treating thousands of minutes from one trading day as independent evidence, the bootstrap resamples the **924 trading sessions** with replacement, rebuilds the correlation matrix from session-level sufficient statistics, re-estimates PCA and the L1 rotation, and aligns each result to the full-sample directions by absolute loading cosine similarity. There are 100 whole-session replications.

| Direction | Median cosine | 5th-percentile cosine | Support result |
| --- | ---: | ---: | --- |
| **LF1** | 0.9991 | 0.9964 | C and MS active in 100% of replications; WFC active in 92% |
| **LF2** | 0.9977 | 0.6891 | JPM, BAC, and WFC active in 100%; C active in 48%; MS active in 52% |
| **LF3** | 0.9999 | 0.9997 | JPM and all seven regional-bank names active in 100% |

The bootstrap supports a nuanced reading:

- **LF1 is highly stable**, although WFC is not as definitive as C and MS.
- **LF2 has a real tail-stability warning.** Its direction is usually well aligned, but the fourth active loading alternates between C and MS. JPM, BAC, and WFC are the stable core.
- **LF3 is exceptionally stable as a direction and support pattern**, but its eight active loadings mean that it does not pass the formal local-factor threshold in the three-factor specification. It is better described as a stable regional-bank cluster direction than as a sharply local factor.

### Factor-count sensitivity

The eigenvalue-ratio diagnostic selects one dominant factor in both the raw and residual panels. The L1 local-factor diagnostic is negative at `K=2` and positive at `K=3`, `K=4`, and `K=5`. The project therefore retains `K=3` as a continuity and interpretation choice, not as a claim that the true factor count has been conclusively estimated.

This distinction is important: the data support a clear one-factor market/financial common direction, plus additional residual structure. The three-factor residual representation is useful for organizing that structure, but its economic labels remain conditional on the retained dimension and the chosen rotation criterion.

## Dynamic rolling local-factor diagnostics

The new stage-14 analysis re-estimates benchmark residualization, correlation PCA, and the `K=3` L1 rotation inside every trailing window. It uses exact trading-session windows of 60 and 120 sessions, advances five sessions at a time, and always includes the final trailing window. The 60-session specification is primary; the 120-session specification checks persistence at a slower horizon.

The analysis uses two distinct factor-label views. The past-only view aligns each window to an average of previously aligned directions, updating that anchor only after the current metrics are stored. This is the no-look-ahead stability view. The ex-post view aligns every window to the full-sample solution and is used only for descriptive heatmaps and factor localization. The rolling instability rule is deliberately descriptive: it flags cosine similarity below 0.80, support Jaccard below 0.50, a change in the local-factor decision, or a rotation condition number above 10. A regime candidate requires two consecutive five-session flags for the same factor.

The current 924-session sample produced **174 60-session windows** and **162 120-session windows**. The primary fit uses 200 random starts per window. **305 windows** received a 500-start sensitivity refit; 228 were stable under the comparison and 77 showed a primary-versus-sensitivity disagreement. Mixed windows are retained and explicitly marked because the pre-crisis, March–May 2023, post-crisis, and recent-2026 labels are majority-session descriptions rather than clean event partitions.

| Window | Factor | Mean cosine to past-only anchor | 5th-percentile cosine | Mean support Jaccard | Local-factor rate | Persistent candidates |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 60 | LF1 | 0.960 | 0.870 | 0.656 | 99.4% | 17 |
| 60 | LF2 | 0.881 | 0.707 | 0.767 | 92.5% | 149 |
| 60 | LF3 | 0.995 | 0.986 | 0.932 | 0.0% | 0 |
| 120 | LF1 | 0.973 | 0.897 | 0.738 | 100.0% | 10 |
| 120 | LF2 | 0.871 | 0.664 | 0.775 | 98.1% | 142 |
| 120 | LF3 | 0.997 | 0.988 | 0.932 | 0.0% | 0 |

The factor interpretation is consistent but not static. In the ex-post full-sample-aligned 60-session loading map, LF1 is concentrated most strongly on MS and C; LF2 on WFC, JPM, BAC, C, and MS; and LF3 on the regional-bank cluster led by CFG, FITB, HBAN, KEY, and RF. The mean first-three explained share is approximately **57.0%** for both window lengths. Mean PC1 explained variance is approximately **37.0%** for 60 sessions and **37.2%** for 120 sessions. Rolling SPY/XLF residualization removes approximately **44.8%** and **45.8%** of stock variance on average, respectively.

LF3 is the stable broad cluster direction: its rolling cosine and support Jaccard remain high, but it never passes the formal local-factor rate in the rolling output. LF1 is also stable overall. LF2 is the main dynamic warning: its lower cosine, persistent-candidate count, and 77-window sensitivity-disagreement set indicate that its active support should not be treated as a fixed portfolio membership. The factor scores are oblique rather than independent; the mean off-diagonal score correlations across 60-session windows are approximately 0.41 for LF1–LF2, 0.25 for LF1–LF3, and 0.42 for LF2–LF3.

The generated stage-14 evidence consists of nine CSV tables and seven PNG diagnostics under `alpaca_us_banks_1m/reports/dynamic_local_factor_regimes/`. The directory is ignored by Git because the tables and figures are reproducible from the local research dataset. The real-data run is reproducible with:

```text
.\\.venv\\Scripts\\python.exe src\\14_dynamic_local_factor_regimes.py
```

Stage 14 by itself does not establish structural breaks, causality, predictability, or trading profitability. Windows overlap, the number of rolling observations is not an effective number of independent tests, the regime groups contain mixed windows, and factor labels remain conditional on `K=3` and on the L1 criterion. Stage 15 now tests historical walk-forward predictability; stronger joint uncertainty estimates and a genuinely untouched future holdout remain necessary.

## Kalman dynamic-factor comparison

The session-level Kalman extension provides a useful robustness and model-selection result, but it does not improve on the rolling static factor estimator. Across 45 holdout windows, the rolling static PCA/L1 model has median holdout residual variance of **0.378961**, compared with **0.390408** for the Kalman-filtered specification. The Kalman result is therefore approximately **3.0% worse** on this metric. Its median loading cosine is **0.792**, median support Jaccard is **0.571**, and median loading turnover is **0.302**.

The dynamic estimates still recover recurring regional-bank, large-bank, and MS-related loading concentrations. This agreement supports the economic interpretation of the sparse loading space, but it does not justify replacing the simpler rolling PCA/L1 model with the Kalman specification. The correct conclusion is a negative model-complexity result: dynamic state-space estimation is feasible and economically legible, but the current implementation adds no demonstrated holdout advantage.

## Factor-adjusted residual-volatility network

### Residual construction and invariance

Stage 15 asks whether directed volatility predictability remains after removing SPY/XLF and the complete rolling `K=3` banking subspace. Every benchmark fit, normalization, PCA estimate, and rotation is obtained from preceding sessions only. The primary factor window contains 120 sessions and is refreshed every five sessions; a 60-session factor window is retained as a robustness check.

If `Lambda` is the oblique L1-rotated loading matrix and `z_t` is the standardized benchmark residual, the factor-adjusted residual is

$$
u_t=D\left[I-\Lambda(\Lambda'\Lambda)^{-1}\Lambda'\right]z_t.
$$

The L1 rotation supplies interpretable coordinates but does not change the residual projector when all three retained directions are removed:

$$
\Lambda(\Lambda'\Lambda)^{-1}\Lambda'=V_3V_3'.
$$

This equality holds in the production run. Across 334 factor fits, the maximum projector discrepancy is approximately **2.83 × 10^-15**, the maximum training residual-orthogonality error is approximately **8.93 × 10^-14**, and the maximum score-block error is approximately **7.32 × 10^-14**. The maximum rotation condition number is **2.70**. These diagnostics rule out numerical instability as an explanation for the forecasting result.

SPY/XLF remove approximately **45.3%** of stock variance on average across the combined rolling diagnostics. The retained three-component space then removes approximately **57.0%** of standardized benchmark-residual variance. These are sequential variance-removal summaries and should not be added mechanically as percentages of the same denominator.

The one-minute residuals are summed into valid non-overlapping five-minute returns within each trading session. Daily idiosyncratic realized variance is the sum of their squares. The variance floor is `1 × 10^-16`; no raw, benchmark-residual, or factor-adjusted daily observation is reported as materially affected by it.

### Network-HAR forecast result

For each bank, the own daily, weekly, and monthly HAR terms remain unpenalized. Daily, weekly, and monthly volatility predictors from the other eleven banks are partialled with respect to the own-HAR regressors and estimated by chronological lasso. The primary penalty uses the one-standard-error rule. The outer evaluation begins on 31 July 2024 for the 120-session specifications and contains 529 target sessions, or 6,348 stock-day forecasts.

| Specification | Own-HAR QLIKE | Network-HAR QLIKE | Relative improvement | Network log-MSE | Directional accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 120-session factor, expanding HAR | 0.198327 | **0.191449** | **3.47%** | 0.276954 | 66.15% |
| 120-session factor, 252-session rolling HAR | 0.198516 | **0.189443** | **4.57%** | 0.277413 | 66.30% |
| 60-session factor, expanding HAR | 0.193153 | **0.185120** | **4.16%** | 0.272016 | 66.19% |

The minimum-validation-loss penalty produces nearly the same pooled QLIKE as the more conservative one-standard-error penalty. This is useful evidence that the aggregate result is not driven by one arbitrary point on the penalty path.

For the primary 120-session expanding specification, the date-level pooled network-minus-own-HAR QLIKE differential is **-0.006878**, with HAC standard error **0.001845**, z-statistic **-3.73**, and nominal two-sided p-value **0.000193**. Negative values favor the network model. The cumulative loss differential trends downward through most of the evaluation period rather than being generated by one isolated observation.

The quarterly result is positive in eight of the nine reported quarters. The network is worse by approximately **1.69%** in the initial partial quarter of 2024Q3, improves by approximately **3.84–5.65%** during most subsequent quarters, and is nearly tied in 2026Q1. The aggregate gain is therefore persistent but not uniform through time.

### Cross-sectional heterogeneity

The pooled improvement does not apply equally to every stock. Under the primary specification, the approximate relative QLIKE changes are:

| Stock | Relative QLIKE improvement | Primary interpretation |
| --- | ---: | --- |
| TFC | **10.60%** | Largest and strongly supported gain |
| RF | **9.14%** | Large regional-bank gain |
| CFG | **6.22%** | Material regional-bank gain |
| KEY | **5.98%** | Positive, borderline after multiple-testing adjustment |
| BAC | **4.96%** | Significant large-bank gain |
| FITB | **4.83%** | Positive, weaker adjusted evidence |
| USB | **4.55%** | Significant regional-bank gain |
| HBAN | **3.89%** | Significant but smaller gain |
| JPM | **3.11%** | Positive, not significant after adjustment |
| WFC | -0.76% | Essentially no improvement |
| MS | -1.16% | Essentially no improvement |
| C | **-3.33%** | Significantly worse in the primary expanding specification |

After Benjamini–Hochberg adjustment across the twelve stock equations, the favorable primary results remain strongest for **BAC, CFG, HBAN, RF, TFC, and USB**. Citi's deterioration is significant in the primary expanding model but does not persist in the 252-session rolling-HAR or 60-session-factor specifications, so it should be treated as specification-sensitive rather than as a general failure for Citi.

### What remains in the network

The primary expanding network selects an average of **32.1%** of the 132 possible directed stock pairs and classifies 30 edges, or **22.7%**, as persistent under the 0.70 outer-refit threshold. Fourteen of those persistent edges connect regional banks to other regional banks. The remaining set contains economically interpretable large-bank links and cross-group connections.

The strongest recurring directions include `C -> BAC`, `C -> JPM`, `C -> WFC`, `BAC -> JPM`, `CFG -> KEY`, `KEY -> RF`, `KEY -> HBAN`, `USB -> TFC`, and `USB -> RF`. The main coefficients for the most robust daily edges are generally positive after controlling for the target bank's own daily, weekly, and monthly volatility. They therefore represent incremental one-session-ahead volatility information, conditional on the chosen HAR information set.

The descriptive network comparison does not show a mechanical monotonic collapse. With 120-session factor estimation, mean edge density falls from **39.8%** for raw realized volatility to **34.2%** after SPY/XLF removal and **33.0%** after full factor adjustment. The number of descriptively stable edges falls from 38 to 31 after benchmark removal and remains 31 after factor removal, although the composition changes. Under the 60-session factor specification, full factor adjustment produces a clearer reduction to 27 stable edges from 46 in the raw and benchmark-residual controls.

This is an important substantive result. The factors explain a material part of observed comovement, but they do not exhaust conditional volatility predictability. At the same time, edge count alone cannot measure how much economic dependence remains because penalized selection can introduce or remove links when conditioning variables change.

Outer-refit persistence is optimistic evidence because adjacent expanding samples differ by only one session. The moving-block bootstrap is therefore the more demanding diagnostic. It uses both five-session and twenty-session blocks at quarterly checkpoints, but only 20 replications per checkpoint. It reinforces a smaller core dominated by `C -> BAC`, `C -> JPM`, `CFG -> KEY`, `KEY -> RF`, and `C -> WFC`, while support for many secondary links is more variable. The bootstrap probabilities have a resolution of five percentage points and remain exploratory rather than definitive.

### Scope of the predictive claim

The correct claim is that cross-bank volatility information improves prediction of factor-adjusted realized variance within a leakage-controlled historical walk-forward design. The evidence is strongest in the regional-bank equations and survives the principal factor-window and HAR-window robustness checks.

The result is not evidence of structural causality, contagion, or tradable alpha. The 2023–2026 sample had already informed factor discovery, lasso selection is repeated within that history, and the March–May 2023 banking-stress interval is absent from the outer forecast because of factor and HAR burn-in. A frozen specification evaluated on subsequently acquired sessions is required before describing the result as genuinely untouched out-of-sample evidence.

## Current interpretation

The strongest current story is:

1. **Global common movement:** the raw financial-stock panel is dominated by one broad common direction.
2. **Benchmark-associated movement:** SPY and XLF account for a large share of that raw co-movement.
3. **Internal residual structure:** after benchmark removal, the remaining movement separates into a C/MS-oriented direction, a large-money-center direction, and a stable regional-bank cluster direction.
4. **Identification caveat:** only the first two residual directions pass the formal local-factor threshold at `K=3`; the regional-bank direction is stable but broader.
5. **Residual volatility network:** after removing the complete `K=3` space, a sparse directed network remains, with especially persistent links within the regional group and from Citi toward several large banks.
6. **Forecasting content:** the network HAR improves pooled pseudo-out-of-sample QLIKE by approximately 3.5–4.6% across the three principal specifications, with the largest stock-level gains concentrated in regional banks.
7. **Complexity discipline:** the Kalman extension does not improve holdout residual variance, whereas the simpler rolling factor model followed by a sparse HAR network produces measurable predictive gains.

This is evidence that the residual panel contains both organized low-rank structure and incremental cross-sectional volatility predictability. It is not evidence of causal transmission, independent structural shocks, a profitable trading strategy, or performance on a genuinely untouched sample.

## What remains to be tested

The next research priority is not another factor-extraction variant. The economically relevant next test is to freeze the present 120-session `K=3` factor specification, the one-standard-error network-HAR rule, and the current preprocessing choices, then evaluate them on newly acquired sessions that played no role in factor discovery, hyperparameter design, or interpretation.

Before making a stronger forecasting claim, the edge-stability analysis should be expanded from 20 to at least 200 moving-block replications at the principal checkpoints. Confidence intervals for pooled QLIKE improvement should also be obtained by resampling complete sessions while preserving the twelve-stock cross-section. The current HAC and Benjamini–Hochberg results are useful, but they do not replace uncertainty that propagates factor estimation and network selection jointly.

Further useful robustness work includes a longer five-minute sampling comparison, alternative realized-volatility estimators, explicit assessment of large-jump days, and a stability comparison of the surviving network across calm and stress regimes that occur inside the valid forecast sample. Covariance shrinkage and random-matrix diagnostics remain valuable as methodological appendices, but they are no longer the main path to the project's central empirical contribution.

Transaction costs are not yet applicable because Stage 15 forecasts volatility rather than returns and does not define a trading rule. A later portfolio experiment must specify how forecasts determine positions, exposures, turnover, execution delay, and costs before any economic-value or alpha claim is evaluated.

## Reproducibility and evidence files

The main pipeline stages are implemented in:

- [`07_baseline_pca.py`](src/07_baseline_pca.py)
- [`08_intraday_normalization.py`](src/08_intraday_normalization.py)
- [`09_benchmark_residualization.py`](src/09_benchmark_residualization.py)
- [`10_variance_decomposition.py`](src/10_variance_decomposition.py)
- [`11_rolling_pca.py`](src/11_rolling_pca.py)
- [`12_internal_factor_isolation.py`](src/12_internal_factor_isolation.py)
- [`13_l1_local_factor_identification.py`](src/13_l1_local_factor_identification.py)
- [`13_bis_kalman_dynamic_factors.py`](src/13_bis_kalman_dynamic_factors.py)
- [`14_dynamic_local_factor_regimes.py`](src/14_dynamic_local_factor_regimes.py)
- [`15_factor_adjusted_residual_network.py`](src/15_factor_adjusted_residual_network.py)

Generated CSV and figure outputs are stored under `alpaca_us_banks_1m/reports/` during a local run and are intentionally excluded from Git. The numerical values in this snapshot were read from the generated baseline, residual, variance-decomposition, internal-factor, local-factor-identification, dynamic-regime, Kalman-comparison, and factor-adjusted-network tables. Re-run the stages above to regenerate the artifacts from the local dataset.

For the data-treatment decisions, mathematical definitions, and research principles, see [`README.md`](README.md).
