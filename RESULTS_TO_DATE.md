# Results to Date: Intraday Statistical Factor Extraction in U.S. Financials

> **Snapshot date:** 11 September 2026
> **Market-data endpoint:** 9 September 2026
> **Research status:** descriptive factor extraction; predictive and out-of-sample testing has not started

## Executive summary

The project now runs from quality-controlled one-minute SIP data through baseline PCA, intraday-normalization checks, SPY/XLF residualization, variance reconciliation, rolling PCA, internal factor isolation, session-level L1 local-factor identification, and dynamic rolling local-factor diagnostics.

The main conclusions so far are:

1. A strong common component is present in the CORE universe. Correlation PCA assigns **63.77%** of standardized variance to PC1 and **75.85%** to the first three components.
2. The result is not explained by the market clock alone. After applying the intraday volatility profile, PC1 explains **64.47%** and the first three components explain **75.81%**, almost unchanged from the raw correlation-PCA baseline.
3. SPY/XLF exposure is economically important but does not exhaust the cross-section. The combined benchmark fit accounts for **43.59%** of raw CORE variance; the residual panel still has a structured first three-component space.
4. The residual loading space can be localized into three useful working directions. Two directions pass the formal local-factor diagnostic, while the third is a highly stable but broader regional-bank direction.
5. Rolling L1 diagnostics preserve the same broad localization but reveal that LF2 is materially less stable than LF1 and LF3. This warning survives both 60-session and 120-session windows.
6. The most defensible interpretation is therefore **a three-factor working representation with two sharply local directions and one broad, stable cluster direction**, not three independently identified causal factors.

The analysis remains descriptive. The factor names below summarize loading supports; they do not claim causal economic shocks, tradable signals, or a uniquely determined true factor count.

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

These results do not establish structural breaks, causality, predictability, or trading profitability. Windows overlap, the number of rolling observations is not an effective number of independent tests, the regime groups contain mixed windows, and factor labels remain conditional on `K=3` and on the L1 criterion. Session/block uncertainty bands and a frozen walk-forward evaluation remain necessary.

## Current interpretation

The strongest current story is:

1. **Global common movement:** the raw financial-stock panel is dominated by one broad common direction.
2. **Benchmark-associated movement:** SPY and XLF account for a large share of that raw co-movement.
3. **Internal residual structure:** after benchmark removal, the remaining movement separates into a C/MS-oriented direction, a large-money-center direction, and a stable regional-bank cluster direction.
4. **Identification caveat:** only the first two residual directions pass the formal local-factor threshold at `K=3`; the regional-bank direction is stable but broader.

This is strong evidence that the residual panel contains organized cross-sectional structure. It is not evidence yet of causality, independent structural shocks, a profitable strategy, or a stable out-of-sample forecasting signal.

## What remains to be tested

The next research steps are:

- compare the dynamic L1 factors with the session-level Kalman extension and with the Varimax/Elastic-Net loading maps;
- add block/session uncertainty bands and formal multiple-testing controls for rolling instability candidates;
- compare L1 supports against Varimax and Elastic-Net supports using explicit overlap and stability metrics;
- add covariance shrinkage and random-matrix diagnostics;
- test residual serial dependence, lead–lag structure, and factor-adjusted networks at a lower intraday frequency;
- lock a walk-forward evaluation before making any predictive or trading claim;
- include realistic transaction costs, turnover, missingness, multiple-testing controls, and a frozen out-of-sample period.

Until those checks are complete, the three localized directions should be used as an interpretable statistical decomposition and as an input to the next stage of research, not as final economic factors.

## Reproducibility and evidence files

The main pipeline stages are implemented in:

- [`07_baseline_pca.py`](src/07_baseline_pca.py)
- [`08_intraday_normalization.py`](src/08_intraday_normalization.py)
- [`09_benchmark_residualization.py`](src/09_benchmark_residualization.py)
- [`10_variance_decomposition.py`](src/10_variance_decomposition.py)
- [`11_rolling_pca.py`](src/11_rolling_pca.py)
- [`12_internal_factor_isolation.py`](src/12_internal_factor_isolation.py)
- [`13_l1_local_factor_identification.py`](src/13_l1_local_factor_identification.py)
- [`14_dynamic_local_factor_regimes.py`](src/14_dynamic_local_factor_regimes.py)

Generated CSV and figure outputs are stored under `alpaca_us_banks_1m/reports/` during a local run and are intentionally excluded from Git. The numerical values in this snapshot were read from the generated baseline, residual, variance-decomposition, internal-factor, local-factor-identification, and dynamic-regime tables. Re-run the stages above to regenerate the artifacts from the local dataset.

For the data-treatment decisions, mathematical definitions, and research principles, see [`README.md`](README.md).
