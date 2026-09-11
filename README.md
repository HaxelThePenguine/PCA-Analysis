# Intraday Statistical Factor Extraction in US Financials

## Why this project exists

Most PCA examples begin with a perfectly clean matrix, draw a scree plot, and stop there. This project is interested in what comes before and after that picture.

We start from one-minute SIP market data for a group of large U.S. financial stocks. The first job is to build a synchronized return panel without hiding the difficult parts of the data: missing bars, feed gaps, early closes, trading halts, and extreme market moves. Once that foundation is reliable, we use PCA and related methods to understand how much of the cross-sectional movement is genuinely common across the stocks.

The longer-term question is whether anything left after removing the broad common factors has a stable structure. If residual movements show persistence, mean reversion, or lead-lag relationships, they may deserve further research. That signal would still need to survive walk-forward testing, realistic transaction costs, and an honest out-of-sample evaluation before it could be considered useful.

For a dated snapshot of the numerical results, factor localization, and current interpretation, see [`RESULTS_TO_DATE.md`](RESULTS_TO_DATE.md).

## Questions we want to answer

The project is organized around a few practical research questions:

1. When these financial stocks move together, how much of that movement can be explained by one dominant common factor?
2. Does the factor structure become more concentrated during periods of market stress, such as the regional-bank crisis in March 2023?
3. Are the results materially different when we use covariance PCA, which preserves volatility differences, versus correlation PCA, which puts the securities on a comparable scale?
4. After removing broad market and financial-sector exposure through SPY and XLF, is there still a meaningful internal structure among the financial stocks?
5. Can sparsity identify economically interpretable local factors inside the residual PCA space, and are their loading supports stable when whole trading sessions are resampled?
6. Do the residual components contain any repeatable short-horizon behavior, or do they look like noise once data quality and multiple testing are taken seriously?

The project is descriptive first and predictive later. No residual pattern will be treated as an alpha signal until it has been tested out of sample and after trading frictions have been included.

## Data scope

Historical data comes from Alpaca Market Data using the SIP feed.

| Item | Choice |
| --- | --- |
| Frequency | 1-minute bars |
| Fields | timestamp, open, high, low, close, volume, trade_count, vwap |
| Period | 2023-01-01 through 2026-09-09 |
| Sessions | 924 regular U.S. trading sessions |
| Normal session | 09:30 to 16:00 America/New_York |
| Expected full-session bars | 390 per symbol, adjusted for early closes |

The market calendar supplied by Alpaca is used to handle early-close sessions rather than assuming that every day has exactly 390 observations.

### Instruments

The downloaded universe contains 18 financial stocks and two broad benchmarks:

```text
Financial stocks:
BAC, COF, JPM, HBAN, WFC, USB, RF, SCHW, C, AXP,
TFC, FITB, KEY, GS, MS, FHN, CFG, SYF

Benchmarks:
XLF, SPY
```

The analysis uses two explicit universes:

| Universe | Constituents | Purpose |
| --- | ---: | --- |
| CORE | 12 | Primary PCA specification with high retention and a coherent bank/financial cross-section |
| FULL | 18 | Robustness check that includes all downloaded financial stocks |

CORE contains `JPM`, `BAC`, `WFC`, `C`, `USB`, `TFC`, `KEY`, `RF`, `FITB`, `CFG`, `HBAN`, and `MS`.

FULL contains all 18 financial stocks listed above. SPY and XLF remain available as benchmarks and residualization factors, but are not part of either stock universe.

## Data-quality decisions

Data treatment is part of the research, not a detail to hide. The following decisions are recorded explicitly so that later PCA results can be reproduced and challenged.

### Common feed gap

Between 09:52 and 09:55 ET on 2023-06-05, 18 of the 20 instruments were missing at the same time. This is treated as a systemic market-data gap. Those timestamps are removed globally rather than forward-filled across the panel.

### Missing bar does not necessarily mean no trading

A security can have no OHLC bar for a minute even though transactions were present in the raw SIP data. This was verified for GS on 2025-09-12 at 09:50 ET, where many odd-lot trades were present.

Previous-tick sampling is used only to construct a synchronized price series. Returns directly affected by an imputed price are marked as contaminated and excluded from the complete PCA panel.

If a price is missing at time `t`, both the synthetic return at `t` and the return at `t+1` are excluded. The second return may contain the whole move accumulated since the last observed price.

### Known bad session

The full session on 2023-01-24 is removed from return analysis because of the NYSE opening-auction malfunction and subsequent trade cancellations. The large early-session moves from that day are not treated as ordinary observations.

### Stress periods and extreme returns

The March 2023 regional-bank crisis remains in the research data because it is economically meaningful for the questions we are asking. Large returns are not clipped or winsorized automatically. They are investigated first to determine whether they reflect real market events, data problems, or a security-specific halt.

## Data layout

Raw observations are stored as compressed Parquet files and are treated as immutable.

```text
alpaca_us_banks_1m/
├── raw/
│   └── by_symbol/              raw one-minute files, one per instrument
├── chunks/                     monthly download chunks and completion markers
├── metadata/
│   └── market_calendar.csv
├── reports/
│   ├── data_quality_summary.csv
│   ├── daily_data_quality.csv
│   ├── missing_matrix.parquet
│   ├── common_missing_gaps.csv
│   ├── rolling_pca/
│   ├── internal_factor_isolation/
│   ├── local_factor_identification/
│   └── dynamic_local_factor_regimes/
├── intermediate/
│   ├── close_matrix.parquet
│   ├── return_matrix.parquet
│   └── missing_mask.parquet
└── processed/
    ├── return_matrix_clean.parquet
    ├── return_matrix_complete.parquet
    ├── contaminated_mask.parquet
    ├── return_core.parquet
    └── return_full.parquet
```

The raw and generated data folders are excluded from Git through `.gitignore`. This keeps the code repository small and avoids treating a large local research dataset as source code.

## Preprocessing pipeline

The pipeline follows a simple sequence:

```text
Raw SIP minute bars
        ↓
Market-calendar alignment
        ↓
Systemic-gap removal
        ↓
Missing-bar inspection
        ↓
Cross-sectional price synchronization
        ↓
Within-session log returns
        ↓
Contamination masking
        ↓
Complete CORE and FULL panels
        ↓
        Baseline PCA
        ↓
        SPY/XLF residualization
        ↓
        Variance decomposition
        ↓
        Internal factor isolation
        ↓
        L1 local-factor identification
        ↓
        Rolling PCA
        ↓
        Dynamic local-factor regimes
```

The scripts deliberately keep the stages separate. Each step reads the previous stage's output and writes a named artifact that can be inspected or reused later.

### One-minute returns

For security `i` and minute `t`, the return is:

$$
r_{i,t} = \log(P_{i,t}) - \log(P_{i,t-1})
$$

Returns are calculated within each trading session. The first minute of every session is set to missing so that the overnight close-to-open move is not mixed into a one-minute intraday return.

## PCA methodology

Let `X` be the centered return matrix. For correlation PCA, `Z` is the same matrix after dividing each column by its sample standard deviation. PCA starts from the appropriate cross-sectional matrix:

$$
\Sigma = \frac{1}{n-1}X^\top X
$$

or, for correlation PCA,

$$
R = \frac{1}{n-1}Z^\top Z
$$

The eigendecomposition is:

$$
R v_k = \lambda_k v_k
$$

The vector `v_k` is the normalized eigenvector used as a set of component weights. The factor scores are calculated afterwards as:

$$
f_k = Z v_k
$$

There are two related quantities that are often both called loadings. The files produced by the baseline scripts preserve the eigenvector weights `v_k` for backward compatibility. The conventional technical loadings are:

$$
L_{i,k} = \sqrt{\lambda_k}\,v_{i,k}
$$

For correlation PCA, `L_{i,k}` is the correlation between stock `i` and component `k`. The new reusable PCA module exposes both representations so that interpretation and projection do not get mixed.

### Covariance PCA

Covariance PCA works with centered returns and preserves the original volatility scale of each security. Higher-volatility stocks therefore have more influence on the estimated covariance structure.

### Correlation PCA

Correlation PCA first standardizes each security:

$$
z_{i,t} = \frac{r_{i,t} - \mu_i}{\sigma_i}
$$

This gives each stock comparable marginal volatility and makes the result more focused on co-movement than on differences in individual volatility. It is the main baseline specification because volatility levels differ materially across the universe.

The first baseline comparison saves eigenvalues, explained-variance ratios, cumulative explained variance, eigenvector weights for PC1 through PC3, technical loadings, the correlation matrix, and the associated plots.

## Market and sector residualization

After the raw-return baseline, broad market and financial-sector exposure will be removed using SPY and XLF:

$$
r_{i,t} = \alpha_i + \beta_{i,M} r_{SPY,t} + \beta_{i,F} r_{XLF,t} + \epsilon_{i,t}
$$

The residuals are the part of each stock's return that is not explained by those two benchmark returns. PCA on the residual matrix will then be compared with PCA on the original returns.

This comparison is intended to answer a specific question: does the financial-stock universe contain internal structure that is hidden by broad market and sector exposure?

### Variance decomposition

The residual analysis also keeps an explicit ledger of where the original variance goes. For each stock, the fitted SPY/XLF part and the residual part are additive:

$$
Var(r_i) = Var(\hat{r}_i) + Var(\epsilon_i)
$$

The residual covariance is then decomposed into its principal components. If `v_{i,k}` is the loading of stock `i` on residual component `k`, that component contributes:

$$
c_{i,k} = \lambda_k v_{i,k}^2
$$

This lets us report, for every stock, the share explained by the combined SPY/XLF fit, residual PC1, residual PC2-PC3, and residual PC4-PC12. The benchmark share is kept combined because separating SPY and XLF into two squared terms would double-count their covariance.

## Internal factor isolation

Once standard PCA has shown where the common variation lies, the next question is whether the factors can be made easier to interpret without pretending that the statistical components are economic causes. Script `12_internal_factor_isolation.py` compares three views of the same correlation-scaled stock panel:

- standard PCA, which remains the variance-maximizing reference;
- Varimax, which rotates the first three-component subspace toward a simpler loading pattern;
- Elastic-Net Sparse PCA, which penalizes the component weights so that some become exactly zero while correlated stocks are less likely to be selected arbitrarily.

### Varimax rotation

Let `L` contain the conventional loadings of the first `K` PCA components. Varimax searches for an orthogonal rotation `Q`:

$$
L_{\mathrm{rot}} = LQ,
\qquad Q^\top Q = I
$$

The rotation maximizes a simple-structure criterion:

$$
\max_Q\;\sum_{k=1}^{K}
\left[
\frac{1}{p}\sum_{i=1}^{p}L_{\mathrm{rot},i,k}^{4}
-
\left(\frac{1}{p}\sum_{i=1}^{p}L_{\mathrm{rot},i,k}^{2}\right)^2
\right]
$$

Varimax does not remove a stock. It changes the coordinate system inside the selected factor subspace, so it is a useful interpretive lens when several components may be rotated without changing the represented subspace. The original criterion is due to [Kaiser (1958), *The Varimax Criterion for Analytic Rotation in Factor Analysis*](https://doi.org/10.1007/BF02289233).

### Elastic-Net Sparse PCA

Sparse PCA modifies the PCA reconstruction problem by adding an L1 penalty and an L2 penalty. In the implementation, `B` contains the sparse component weights and `A` keeps the auxiliary factors orthonormal:

$$
\min_{A,B}\;
\lVert Z-ZBA^\top\rVert_F^2
+\lambda_2\lVert B\rVert_F^2
+\sum_{j=1}^{K}\lambda_{1,j}\lVert b_j\rVert_1,
\qquad A^\top A=I
$$

The two penalties have different jobs:

- `L1` (`lambda_1`) creates exact zero weights and controls sparsity;
- `L2` (`lambda_2`) stabilizes correlated groups and reduces the tendency to keep one arbitrary representative.

The method follows the regression formulation introduced by [Zou, Hastie, and Tibshirani (2006), *Sparse Principal Component Analysis*](https://doi.org/10.1198/106186006X113430). The Elastic-Net grouping motivation comes from [Zou and Hastie (2005), *Regularization and Variable Selection Via the Elastic Net*](https://doi.org/10.1111/j.1467-9868.2005.00503.x). The earlier direct L1-constrained approach, SCoTLASS, is documented by [Jolliffe, Trendafilov, and Uddin (2003), *A Modified Principal Component Technique Based on the LASSO*](https://doi.org/10.1198/1061860032148).

The least-squares reconstruction used by the script is only an internal diagnostic of how much information the sparse score space retains. It is not an additional economic regression model. Sparse components can also lose the exact orthogonality and variance-ordering properties of ordinary PCA, so their reconstruction percentage is reported separately from ordinary PCA explained-variance shares. This distinction is also emphasized in the finance-oriented discussion by [Despois (2023), *Identifying and Interpreting the Factors in Factor Models via Sparsity: Different Approaches*](https://doi.org/10.1002/jae.2967).

The project therefore uses the following hierarchy:

```text
Correlation PCA       official variance reference
Varimax               interpretable rotation of the same subspace
Elastic-Net Sparse PCA sparse exploratory factor composition
Pure Lasso             sensitivity comparison, not the main result
```

The current penalty path records reconstruction and non-zero-weight counts for several `lambda_1` values, both with and without the `lambda_2` grouping term. A penalty will be considered useful only if the selected stocks are reasonably stable and the information loss is transparent; a visually sparse result alone is not sufficient evidence of a real factor.

## L1 rotation and local-factor identification

Script `13_l1_local_factor_identification.py` addresses a different problem from Sparse PCA. Sparse PCA changes the estimated directions through penalization; L1 rotation starts from an estimated PCA loading space and selects a sparse coordinate system *inside that same space*. It implements the multistart procedure and diagnostic of [Freyaldenhoven (2026), *Identification through sparsity in factor models: The L1-rotation criterion*](https://doi.org/10.3982/QE2369), following the accompanying [`l1rotation` reference implementation](https://kobleary.github.io/l1rotation/).

For `n` stocks and `K` retained components, let

$$
\Lambda_0 = \sqrt{n}\,[v_1,\ldots,v_K],
\qquad
\frac{1}{n}\Lambda_0^\top\Lambda_0=I_K,
$$

where the `v_k` are the correlation-PCA eigenvectors. For a unit direction `q`, the criterion is

$$
Q(q)=\lVert\Lambda_0q\rVert_1
=\sum_{i=1}^{n}|(\Lambda_0q)_i|,
\qquad \lVert q\rVert_2=1.
$$

The objective is non-convex on the unit sphere and can have several economically relevant local minima. The implementation therefore uses 1,000 random starting directions for the primary three-factor specification, optimizes in hyperspherical coordinates, consolidates sign-equivalent nearby solutions, and selects `K` linearly independent directions. If `R=[q_1,\ldots,q_K]` is the resulting nonsingular—generally oblique—rotation, then

$$
\widehat{\Lambda}^{*}=\Lambda_0R.
$$

This is selection by rotation, not shrinkage: small entries are not forced to zero. Factor scores are recovered by least squares,

$$
\widehat F
=Z\widehat{\Lambda}^{*}
\left(\widehat{\Lambda}^{*\top}\widehat{\Lambda}^{*}\right)^{-1},
$$

and, because `R` is nonsingular,

$$
\widehat F\widehat{\Lambda}^{*\top}
=ZV_KV_K^\top.
$$

The rotated representation therefore preserves the selected PCA subspace exactly. On the current sample, the numerical discrepancy is below `3e-14`; the first three residual components still reconstruct `56.1085%` of standardized residual variance.

### Local-factor diagnostic

The reference diagnostic defines a loading as small when

$$
|\widehat\lambda_{ik}^{*}|<h_n,
\qquad h_n=\frac{1}{\log n},
$$

and counts small entries in every factor,

$$
\mathcal L_k=\sum_{i=1}^{n}
\mathbf 1\{|\widehat\lambda_{ik}^{*}|<h_n\}.
$$

With `p_h=2\Phi(h_n)-1`, the implementation uses the package's 5% critical rule

$$
\gamma_n=\left\lfloor n\left[
0.03+p_h+z_{0.975}
\sqrt{\frac{p_h(1-p_h)}{n}}
\right]\right\rfloor,
\qquad
\max_k\mathcal L_k>\gamma_n.
$$

Here `n=12`, so `h_n=0.4024` and `gamma_n=7`. Conditional on the three-component working space, the residual rotation has small-loading counts `[9, 8, 4]`: LF1 and LF2 meet the local-factor threshold, while LF3 is a broader direction. The corresponding residual loading map is:

- **LF1:** `MS`, `C`, and `WFC` are active under the reference threshold;
- **LF2:** `JPM`, `BAC`, `WFC`, and `MS` are active;
- **LF3:** `JPM` plus the regional-bank block `USB`, `TFC`, `KEY`, `RF`, `FITB`, `CFG`, and `HBAN` are active.

These labels describe loading support, not causal shocks. They are also conditional on `K=3`. The eigenvalue-ratio diagnostic selects one dominant residual factor, and the L1 local-factor test is negative at `K=2` but positive for `K=3,4,5`. The three-factor specification is retained as an interpretive working space because it exposes internal cross-sectional structure; it is not presented as an undisputed estimate of the true factor count.

### Whole-session bootstrap

Minute observations within a day are not treated as independent bootstrap units. The script resamples the 924 trading sessions with replacement 100 times, rebuilds the correlation matrix from session-level sufficient statistics, re-estimates PCA and the L1 rotation, and aligns each bootstrap estimate to the full-sample factors by maximizing absolute loading cosine similarity. For reference loading `k` and bootstrap loading `j`, the matching score is

$$
s_{kj}^{(b)}=
\frac{|\widehat\lambda_k^\top\widehat\lambda_j^{(b)}|}
{\lVert\widehat\lambda_k\rVert_2
 \lVert\widehat\lambda_j^{(b)}\rVert_2}.
$$

Residual LF1 and LF3 have fifth-percentile cosine similarities of `0.996` and `1.000`. LF2 has a median of `0.998` but a fifth percentile of `0.690`, which reveals a real tail-stability warning: `JPM`, `BAC`, and `WFC` remain active in every replication, while the fourth active loading alternates mainly between `MS` (`52%`) and `C` (`48%`). This uncertainty is retained in the output instead of assigning an overconfident economic label.

The stage writes inspectable loading, rotation, factor-count, `K`-sensitivity, factor-score, bootstrap-direction, and bootstrap-support tables, together with three diagnostic figures. The generated research artifacts live under `reports/local_factor_identification/` and remain outside version control.

## Dynamic rolling local-factor regimes

Script [`14_dynamic_local_factor_regimes.py`](src/14_dynamic_local_factor_regimes.py) extends the full-sample L1 rotation to a past-only rolling diagnostic. It is deliberately a stability and regime-description stage, not a predictive model or a structural-break test.

### Window design and no-look-ahead rule

The production specification uses exactly two trailing windows:

| Window | Step | Purpose |
| ---: | ---: | --- |
| 60 trading sessions | 5 sessions | Primary quarterly-scale stability view |
| 120 trading sessions | 5 sessions | Longer-window persistence check |

The index is converted to unique trading sessions before windows are built. Every window contains exactly `w` sessions, including the final trailing window even when it is not on the regular five-session grid. For each window, the code re-estimates the SPY/XLF residualization, correlation PCA, and `K=3` L1 rotation using only observations whose session belongs to that window. No benchmark coefficient, standardization parameter, PCA estimate, or rotation direction is carried forward from the full sample.

For a window ending at session `t`, the residual panel is:

$$
e_{i,s}^{(w)}=r_{i,s}-\widehat\beta_{i,SPY}^{(w)}r_{SPY,s}
-\widehat\beta_{i,XLF}^{(w)}r_{XLF,s},
\qquad s\in W_t^{(w)}.
$$

Correlation PCA is then fitted to the standardized residual observations in `W_t^(w)`. If `V_K^(w)` is the first-three eigenvector matrix and `R_t^(w)` is the selected nonsingular L1 rotation,

$$
\Lambda_t^{*(w)}=\sqrt{n}V_K^{(w)}R_t^{(w)},
\qquad
\widehat F_t^{(w)}=Z_t^{(w)}\Lambda_t^{*(w)}
\left(\Lambda_t^{*(w)\top}\Lambda_t^{*(w)}\right)^{-1}.
$$

The rotation preserves the selected PCA subspace exactly up to numerical precision. The primary run uses 200 random starts per window. Windows that are late-quarter checkpoints or show a primary instability flag are re-fitted with 500 starts. The seed is deterministic: it is derived from the base seed, window size, window start/end dates, and the requested number of starts. This makes the primary and sensitivity results reproducible without reusing a single random stream across windows.

### Two alignment views

Factor labels are not compared by their raw column number because L1 directions can permute and change sign. The implementation maximizes absolute loading cosine similarity and applies the same permutation/sign transformation to structural loadings, stock–factor score loadings, and the oblique score-correlation matrix.

Two views are retained separately:

- **Past-only alignment:** the reference is the average of previously aligned rolling directions for the same window length. The current window is added to that anchor only after its metrics have been computed. This is the causal stability view.
- **Ex-post full-sample alignment:** every window is aligned to the full-sample L1 solution. It is used only for descriptive heatmaps and factor localization; it must not be used for forecasting or a real-time decision.

For aligned loading columns `k`, the main stability measures are:

$$
C_{k,t}=\frac{|\lambda_{k,t}^{\top}\lambda_{k,t-1}|}
{\|\lambda_{k,t}\|_2\|\lambda_{k,t-1}\|_2},
\qquad
J_{k,t}=\frac{|A_{k,t}\cap A_{k,ref}|}
{|A_{k,t}\cup A_{k,ref}|},
$$

where `A` is the active support under `|loading| >= h_n`. The output also records the anchor cosine, previous-window cosine, full-sample cosine, support cardinality, small-loading count, `gamma_n`, local-factor flag, L1 norm, solution frequency, candidate source, optimizer success rate, rotation condition number, factor-score correlations, explained variance, and the percentage of stock variance removed by SPY/XLF.

An instability flag is descriptive when a cosine falls below `0.80`, a support Jaccard falls below `0.50`, the local-factor decision changes, or the rotation condition number exceeds `10`. A regime candidate is recorded only after the same factor is flagged in two consecutive five-session checkpoints. Mixed windows are retained rather than silently assigned to one period.

### Regime definition and current real-data results

The explicit labels are `pre_banking_crisis`, `banking_crisis_mar_may_2023`, `post_banking_crisis`, and `recent_2026`. A window receives the majority label of its sessions and retains its full regime sequence plus a mixed-window flag. On the current 924-session sample, the run produced 174 60-session windows and 162 120-session windows. There are 305 sensitivity windows with a 500-start refit; 228 are stable under the stated comparison and 77 show a primary-versus-sensitivity disagreement. These are descriptive numerical diagnostics, not evidence of a formal break.

The past-only factor-level summary is:

| Window | Factor | Mean cosine | 5th-percentile cosine | Mean support Jaccard | Local-factor rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 60 | LF1 | 0.960 | 0.870 | 0.656 | 99.4% |
| 60 | LF2 | 0.881 | 0.707 | 0.767 | 92.5% |
| 60 | LF3 | 0.995 | 0.986 | 0.932 | 0.0% |
| 120 | LF1 | 0.973 | 0.897 | 0.738 | 100.0% |
| 120 | LF2 | 0.871 | 0.664 | 0.775 | 98.1% |
| 120 | LF3 | 0.997 | 0.988 | 0.932 | 0.0% |

LF3 is the most stable direction and remains broad rather than formally local. LF1 is generally stable but has occasional support changes. LF2 is the main instability candidate: it has the lowest rolling cosine, the largest number of persistent flags (149 for the 60-session specification and 142 for the 120-session specification), and the most visible time variation in its support. The longer window does not eliminate this behavior, so it should be treated as a real sensitivity warning rather than a single noisy checkpoint.

The ex-post full-sample-aligned loading map gives a useful localization summary. Across 60-session windows, LF1 is concentrated most strongly on MS and C, LF2 on WFC, JPM, BAC, C, and MS, and LF3 on the regional-bank cluster led by CFG, FITB, HBAN, KEY, and RF. The exact active support changes with the window and threshold; the labels are therefore conditional descriptions, not fixed portfolio memberships. The mean first-three explained share is approximately 57.0% in both window specifications, while rolling PC1 explains approximately 37.0% (60 sessions) and 37.2% (120 sessions). The rolling SPY/XLF residualization removes approximately 44.8% and 45.8% of stock variance on average, respectively, with residual benchmark correlations at numerical zero.

The L1 rotation is generally oblique: the mean off-diagonal score correlations across 60-session past-only windows are approximately 0.41 for LF1–LF2, 0.25 for LF1–LF3, and 0.42 for LF2–LF3, with larger local maxima. These directions should not be interpreted as mutually orthogonal shocks merely because they are labelled LF1–LF3.

### Outputs, execution, and limits

The ignored directory `alpaca_us_banks_1m/reports/dynamic_local_factor_regimes/` contains:

- `14_rolling_l1_loadings.csv`, `14_rolling_score_loadings.csv`, and `14_rolling_score_correlations.csv`;
- `14_rolling_factor_stability.csv`, `14_rolling_support_membership.csv`, and `14_rolling_window_diagnostics.csv`;
- `14_regime_summary.csv`, `14_unstable_windows.csv`, and `14_start_count_sensitivity.csv`;
- seven PNG diagnostics: loading and support heatmaps, cosine/Jaccard stability, small-loading counts, explained-variance/benchmark diagnostics, regime comparison, and 60-versus-120 window comparison.

Run the complete real-data stage from the project root with:

```text
.\\.venv\\Scripts\\python.exe src\\14_dynamic_local_factor_regimes.py
```

The analysis is intentionally conservative about interpretation. Overlapping windows are dependent; threshold crossings are not p-values; the March 2023 and recent-period groups contain many mixed windows; factor labels remain conditional on `K=3`; and SPY/XLF are benchmark controls rather than exogenous instruments. The stage does not test predictability, causality, trading profitability, or out-of-sample stability. The next useful checks are block/session bootstrap uncertainty for rolling paths, comparison with the 13-bis Kalman extension, and a frozen walk-forward evaluation.

## Stage 15: Factor-adjusted residual volatility network

Script [`15_factor_adjusted_residual_network.py`](src/15_factor_adjusted_residual_network.py) asks whether directed volatility predictability remains after removing the SPY/XLF benchmark component and the three-dimensional banking subspace identified by the rolling L1 analysis. The production specification freezes `K=3`, uses a 120-session trailing factor window with five-session updates, and applies a 60-session window plus a 252-session rolling HAR estimation window as robustness checks. The factor transformation is estimated from preceding sessions only and is then applied to the subsequent score block.

For each training window, the stock returns are first projected on SPY and XLF. The residuals are centered and standardized using training-window means and scales, correlation PCA is fitted, and the existing L1 rotation is applied to the first three PCA directions. If `Lambda` denotes the rotated loading matrix, the factor-adjusted return is constructed through

$$
u_t=D\left[I-\Lambda(\Lambda'\Lambda)^{-1}\Lambda'\right]z_t.
$$

The rotation is an interpretive change of coordinates, not an additional residualization step. Since the retained L1 directions span the same space as the first three PCA directions,

$$
\Lambda(\Lambda'\Lambda)^{-1}\Lambda'=V_3V_3',
$$

up to floating-point precision. In the production output the maximum projector discrepancy and the reported residual orthogonality errors are at numerical zero, while the largest rotation condition number is approximately `2.70`. The rolling first-three PCA space explains approximately `57.0%` of standardized benchmark-residual variance on average, after SPY/XLF remove approximately `45.8%` of stock variance in the 120-session specification and `44.8%` in the 60-session specification.

Residualized one-minute log returns are aggregated into valid, non-overlapping five-minute intervals within each trading session. Daily realized variance is the sum of squared five-minute residual returns, with a documented positive numerical floor. Raw and benchmark-residualized realized-variance panels are generated as descriptive controls; forecast losses are evaluated only on the factor-adjusted target.

The forecasting model is a one-session-ahead HAR equation for each target stock. Its own daily, weekly, and monthly terms remain unpenalized, while the corresponding terms of the other stocks are estimated through chronological partialling-out lasso cross-validation. The one-standard-error penalty is the primary selection, with the minimum-loss penalty retained as a sensitivity result. Forecasts are transformed back to variance units with a training-only smearing correction, and QLIKE is the primary loss.

The completed production comparison is:

| Specification | Own-HAR QLIKE | Network-HAR QLIKE | Relative QLIKE improvement |
| --- | ---: | ---: | ---: |
| 120-session expanding | 0.198327 | 0.191449 | 3.47% |
| 120-session factor, 252-session HAR rolling | 0.198516 | 0.189443 | 4.57% |
| 60-session expanding | 0.193153 | 0.185120 | 4.16% |

The network wins on pooled QLIKE in all three specifications. For the primary 120-session expanding design, the network-minus-own loss differential is `-0.006878`, with HAC standard error `0.001845` and nominal two-sided p-value `0.000193`. After Benjamini–Hochberg adjustment across the twelve stock equations, network improvement is strongest for BAC, CFG, HBAN, RF, TFC, and USB, whereas Citi (`C`) is significantly worse under the network specification. The effect is therefore heterogeneous: the pooled result is not evidence that every bank benefits from the same spillover structure.

The strongest primary stable edges include `C→BAC`, `C→JPM`, `BAC→JPM`, `CFG→KEY`, `KEY→RF`, `C→WFC`, and `USB→TFC`. The loading evolution is also economically legible: LF1 remains concentrated on MS and C, LF2 is concentrated on WFC, JPM, BAC, and C with more time variation, and LF3 remains the broad regional-bank direction led by CFG, FITB, HBAN, KEY, RF, TFC, and USB. After factor adjustment, descriptive within-group selection falls relative to raw volatility for the regional group, especially in the 60-session comparison, but a non-trivial residual network remains. This is consistent with the view that some apparent links were common-factor exposure while others survive in idiosyncratic volatility dynamics.

The March–May 2023 stress interval is correctly reported as unavailable for the outer forecast because the factor-estimation and HAR burn-in place the first valid forecasts later. Consequently, this stage cannot establish whether the residual regional-bank network becomes stronger during that episode. The rolling LF3 loadings remain geometrically stable, but that descriptive fact must not be substituted for a stress-period out-of-sample test.

All stage-15 artifacts are kept separately under `alpaca_us_banks_1m/reports/factor_adjusted_residual_network/`. The main inspectable tables are `15_daily_realized_variance.csv`, `15_factor_model_diagnostics.csv`, `15_factor_loadings.csv`, `15_walk_forward_forecasts.csv`, `15_forecast_summary.csv`, `15_stock_forecast_summary.csv`, `15_hac_tests.csv`, `15_har_coefficients.csv`, `15_edge_history.csv`, `15_edge_stability.csv`, `15_network_density.csv`, `15_network_centrality.csv`, `15_group_connectivity.csv`, `15_descriptive_edge_stability.csv`, `15_descriptive_group_connectivity.csv`, `15_bootstrap_edge_selection.csv`, and `15_factor_window_comparison.csv`. The figures are `15_cumulative_qlike_difference.png`, `15_stock_qlike_improvement.png`, `15_stable_directed_network_heatmap.png`, `15_edge_selection_persistence.png`, `15_group_connectivity.png`, `15_variance_removed.png`, and `15_factor_loading_evolution.png`; `15_summary.txt` contains the compact machine-generated narrative.

Run the stage from the project root with, for example,

```text
.\\.venv\\Scripts\\python.exe src\\15_factor_adjusted_residual_network.py --n-jobs 12
```

The result is a rigorous pseudo-out-of-sample volatility-forecasting and dependence-structure result, not a claim of structural causality, contagion, alpha, or deployable trading profitability. The 2023–2026 sample was already examined during factor discovery, the lasso is selected repeatedly within that historical sample, and the bootstrap uses a deliberately small exploratory replication count with block-length sensitivity. Edge-selection probability is a stability descriptor rather than a p-value. A genuinely untouched future holdout remains necessary before stronger claims can be made.

## Research roadmap

### 1. Baseline PCA

Run covariance and correlation PCA on `return_core.parquet` before applying any intraday normalization. This gives us a transparent reference point and lets us understand the raw geometry of the data before adding more choices.

### 2. Intraday seasonality

Volatility is usually higher near the open and close. We will estimate volatility by minute of day for each security and repeat PCA on returns scaled by that intraday profile. The goal is to see whether the apparent factor structure is being driven mainly by the market clock.

### 3. Market and sector residuals

We will remove SPY and XLF exposure and compare the residual eigenstructure with the raw-return result. Any interpretation of a residual component will be based on its actual loadings and constituent composition, not on intuition alone.

### 4. Rolling and stress-regime analysis

Rolling PCA will track eigenvalues, PC1 explained variance, loading stability, and eigenvector rotation through time. Stress windows such as March 2023 will be examined separately to test whether correlation rises and effective cross-sectional dimensionality falls during crisis conditions.

When comparing eigenvectors across windows, the sign is normalized through an absolute dot product because the sign of an eigenvector is arbitrary.

$$
S_k(t,t+\Delta) = |v_k(t)' v_k(t+\Delta)|
$$

#### Rolling-window design and literature rationale

The rolling analysis will use trading sessions as the time unit. This keeps a window tied to an economically meaningful number of market days and avoids treating weekends, holidays, and early closes as ordinary observations. The window length is a design choice rather than a universal constant: shorter windows react faster to regime changes but produce noisier covariance and PCA estimates, while longer windows are more stable but can average together distinct regimes.

The primary specification will use a **20-session window** with a **5-session step**. Twenty sessions are approximately one month of trading and contain about 7,800 scheduled one-minute slots per stock; in this cleaned sample, early closes and data-quality exclusions leave roughly 7,500-7,800 usable observations. This is a reasonable compromise for a 12-stock cross-section: it is short enough to detect changes around stress periods, while high-frequency sampling supplies many observations inside the calendar window. The count is not an effective independent sample size because intraday returns can be dependent and affected by microstructure noise. High-frequency factor research explicitly motivates short rolling intervals for this reason, including one-month windows for time-varying betas.

The analysis will repeat every comparison with a **60-session window**, also evaluated every five sessions. Sixty sessions are approximately one quarter and provide a slower, more stable view of the same structure. Sixty-day rolling correlations are also a familiar diagnostic in the co-movement literature. The 60-session result is therefore a robustness and persistence check, not a claim that three months is intrinsically optimal.

We will not use a one-week window as the main estimate. High-frequency PCA papers show that very short horizons can be informative when the cross-section and estimator are designed for that setting, but our first implementation uses a simple rolling sample covariance/correlation matrix on a small 12-stock universe. A one-week estimate would be more sensitive to individual events, missing observations, microstructure effects, and estimation noise. It remains a useful stress-sensitivity experiment later.

The five-session step is chosen for readable weekly monitoring. Because neighbouring rolling windows overlap, the resulting observations are not independent. We will not interpret a smooth rolling chart as a sequence of independent tests; confidence bands or block-bootstrap checks will be added before making formal claims about changes in loadings or eigenvalues.

The first rolling implementation will run both correlation PCA and covariance PCA on:

- raw returns;
- intraday-normalized returns;
- SPY/XLF residual returns.

`11_rolling_pca.py` implements this first descriptive pass. The intraday-normalized series uses the same fixed full-sample minute-of-day profile as the earlier robustness check; this is acceptable for descriptive comparison, but it must be replaced with a past-only profile before any forecasting or walk-forward test.

For each window we will record PC1 explained variance, cumulative variance explained by the first three components, effective dimension, loading similarity to the previous window, benchmark variance removed, and residual PC1 strength. This makes the interpretation explicit:

- a change in raw PCA that disappears after SPY/XLF residualization is benchmark-associated;
- a change that remains in residual PCA is evidence of internal financial-stock structure;
- a change in covariance PCA without a comparable change in correlation PCA is primarily a volatility-scale effect;
- a short-lived isolated spike is a regime or event candidate, not proof of a persistent factor.

Rolling PCA can locate timing, persistence, and structural changes. It cannot by itself establish that an external variable caused them. To investigate external explanations, the rolling results must later be aligned with independent series and event dates, such as market volatility, interest rates, credit conditions, or documented market events. SPY and XLF are treated as benchmark-associated controls, not as exogenous causal instruments, especially because XLF contains financial stocks.

This design is grounded in the following literature:

- [Aït-Sahalia and Xiu (2019), *Principal Component Analysis of High-Frequency Data*](https://doi.org/10.1080/01621459.2017.1401542) study time-varying high-frequency principal components and show that their explanatory power can change materially during stress periods.
- [Aït-Sahalia, Kalnina, and Xiu (2020), *High-Frequency Factor Models and Regressions*](https://doi.org/10.1016/j.jeconom.2020.01.007) discuss short rolling intervals such as one month, the time variation of betas, and the ability of high-frequency observations to reduce estimation noise.
- [Pelger (2019), *Large-dimensional factor modeling based on high-frequency observations*](https://doi.org/10.1016/j.jeconom.2018.09.004) shows why high-frequency factor methods can study short horizons, while also making clear that the appropriate horizon depends on the estimator and the cross-section.
- [Gospodinov (2017), *Asset Co-movements: Features and Challenges*](https://fraser.stlouisfed.org/title/working-papers-federal-reserve-bank-atlanta-8586/asset-co-movements-657145/content/fulltext/frbatl_wp_2017-11) uses 60- and 120-day rolling co-movement diagnostics and warns that apparent time variation can also arise from finite-sample uncertainty and overlapping windows.
- [Zhang and Tong (2022), *Asymptotic Theory of Principal Component Analysis for Time Series Data with Cautionary Comments*](https://doi.org/10.1111/rssa.12793) show why time-series dependence matters for inference on PCA loadings and motivate bootstrap-based uncertainty checks.

### 5. Internal factor composition

The analysis compares standard correlation PCA with Varimax and Elastic-Net Sparse PCA. The purpose is not to force an economic story onto PC1, PC2, or PC3. It asks whether the statistical structure can be described using a smaller, more stable set of stocks and whether that description survives benchmark residualization.

The first pass uses three components and a transparent penalty path. Pure L1 and Elastic Net are both reported so that the grouping effect can be seen rather than assumed.

### 6. Sparse identification of local factors

The completed L1-rotation stage searches for sparse directions within the PCA loading space and tests whether any factor is local. Whole-session bootstrap probabilities distinguish stable loading support from one attractive full-sample rotation. The natural extension is a rolling or regime-conditional version that asks whether the same local directions persist through the March 2023 banking stress window.

The dynamic rolling extension is now implemented. It provides past-only and ex-post alignment views, explicit mixed-regime flags, persistence-qualified instability candidates, and 200-versus-500-start sensitivity checks. The next question is whether these local directions agree with the session-level Kalman factors and with the Varimax/Elastic-Net loading maps.

### 7. Covariance estimation and random-matrix diagnostics

Sample covariance will be compared with shrinkage estimators such as Ledoit-Wolf, and potentially with exponentially weighted covariance. Random Matrix Theory will be used as a diagnostic benchmark for separating strong empirical components from noise. The Marchenko-Pastur distribution will not be treated as literal truth because the returns are not iid Gaussian observations.

### 8. Residual dynamics and out-of-sample testing

Only after the factor structure is understood will we study autocorrelation, mean reversion, stationarity, lead-lag relationships, and short-horizon forecasting. Any apparent signal must be evaluated walk-forward, out of sample, and with realistic transaction costs.

## Robustness checks

The main comparisons are:

```text
CORE universe             vs FULL universe
Covariance PCA             vs Correlation PCA
Raw returns                vs Intraday-normalized returns
Raw returns                vs SPY/XLF residual returns
1-minute data              vs 5-minute aggregation
Sample covariance          vs Shrinkage covariance
All periods                vs Stress periods excluded
Full-quality names         vs Lower-coverage names such as GS
```

The purpose of these checks is not to accumulate techniques. It is to understand which conclusions survive reasonable changes in data treatment and model specification.

## Current status

Completed:

- Historical SIP data acquisition
- Official session-calendar handling, including early closes
- Monthly download chunking and resume support
- Per-symbol and per-day data-quality reports
- Minute-level missing-data matrix and systemic-gap detection
- Initial classification of security-specific missing bars and stress intervals
- Within-session log-return construction
- Contaminated-return masking
- Complete-panel construction
- CORE and FULL research datasets
- Baseline covariance and correlation PCA on the CORE universe
- Intraday-volatility profile and normalized PCA robustness check
- SPY/XLF residualization and PCA of the remaining CORE structure
- Variance ledger showing how benchmark and residual PCA components add back to raw variance
- Initial code cleanup and local Git versioning
- Descriptive rolling covariance/correlation PCA with 20- and 60-session windows
- Reusable PCA and benchmark-residualization modules
- Internal factor-isolation script with Varimax and Elastic-Net Sparse PCA
- L1 local-factor identification with oblique rotation and the reference small-loading test
- Whole-session bootstrap support probabilities with optimal factor alignment
- Factor-count and `K=2,...,5` sensitivity diagnostics for the local-factor conclusion
- Dynamic 60/120-session rolling L1 local-factor stability, regime summaries, alignment diagnostics, and 500-start sensitivity checks
- Shared session-window helpers, synthetic support-shift tests, and import-safe stage-14 regression coverage
- Leakage-controlled factor-adjusted realized-volatility network HAR, HAC comparisons, block bootstrap stability, descriptive controls, and production diagnostics

Next:

- Freeze the Stage 15 primary specification and evaluate it on a genuinely untouched future holdout
- Extend the March–May 2023 stress comparison with a forecast design whose burn-in permits valid event-period predictions
- Increase block-bootstrap replications when the production specification is finalized

Later:

- Rolling Sparse PCA and stress-regime comparison
- Shrinkage covariance
- Random-matrix diagnostics
- Residual dynamics
- Walk-forward testing and transaction-cost analysis

## Reproducibility

The downloader reads credentials from environment variables:

```text
ALPACA_API_KEY
ALPACA_SECRET_KEY
```

Install the Python dependencies listed in `requirements.txt`. The current scripts can then be run from the project root in this order:

```text
01-crwal.py             download or resume raw SIP data and build reports
02-inspect_missing.py   inspect the minute-by-symbol missing matrix
03-preprocess.py       build synchronized prices and returns
04-check_returns.py    inspect return distributions and extremes
05-clean_returns.py    remove bad sessions and contaminated returns
06_save_universes.py   save the CORE and FULL research panels
07_baseline_pca.py     run covariance and correlation PCA with baseline plots
08_intraday_normalization.py  estimate intraday volatility and repeat PCA
09_benchmark_residualization.py remove SPY/XLF exposure and run residual PCA
10_variance_decomposition.py  reconcile raw variance with benchmark and residual PCA parts
11_rolling_pca.py             run descriptive 20/60-session rolling PCA diagnostics
12_internal_factor_isolation.py  compare PCA, Varimax, and Elastic-Net Sparse PCA
13_l1_local_factor_identification.py  identify and bootstrap sparse local factors
13_bis_kalman_dynamic_factors.py  compare session Kalman factors + L1 with static PCA + L1
14_dynamic_local_factor_regimes.py  run 60/120-session rolling L1 local-factor regime diagnostics
15_factor_adjusted_residual_network.py  forecast factor-adjusted realized volatility with a sparse network HAR
```

The 13-bis extension starts from the intraday benchmark-residualized panel,
aggregates residuals by trading session, estimates AR(1) latent factors with a
Kalman filter/smoother in 60-session windows, and applies the existing L1
rotation inside each PCA subspace. Results are kept separately in
`alpaca_us_banks_1m/reports/kalman_dynamic_local_factors/`, including K=2,4,5
sensitivity checks, holdout residual variance, factor-alignment diagnostics,
structural-group candidates, and the regional-bank stress comparison.

The Stage 15 network extension lives in `utils/factor_adjusted_residuals.py`,
`utils/realized_volatility.py`, and `utils/network_har.py`, with figures and
the compact narrative in `reporting/network.py`. Its generated outputs are
kept separately in `alpaca_us_banks_1m/reports/factor_adjusted_residual_network/`.

The source tree separates research orchestration, reusable calculations, and
presentation. The numbered scripts in `src/` describe each research objective:
they select the input panel, call the required estimators, assemble the output
tables, and request the corresponding reports. Every stage exposes a `main()`
entry point and can be imported without running the pipeline. Existing script
names, report directories, table schemas, and figure filenames are retained.

All shared routines live in the `src/utils/` package. `pca.py` owns ordinary
PCA, Varimax, Elastic-Net Sparse PCA, and reconstruction diagnostics;
`benchmark.py` owns benchmark projection and raw/residual panel construction.
`rolling.py` defines session windows, while `rolling_pca.py` applies PCA across
explicit stock, window, step, and estimator choices. `l1_rotation.py` contains
the rotation geometry and alignment, `local_factor.py` contains session-moment
bootstrap and identification diagnostics, and `kalman.py` contains filtering,
smoothing, and rolling comparison calculations. The numerical stage-14
implementation lives in `utils/dynamic_local_factor_regimes.py`; its execution
flow lives directly in `14_dynamic_local_factor_regimes.py`.

Data acquisition, missingness inspection, preprocessing, panel validation, and
intraday normalization are also kept in `utils/`. Charts and console summaries
live in `src/reporting/`, grouped by research objective. Plot functions consume
computed results and receive their output directory explicitly. Shared figure
styling and saving remain in `utils/plotting.py`; numerical estimators do not
depend on Matplotlib or report modules. Imports therefore use names such as
`from utils.pca import fit_pca`, with the project configuration retained in
`src/config.py`.

The refactor preserves the statistical specifications and their interpretation
limits. In particular, stage 11 still uses a fixed full-sample intraday profile
for descriptive comparisons, and the Kalman extension still residualizes the
intraday panel before session aggregation. Moving these operations into shared
modules does not make either specification a leakage-free predictive test.
Rolling chart dates are parsed explicitly in UTC so that windows spanning
daylight-saving transitions remain renderable.

Run the focused numerical regression suite from the project root with:

```text
python -m unittest discover -s tests -v
```

The September 2026 refactor was checked against the previous implementation on
an identical synthetic panel spanning a daylight-saving transition. Across
stages 07 through 14, all 66 CSV/Parquet tables agreed within a relative
tolerance of `1e-10` and an absolute tolerance of `1e-12`; all 22 rendered
figures were pixel-identical after applying the UTC date-parsing correction to
the original rolling renderer. Bootstrap replications and multistart searches
were reduced equally in both implementations for this comparison. This was
an integration regression, not a rerun of the full production research sample.
The 41-test suite also checks variance reconciliation, session resampling,
configurable rolling windows, holdout isolation in the Kalman training fit,
import safety, computation without file output, factor-projector invariance,
factor-adjustment holdout isolation, and serial-versus-parallel numerical
equivalence for the Stage 15 estimators.

The downloader filename contains a historical typo (`crwal`). It is kept for compatibility with the existing workflow and can be renamed once any external run commands have been updated.

## Research principles

- Never overwrite raw Alpaca Parquet files during analysis.
- Do not interpret a missing OHLC bar as proof that no trades occurred.
- Do not mix overnight returns with one-minute intraday returns.
- Do not clip extreme returns without investigating their economic or data origin.
- Avoid look-ahead bias in any future predictive experiment.
- Remember that PCA eigenvector signs are arbitrary.
- Support economic labels with actual loadings and constituent composition.
- Treat Varimax and Sparse PCA as interpretive tools, not causal identification.
- Do not interpret sparse zero weights as proof that a stock has no economic exposure.
- Treat L1-rotation factors as conditional on the retained PCA dimension and report session-bootstrap instability.
- Treat the CORE universe as the primary specification and the FULL universe as a robustness check.
