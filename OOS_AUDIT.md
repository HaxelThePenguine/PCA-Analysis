# Stage 16 Audit of the Existing HAR Network

The audit began from commit `7aa9c8a32e4c495a45fd7adce672c839703b039b` and
keeps the Stage 15 generated artifacts and historical claims separate. The
table records the implementation-level findings that motivated the corrected
version. A historical pseudo-OOS replay can measure their numerical effect;
the prospective track is frozen separately and cannot be retroactively
treated as a holdout.

| Component | Current behavior at audit start | Evidence | Issue or verified invariant | Action in v2 | Effect on comparability |
| --- | --- | --- | --- | --- | --- |
| Stage 15 outer forecasting | Past-only factor windows, chronological tuning, training-only smearing and HAC comparison | `src/15_factor_adjusted_residual_network.py`; Stage 15 ledgers | Correctly walk-forward at the model level, but historical sample was already researched | Preserve legacy stage and add a separate target-free ledger | Legacy outputs remain intact; v2 is versioned |
| Lasso `alpha_max` | Threshold used an unstandardized partialled-out cross design while the solver standardized it | `src/utils/network_har.py`; synthetic threshold check | Zero-solution threshold was not on the solver scale | Standardize cross residuals before `alpha_max` in every fold and full fit | Historical numbers may change; v1 artifacts are not overwritten |
| Missing-minute returns | Differencing after removed timestamps could label a multi-minute move as a one-minute return | `src/utils/preprocessing.py`; 2023-06-05 gap reproduction | First post-gap return could enter factor estimation | Require consecutive same-session endpoints; retain masks | v2 target and factor vintages are not numerically identical to v1 |
| Session calendar | Existing aggregation handled session clock and early closes, but daily feature construction could use row adjacency | `src/utils/realized_volatility.py`; calendar metadata | Missing sessions must not collapse HAR lags | Reindex daily targets to the exchange calendar before D/W/M features | Calendar-based eligibility is explicit; historical date counts can change |
| HAR feature timing | Existing features end at the forecast origin, but target-free storage was not separated from scoring | `src/utils/network_har.py` and Stage 15 ledger | Need an immutable forecast object before outcome join | Separate issuance, scoring and reporting modes | Enables genuine prospective confirmation |
| CV fallback | Fallback behavior was not explicit when valid folds or losses were absent | `tune_network_penalty` path | A silent weakest-penalty selection is not auditable | Record status, fold counts and fallback reason; use strongest candidate on no loss | Candidate choice is inspectable and versioned |
| Numerical conditioning | `condition_number` described a column-norm ratio | HAR fit diagnostics | It was not a singular-value condition number | Record SVD condition number and preserve norm ratio as a separate field | Existing diagnostic names remain available with corrected meaning |
| Factor rotation | L1 rotation was used to define the residual projector | Factor projector tests | Full retained-subspace residualization is rotation invariant | Use the PCA projector for residual construction; keep L1 for interpretation | Factor-adjusted target is geometrically stable even if labels fail |
| Realized-variance scoring | Legacy path could score a clipped log target proxy | Stage 15 forecast/scoring code | Observed positive RV must enter QLIKE directly | Join raw positive RV only after issuance; record floors and unscorable targets | QLIKE is no longer silently changed by target clipping |
| Network computation | Stage 15 uses target-wise workers, partialling-out reuse and warm starts | `src/utils/network_har.py` | These are useful performance invariants | Reuse design arrays, shared projections, Gram products and warm starts; execute independent target equations concurrently while the parent process owns checkpoint state | Forecasts remain deterministic across worker counts; checkpoint snapshots use compressed Parquet at a lower operational cadence |
| Evidence status | Stage 15 was called pseudo-OOS and future holdout was pending | `README.md`, `RESULTS_TO_DATE.md` | Historical evidence is not untouched confirmation | Freeze `oos-har-network-v2.0.0` with actual UTC timestamp | Claims are separated into historical replay and prospective readiness |

The full historical numerical effect of the corrected strict-return and
standardized-penalty rules is reported only after a successful v2 audit run in
`OOS_RESULTS.md`. No corrected market-data performance number is inferred from
the Stage 15 table alone.
