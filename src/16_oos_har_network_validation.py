r"""Stage 16: frozen-protocol HAR and Network HAR validation.

Examples from the repository root::

    .\.venv\Scripts\python.exe src\16_oos_har_network_validation.py --mode freeze
    .\.venv\Scripts\python.exe src\16_oos_har_network_validation.py --mode audit
    .\.venv\Scripts\python.exe src\16_oos_har_network_validation.py --mode prospective
    .\.venv\Scripts\python.exe src\16_oos_har_network_validation.py --mode score --run-id RUN_ID
    .\.venv\Scripts\python.exe src\16_oos_har_network_validation.py --mode report --run-id RUN_ID
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import (
    CALENDAR_FILE,
    CLOSE_MATRIX_FILE,
    CONTAMINATED_MASK_FILE,
    MISSING_MASK_FILE,
    REPORTS_DIR,
    ensure_project_directories,
)
from reporting.oos_validation import write_figures, write_results_markdown
from utils.network_har import (
    HAR_LOOKBACK,
    edge_stability,
    group_connectivity,
    network_centrality,
    network_density,
)
from utils.oos_common import (
    dataframe_hash,
    file_prefix_fingerprint,
    file_sha256,
    normalize_calendar,
    stable_hash,
    strict_clean_return_panel,
    utc_now_iso,
    write_json,
)
from utils.oos_har_network import (
    FORECAST_MODELS,
    GROUPS,
    PRIMARY_SPEC_NAME,
    OOSConfig,
    FactorVintageRun,
    ForecastRunResult,
    build_factor_vintage_run,
    build_hac_inference,
    build_protocol_manifest,
    bootstrap_loss_series,
    conditional_edge_bootstrap,
    deserialize_config,
    forecast_checkpoint_specification,
    issue_target_free_forecasts,
    paired_loss_differentials,
    score_forecasts,
    serialize_config,
    summarize_scored_forecasts,
    validate_forecast_ledger,
)


OUT_DIR = REPORTS_DIR / "oos_har_network_validation"
RUNS_DIR = OUT_DIR / "runs"


def _load_calendar() -> pd.DataFrame:
    return pd.read_csv(CALENDAR_FILE)


def _load_strict_complete_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the processed price/missingness artifacts and rebuild strict returns."""

    prices = pd.read_parquet(CLOSE_MATRIX_FILE)
    missing = pd.read_parquet(MISSING_MASK_FILE)
    cleaned, complete, contaminated = strict_clean_return_panel(prices, missing)
    return prices, cleaned, complete, contaminated


def _data_provenance() -> dict[str, object]:
    return {
        "calendar": file_prefix_fingerprint(CALENDAR_FILE),
        "close_matrix": file_prefix_fingerprint(CLOSE_MATRIX_FILE),
        "missing_mask": file_prefix_fingerprint(MISSING_MASK_FILE),
        "legacy_contaminated_mask": file_prefix_fingerprint(CONTAMINATED_MASK_FILE),
    }


def _code_provenance() -> dict[str, object]:
    """Hash the tracked implementation files that define the OOS protocol."""

    files = (
        Path(__file__),
        Path(__file__).parent / "utils" / "oos_common.py",
        Path(__file__).parent / "utils" / "oos_har_network.py",
        Path(__file__).parent / "utils" / "network_har.py",
        Path(__file__).parent / "utils" / "factor_adjusted_residuals.py",
        Path(__file__).parent / "utils" / "preprocessing.py",
        Path(__file__).parent / "utils" / "realized_volatility.py",
        Path(__file__).parent / "reporting" / "oos_validation.py",
    )
    return {path.name: file_sha256(path) for path in files if path.exists()}


def _run_id(mode: str, config: OOSConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{mode}_{config.protocol_version}_{stamp}"


def _run_directory(run_id: str) -> Path:
    directory = RUNS_DIR / run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_table(directory: Path, filename: str, frame: pd.DataFrame) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / filename, index=False)


def _write_indexed_table(directory: Path, filename: str, frame: pd.DataFrame) -> None:
    """Write a session-indexed panel with an explicit session_date column."""

    value = frame.copy()
    value.index.name = "session_date"
    _write_table(directory, filename, value.reset_index())


CHECKPOINT_TABLE_STEMS = (
    "forecast_ledger",
    "har_coefficients",
    "edge_history",
    "tuning_history",
    "forecast_diagnostics",
)
CHECKPOINT_KEY_COLUMNS = {
    "forecast_ledger": (
        "spec_name",
        "model",
        "stock",
        "forecast_origin",
        "target_date",
    ),
    "har_coefficients": (
        "spec_name",
        "model",
        "stock",
        "forecast_origin",
        "target_date",
        "predictor_stock",
        "horizon",
    ),
    "edge_history": (
        "spec_name",
        "model",
        "forecast_origin",
        "target_date",
        "source",
        "target",
        "horizon",
    ),
    "tuning_history": (
        "spec_name",
        "stock",
        "forecast_origin",
        "fraction",
    ),
    "forecast_diagnostics": (
        "spec_name",
        "forecast_origin",
        "target_date",
        "stock",
    ),
}
FACTOR_TABLE_NAMES = (
    "factor_fit_metadata",
    "rv_vintage_metadata",
    "factor_diagnostics",
    "factor_loadings",
    "benchmark_coefficients",
    "factor_metrics",
)


def _read_checkpoint_table(checkpoint_dir: Path, stem: str, table_format: str) -> pd.DataFrame:
    """Read one completed forecast checkpoint without materializing row dictionaries."""

    if table_format not in {"csv", "parquet"}:
        raise ValueError(f"Unsupported checkpoint table format: {table_format}")
    suffix = ".parquet" if table_format == "parquet" else ".csv"
    path = checkpoint_dir / f"{stem}{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"Completed checkpoint is missing {path}.")
    return pd.read_parquet(path) if table_format == "parquet" else pd.read_csv(path)


def _canonicalize_checkpoint_table(stem: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Normalize checkpoint identities and remove equivalent resume overlaps.

    Older CSV rows used a space-separated timestamp while migrated Parquet
    rows use ISO ``T`` separators.  A resume at a checkpoint boundary could
    therefore append the same economic key twice without triggering the raw
    string duplicate check.  Forecast duplicates are accepted only when their
    numerical predictions agree to floating-point tolerance.
    """

    if frame.empty:
        return frame.copy(), 0
    value = frame.copy()
    for column in ("forecast_origin", "target_date"):
        if column in value.columns:
            value[column] = pd.to_datetime(
                value[column], format="mixed", errors="raise"
            ).dt.normalize()
    if "har_window" in value.columns:
        value["har_window"] = value["har_window"].astype("string")
    key = list(CHECKPOINT_KEY_COLUMNS[stem])
    missing = set(key).difference(value.columns)
    if missing:
        raise ValueError(f"Checkpoint {stem} is missing identity columns: {sorted(missing)}")
    duplicate = value.duplicated(key, keep=False)
    if stem == "forecast_ledger" and duplicate.any():
        duplicate_rows = value.loc[duplicate]
        for column in ("predicted_log_variance", "predicted_variance"):
            numeric = duplicate_rows.assign(
                _numeric_value=pd.to_numeric(duplicate_rows[column], errors="raise")
            )
            bounds = numeric.groupby(key, dropna=False)["_numeric_value"].agg(["min", "max"])
            equivalent = np.isclose(
                bounds["min"].to_numpy(dtype=float),
                bounds["max"].to_numpy(dtype=float),
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            )
            if not equivalent.all():
                raise ValueError(
                    f"Checkpoint contains conflicting forecasts after date normalization: {column}"
                )
    before = len(value)
    value = value.drop_duplicates(key, keep="last").reset_index(drop=True)
    if value.duplicated(key).any():
        raise ValueError(f"Checkpoint {stem} remains duplicated after canonicalization.")
    return value, before - len(value)


def _read_indexed_table(path: Path) -> pd.DataFrame:
    """Read a persisted session-indexed panel written by ``_write_indexed_table``."""

    value = pd.read_csv(path)
    if "session_date" not in value.columns:
        raise ValueError(f"Persisted panel has no session_date column: {path}")
    value["session_date"] = pd.to_datetime(
        value["session_date"], format="mixed", errors="raise"
    ).dt.normalize()
    return value.set_index("session_date")


def _load_existing_factor_tables(run_dir: Path) -> list[dict[str, pd.DataFrame]]:
    tables: list[dict[str, pd.DataFrame]] = []
    for name in FACTOR_TABLE_NAMES:
        path = run_dir / f"{name}.csv"
        if path.exists() and path.stat().st_size:
            tables.append({name: pd.read_csv(path)})
    return tables


def _write_factor_tables(run_dir: Path, factor_tables: list[dict[str, pd.DataFrame]]) -> None:
    """Persist factor diagnostics before the long forecast stage starts."""

    for name in FACTOR_TABLE_NAMES:
        _write_table(run_dir, f"{name}.csv", _combine_factor_table(factor_tables, name))


def _load_completed_historical_resume(
    run_dir: Path,
    *,
    specs: list[tuple[int, str, str]],
    calendar: pd.DataFrame,
    config: OOSConfig,
    run_id: str,
) -> tuple[
    ForecastRunResult,
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, int],
] | None:
    """Recover a fully issued historical run without rebuilding factor vintages.

    The shortcut is intentionally all-or-nothing.  Any incomplete checkpoint
    or missing realized-outcome artifact returns control to the normal causal
    construction/resume path.  Incompatible completed artifacts fail loudly.
    """

    calendar_value = normalize_calendar(calendar)
    expected_origins = len(calendar_value) - max(HAR_LOOKBACK.values())
    if expected_origins <= 0:
        return None
    expected_last_origin = pd.Timestamp(
        calendar_value["session_date"].iloc[-2]
    ).normalize()
    states: dict[str, tuple[Path, dict[str, object]]] = {}
    for _, har_window, spec_name in specs:
        checkpoint_dir = run_dir / "checkpoints" / spec_name
        state_path = checkpoint_dir / "forecast_state.json"
        realized_path = run_dir / f"realized_variance_{spec_name}.csv"
        metadata_path = run_dir / f"rv_metadata_{spec_name}.csv"
        if not state_path.exists() or not realized_path.exists() or not metadata_path.exists():
            return None
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("protocol_version") != config.protocol_version:
            raise ValueError(f"Completed checkpoint protocol is incompatible: {spec_name}")
        if state.get("run_id") != run_id or state.get("spec_name") != spec_name:
            raise ValueError(f"Completed checkpoint identity is incompatible: {spec_name}")
        stored_specification_hash = state.get("forecast_specification_hash")
        expected_specification_hash = stable_hash(
            forecast_checkpoint_specification(
                config,
                har_window=har_window,
                mode="historical_pseudo_oos",
                freeze_timestamp=None,
            )
        )
        if (
            stored_specification_hash is not None
            and stored_specification_hash != expected_specification_hash
        ):
            raise ValueError(f"Completed checkpoint specification is incompatible: {spec_name}")
        if int(state.get("next_row", -1)) != expected_origins:
            return None
        last_origin = pd.to_datetime(
            [state.get("last_completed_origin")], format="mixed", errors="raise"
        )[0].normalize()
        if last_origin != expected_last_origin:
            raise ValueError(f"Completed checkpoint calendar is incompatible: {spec_name}")
        states[spec_name] = (checkpoint_dir, state)

    forecasts: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    edges: list[pd.DataFrame] = []
    tuning: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    realized: dict[str, pd.DataFrame] = {}
    metadata: dict[str, pd.DataFrame] = {}
    duplicates_removed: dict[str, int] = {}
    destinations = {
        "forecast_ledger": forecasts,
        "har_coefficients": coefficients,
        "edge_history": edges,
        "tuning_history": tuning,
        "forecast_diagnostics": diagnostics,
    }
    for _, _, spec_name in specs:
        checkpoint_dir, state = states[spec_name]
        table_format = str(state.get("table_format") or "csv")
        for stem in CHECKPOINT_TABLE_STEMS:
            frame, removed = _canonicalize_checkpoint_table(
                stem,
                _read_checkpoint_table(checkpoint_dir, stem, table_format),
            )
            destinations[stem].append(frame)
            if removed:
                duplicates_removed[f"{spec_name}:{stem}"] = removed
        validate_forecast_ledger(forecasts[-1])
        realized[spec_name] = _read_indexed_table(
            run_dir / f"realized_variance_{spec_name}.csv"
        )
        metadata[spec_name] = pd.read_csv(run_dir / f"rv_metadata_{spec_name}.csv")

    return (
        ForecastRunResult(
            forecasts=pd.concat(forecasts, ignore_index=True),
            coefficients=pd.concat(coefficients, ignore_index=True),
            edge_history=pd.concat(edges, ignore_index=True),
            tuning_history=pd.concat(tuning, ignore_index=True),
            origin_diagnostics=pd.concat(diagnostics, ignore_index=True),
            status="complete",
        ),
        realized,
        metadata,
        duplicates_removed,
    )


def _long_controls(vintage: FactorVintageRun) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for control, frame in vintage.variance_result.daily_ivar.items():
        if frame.empty:
            continue
        long = (
            frame.rename_axis("session_date")
            .reset_index()
            .melt(id_vars=["session_date"], var_name="stock", value_name="realized_variance")
        )
        long.insert(0, "spec_name", vintage.spec_name)
        long.insert(1, "factor_window_sessions", vintage.factor_window_sessions)
        long.insert(2, "control", control)
        rows.append(long)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _historical_specs(config: OOSConfig) -> list[tuple[int, str, str]]:
    primary_window = int(config.factor_window_primary)
    specs = [(primary_window, str(config.har_window_primary), f"factor_adjusted_{primary_window}_expanding")]
    specs.append(
        (
            primary_window,
            str(config.har_window_robustness),
            f"factor_adjusted_{primary_window}_rolling252",
        )
    )
    for window in config.factor_windows:
        if int(window) == primary_window:
            continue
        specs.append((int(window), str(config.har_window_primary), f"factor_adjusted_{int(window)}_expanding"))
    return specs


def _factor_result_tables(
    vintage: FactorVintageRun,
    *,
    protocol_version: str,
    run_id: str,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name, frame in (
        ("factor_fit_metadata", vintage.factor_metadata),
        ("rv_vintage_metadata", vintage.rv_metadata),
        ("factor_diagnostics", vintage.factor_result.diagnostics),
        ("factor_loadings", vintage.factor_result.factor_loadings),
        ("benchmark_coefficients", vintage.factor_result.benchmark_coefficients),
        ("factor_metrics", vintage.factor_result.factor_metrics),
    ):
        value = frame.copy()
        for position, (column, content) in enumerate(
            (
                ("protocol_version", protocol_version),
                ("run_id", run_id),
                ("factor_window_sessions", vintage.factor_window_sessions),
            )
        ):
            if column in value.columns:
                value[column] = content
            else:
                value.insert(position, column, content)
        tables[name] = value
    return tables


def _combine_factor_table(tables: list[pd.DataFrame], name: str) -> pd.DataFrame:
    values = [table[name] for table in tables if name in table and not table[name].empty]
    return pd.concat(values, ignore_index=True) if values else pd.DataFrame()


def _period_summary(scored: pd.DataFrame) -> pd.DataFrame:
    value = scored[scored["score_status"] == "scored"].copy()
    if value.empty:
        return pd.DataFrame()
    dates = pd.to_datetime(value["target_date"])
    value["calendar_period"] = dates.dt.to_period("Q").astype(str)
    return summarize_scored_forecasts(
        value,
        ("spec_name", "calendar_period", "model"),
    )


def _comparison_summary(scored: pd.DataFrame) -> pd.DataFrame:
    differences = paired_loss_differentials(scored)
    if differences.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for spec_name, group in differences.groupby("spec_name", dropna=False):
        rows.append(
            {
                "spec_name": spec_name,
                "n_dates": int(group["target_date"].nunique()),
                "n_stock_date": int(len(group)),
                "network_qlike_minus_own": float(group["qlike_difference"].mean()),
                "network_log_mse_minus_own": float(group["log_mse_difference"].mean()),
                "network_log_mae_minus_own": float(group["log_mae_difference"].mean()),
                "relative_qlike_reduction_pct": np.nan,
            }
        )
    # Compute levels and the relative reduction from the matched score ledger.
    result = pd.DataFrame(rows)
    for spec_name in result["spec_name"]:
        own = scored[
            (scored["spec_name"] == spec_name)
            & (scored["model"] == "own_har")
            & (scored["score_status"] == "scored")
        ]["qlike"].mean()
        network = scored[
            (scored["spec_name"] == spec_name)
            & (scored["model"] == "network_har_l1_1se")
            & (scored["score_status"] == "scored")
        ]["qlike"].mean()
        result.loc[result["spec_name"] == spec_name, "own_qlike"] = own
        result.loc[result["spec_name"] == spec_name, "network_qlike"] = network
        result.loc[result["spec_name"] == spec_name, "relative_qlike_reduction_pct"] = (
            100.0 * (own - network) / own if np.isfinite(own) and own != 0 else np.nan
        )
    return result


def _write_run_results(
    run_dir: Path,
    *,
    manifest: dict[str, object],
    forecast_result: ForecastRunResult,
    scored: pd.DataFrame,
    factor_tables: list[dict[str, pd.DataFrame]],
    config: OOSConfig,
    run_id: str,
    edge_bootstrap: pd.DataFrame | None = None,
    make_figures: bool = True,
    write_forecast_ledger: bool = True,
) -> dict[str, pd.DataFrame]:
    """Persist all inspectable tables and return the headline summaries."""

    validate_forecast_ledger(forecast_result.forecasts)
    if write_forecast_ledger:
        _write_table(run_dir, "forecast_ledger.csv", forecast_result.forecasts)
    _write_table(run_dir, "score_ledger.csv", scored)
    _write_table(run_dir, "har_coefficients.csv", forecast_result.coefficients)
    _write_table(run_dir, "edge_history.csv", forecast_result.edge_history)
    _write_table(run_dir, "tuning_history.csv", forecast_result.tuning_history)
    _write_table(run_dir, "forecast_diagnostics.csv", forecast_result.origin_diagnostics)
    # A finalization-only recovery of a legacy completed checkpoint may not
    # have factor diagnostics, because older code deferred writing them until
    # after scoring.  Preserve any existing files and do not replace them with
    # empty tables.  New runs persist these diagnostics before HAR issuance.
    if factor_tables:
        _write_factor_tables(run_dir, factor_tables)

    model_summary = summarize_scored_forecasts(scored, ("spec_name", "model"))
    stock_summary = summarize_scored_forecasts(scored, ("spec_name", "model", "stock"))
    period_summary = _period_summary(scored)
    comparison = _comparison_summary(scored)
    hac = build_hac_inference(scored, max_lags=(5, 20)) if not scored.empty else pd.DataFrame()
    bootstraps = []
    if not scored.empty:
        for offset, block_length in enumerate(config.loss_bootstrap_block_lengths):
            bootstraps.append(
                bootstrap_loss_series(
                    scored,
                    repetitions=config.loss_bootstrap_repetitions,
                    block_length=int(block_length),
                    seed=config.factor_seed + offset,
                )
            )
    bootstrap = (
        pd.concat([frame for frame in bootstraps if not frame.empty], ignore_index=True)
        if bootstraps
        else pd.DataFrame()
    )
    edge_stability_table = edge_stability(forecast_result.edge_history, model="network_har_l1_1se")
    density = network_density(forecast_result.edge_history)
    centrality = network_centrality(edge_stability_table)
    groups = group_connectivity(edge_stability_table, GROUPS)
    for filename, frame in (
        ("model_summary.csv", model_summary),
        ("stock_summary.csv", stock_summary),
        ("period_summary.csv", period_summary),
        ("comparison_summary.csv", comparison),
        ("hac_inference.csv", hac),
        ("bootstrap_inference.csv", bootstrap),
        ("edge_stability.csv", edge_stability_table),
        ("network_density.csv", density),
        ("network_centrality.csv", centrality),
        ("group_connectivity.csv", groups),
        ("conditional_edge_bootstrap.csv", edge_bootstrap if edge_bootstrap is not None else pd.DataFrame()),
    ):
        _write_table(run_dir, filename, frame)
    if make_figures:
        write_figures(
            scored,
            comparison,
            bootstrap,
            edge_stability_table,
            out_dir=run_dir,
        )
    write_results_markdown(
        run_dir / "OOS_RESULTS.md",
        manifest=manifest,
        model_summary=model_summary,
        stock_summary=stock_summary,
        comparison=comparison,
        hac=hac,
        bootstrap=bootstrap,
        edge_stability=edge_stability_table,
        forecast_status=forecast_result.status,
    )
    return {
        "model_summary": model_summary,
        "stock_summary": stock_summary,
        "period_summary": period_summary,
        "comparison": comparison,
        "hac": hac,
        "bootstrap": bootstrap,
        "edge_stability": edge_stability_table,
        "density": density,
        "centrality": centrality,
        "groups": groups,
    }


def _effective_config(base: OOSConfig, *, smoke: bool, n_jobs: int) -> OOSConfig:
    har = replace(base.har, n_jobs=n_jobs)
    value = replace(base, har=har)
    if not smoke:
        return value
    smoke_har = replace(
        har,
        min_training_observations=40,
        cv_splits=2,
        tuning_frequency=10,
        lasso_max_iter=500,
        hac_lag=5,
    )
    return replace(
        value,
        factor_windows=(60,),
        factor_window_primary=60,
        factor_l1_starts=8,
        min_valid_training_observations=40,
        loss_bootstrap_repetitions=50,
        loss_bootstrap_block_lengths=(5,),
        edge_bootstrap_repetitions=10,
        edge_bootstrap_checkpoint_step=30,
        har=smoke_har,
    )


def run_historical(
    *,
    config: OOSConfig,
    run_id: str | None = None,
    smoke: bool = False,
    make_figures: bool = True,
    run_edge_bootstrap: bool = False,
    resume: bool = False,
) -> tuple[Path, dict[str, pd.DataFrame]]:
    """Run historical pseudo-OOS audit/replay for the frozen corrected procedure."""

    started = time.perf_counter()
    run_id = run_id or _run_id("historical_pseudo_oos" if not smoke else "smoke", config)
    run_dir = _run_directory(run_id)
    calendar = _load_calendar()
    prices, cleaned, complete, contaminated = _load_strict_complete_panel()
    if smoke:
        # A smoke run is an execution-path check, not a second historical
        # result.  A final calendar slice remains long enough for the 60-day
        # factor warm-up, 22-session HAR lag, and reduced training minimum.
        calendar_value = normalize_calendar(calendar)
        smoke_dates = pd.DatetimeIndex(calendar_value["session_date"].tail(180))
        calendar_dates = pd.DatetimeIndex(pd.to_datetime(calendar["date"])).normalize()
        calendar = calendar.loc[calendar_dates.isin(smoke_dates)].copy()
        complete_dates = pd.DatetimeIndex(complete.index)
        if complete_dates.tz is not None:
            complete_dates = complete_dates.tz_convert("America/New_York").tz_localize(None)
        complete_dates = complete_dates.normalize()
        keep_rows = complete_dates.isin(smoke_dates)
        complete = complete.loc[keep_rows].copy()
        cleaned = cleaned.loc[cleaned.index.isin(complete.index)].copy()
        contaminated = contaminated.loc[contaminated.index.isin(complete.index)].copy()
    if not isinstance(complete.index, pd.DatetimeIndex):
        raise TypeError("Strict complete returns require a DatetimeIndex.")
    latest_data = str(prices.index.max()) if len(prices.index) else None
    manifest = build_protocol_manifest(
        config,
        freeze_timestamp=utc_now_iso(),
        last_data_examined=latest_data,
        data_provenance=_data_provenance(),
        code_provenance=_code_provenance(),
        mode="historical_pseudo_oos",
        status="historical_pseudo_oos",
    )
    manifest["run_id"] = run_id
    manifest["strict_return_rows"] = int(cleaned.notna().sum().sum())
    manifest["strict_complete_rows"] = int(len(complete))
    manifest["strict_contamination_cells"] = int(contaminated.to_numpy(dtype=bool).sum())
    manifest["input_data_hash"] = dataframe_hash(complete)
    write_json(run_dir / "manifest.json", manifest)

    specs = _historical_specs(config)
    factor_runs: dict[int, FactorVintageRun] = {}
    factor_tables: list[dict[str, pd.DataFrame]] = []
    recovered = (
        _load_completed_historical_resume(
            run_dir,
            specs=specs,
            calendar=calendar,
            config=config,
            run_id=run_id,
        )
        if resume and not run_edge_bootstrap
        else None
    )
    if recovered is not None:
        print(
            "[stage16] all forecast checkpoints complete; "
            "skipping factor reconstruction and HAR estimation",
            flush=True,
        )
        forecast_result, realized, metadata, duplicates_removed = recovered
        factor_tables = _load_existing_factor_tables(run_dir)
        manifest["resume_path"] = "completed_checkpoint_finalization_only"
        manifest["checkpoint_semantic_duplicates_removed"] = duplicates_removed
        manifest["factor_diagnostics_status"] = (
            "loaded_from_run_artifacts"
            if factor_tables
            else "not_available_from_legacy_pre_scoring_checkpoint"
        )
    else:
        for window in sorted(set(int(item[0]) for item in specs)):
            print(f"[stage16] causal factor/RV vintages: {window} sessions", flush=True)
            vintage = build_factor_vintage_run(
                complete,
                calendar,
                factor_window_sessions=window,
                config=config,
                spec_name=f"factor_adjusted_{window}",
            )
            factor_runs[window] = vintage
            factor_tables.append(
                _factor_result_tables(
                    vintage,
                    protocol_version=config.protocol_version,
                    run_id=run_id,
                )
            )
            _write_table(run_dir, f"control_vintages_{window}.csv", _long_controls(vintage))
        # Factor diagnostics are valuable audit artifacts and must survive a
        # later failure in the much longer HAR/scoring portion of the run.
        _write_factor_tables(run_dir, factor_tables)
        manifest["factor_diagnostics_status"] = "persisted_before_forecast_issuance"
        write_json(run_dir / "manifest.json", manifest)

        all_forecasts: list[pd.DataFrame] = []
        all_coefficients: list[pd.DataFrame] = []
        all_edges: list[pd.DataFrame] = []
        all_tuning: list[pd.DataFrame] = []
        all_diagnostics: list[pd.DataFrame] = []
        realized = {}
        metadata = {}
        statuses: list[str] = []
        for window, har_window, spec_name in specs:
            vintage = factor_runs[window]
            print(f"[stage16] target-free forecast issuance: {spec_name}", flush=True)
            checkpoint_dir = run_dir / "checkpoints" / spec_name
            resume_spec = resume and (checkpoint_dir / "forecast_state.json").exists()
            result = issue_target_free_forecasts(
                vintage.daily_log_variance,
                calendar,
                spec_name=spec_name,
                config=config,
                run_id=run_id,
                factor_metadata=vintage.factor_metadata,
                har_window=har_window,
                mode="historical_pseudo_oos",
                checkpoint_dir=checkpoint_dir,
                resume=resume_spec,
            )
            statuses.append(result.status)
            all_forecasts.append(result.forecasts)
            all_coefficients.append(result.coefficients)
            all_edges.append(result.edge_history)
            all_tuning.append(result.tuning_history)
            all_diagnostics.append(result.origin_diagnostics)
            realized[spec_name] = vintage.daily_variance
            metadata[spec_name] = vintage.rv_metadata.assign(spec_name=spec_name)
            _write_indexed_table(
                run_dir,
                f"realized_variance_{spec_name}.csv",
                vintage.daily_variance,
            )
            _write_table(run_dir, f"rv_metadata_{spec_name}.csv", metadata[spec_name])
        forecast_result = ForecastRunResult(
            forecasts=pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame(),
            coefficients=pd.concat(all_coefficients, ignore_index=True) if all_coefficients else pd.DataFrame(),
            edge_history=pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame(),
            tuning_history=pd.concat(all_tuning, ignore_index=True) if all_tuning else pd.DataFrame(),
            origin_diagnostics=pd.concat(all_diagnostics, ignore_index=True) if all_diagnostics else pd.DataFrame(),
            status=(
                "complete"
                if all(status == "complete" for status in statuses)
                else "checkpointed_partial"
            ),
        )
    scored = score_forecasts(
        forecast_result.forecasts,
        realized,
        metadata,
        config=config,
    )
    edge_bootstrap = None
    if run_edge_bootstrap and PRIMARY_SPEC_NAME in realized:
        edge_bootstrap = conditional_edge_bootstrap(
            factor_runs[config.factor_window_primary].daily_log_variance,
            calendar,
            spec_name=PRIMARY_SPEC_NAME,
            config=config,
        )
    manifest["origin_calendar"] = sorted(
        str(value)
        for value in forecast_result.forecasts["target_date"].dropna().unique()
    )
    manifest["historical_execution_seconds"] = time.perf_counter() - started
    manifest["forecast_status"] = forecast_result.status
    manifest["edge_bootstrap_status"] = "executed" if run_edge_bootstrap else "not_run_optional"
    write_json(run_dir / "manifest.json", manifest)
    summaries = _write_run_results(
        run_dir,
        manifest=manifest,
        forecast_result=forecast_result,
        scored=scored,
        factor_tables=factor_tables,
        config=config,
        run_id=run_id,
        edge_bootstrap=edge_bootstrap,
        make_figures=make_figures,
    )
    print(f"[stage16] completed in {time.perf_counter() - started:.1f}s: {run_dir}", flush=True)
    return run_dir, summaries


def run_freeze(*, config: OOSConfig) -> Path:
    """Register the corrected protocol before any eligible future outcome exists."""

    run_id = _run_id("protocol_freeze", config)
    run_dir = _run_directory(run_id)
    latest_data = None
    if CLOSE_MATRIX_FILE.exists():
        latest_data = str(pd.read_parquet(CLOSE_MATRIX_FILE).index.max())
    freeze = utc_now_iso()
    manifest = build_protocol_manifest(
        config,
        freeze_timestamp=freeze,
        last_data_examined=latest_data,
        data_provenance=_data_provenance(),
        code_provenance=_code_provenance(),
        mode="protocol_freeze",
        status="ready / awaiting_future_data",
    )
    manifest["run_id"] = run_id
    manifest["prospective_rule"] = "only target sessions whose open is after freeze_timestamp_utc"
    manifest["prospective_evidence"] = "none until a forecast is issued before target outcome availability"
    write_json(run_dir / "manifest.json", manifest)
    write_json(OUT_DIR / "latest_protocol_manifest.json", manifest)
    print(f"[stage16] protocol frozen: {run_dir}", flush=True)
    return run_dir


def run_prospective(*, config: OOSConfig, freeze_manifest: Path | None = None, make_figures: bool = True) -> Path:
    """Issue only post-freeze forecasts; current data normally yields an empty ledger."""

    if freeze_manifest is None:
        freeze_manifest = OUT_DIR / "latest_protocol_manifest.json"
    if not freeze_manifest.exists():
        raise FileNotFoundError("Run --mode freeze before prospective issuance.")
    frozen = json.loads(freeze_manifest.read_text(encoding="utf-8"))
    if frozen.get("status") != "ready / awaiting_future_data":
        raise ValueError("The supplied manifest is not a prospective protocol freeze.")
    frozen_config = deserialize_config(frozen.get("configuration", {}))
    if frozen.get("protocol_hash") != stable_hash(serialize_config(frozen_config)):
        raise ValueError("The frozen protocol manifest failed its configuration hash check.")
    if frozen.get("code_provenance") != _code_provenance():
        raise ValueError("Protocol code changed after freeze; run freeze again before issuance.")
    # Worker count is an execution control, not a research decision.  Every
    # statistical setting comes from the immutable freeze rather than today's
    # source defaults or command line.
    config = replace(
        frozen_config,
        har=replace(frozen_config.har, n_jobs=config.har.n_jobs),
    )
    issuance_timestamp = utc_now_iso()
    run_id = _run_id("prospective", config)
    run_dir = _run_directory(run_id)
    calendar = _load_calendar()
    _, _, complete, _ = _load_strict_complete_panel()
    factor_runs: dict[int, FactorVintageRun] = {}
    factor_tables: list[dict[str, pd.DataFrame]] = []
    for window in sorted(set(int(item[0]) for item in _historical_specs(config))):
        vintage = build_factor_vintage_run(
            complete,
            calendar,
            factor_window_sessions=window,
            config=config,
            spec_name=f"factor_adjusted_{window}",
        )
        factor_runs[window] = vintage
        factor_tables.append(_factor_result_tables(vintage, protocol_version=config.protocol_version, run_id=run_id))
        _write_table(run_dir, f"control_vintages_{window}.csv", _long_controls(vintage))
    for name in (
        "factor_fit_metadata",
        "rv_vintage_metadata",
        "factor_diagnostics",
        "factor_loadings",
        "benchmark_coefficients",
        "factor_metrics",
    ):
        _write_table(run_dir, f"{name}.csv", _combine_factor_table(factor_tables, name))
    forecasts: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    edges: list[pd.DataFrame] = []
    tuning: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    for window, har_window, spec_name in _historical_specs(config):
        vintage = factor_runs[window]
        result = issue_target_free_forecasts(
            vintage.daily_log_variance,
            calendar,
            spec_name=spec_name,
            config=config,
            run_id=run_id,
            factor_metadata=vintage.factor_metadata,
            har_window=har_window,
            mode="prospective",
            freeze_timestamp=frozen["freeze_timestamp_utc"],
            issuance_timestamp=issuance_timestamp,
        )
        forecasts.append(result.forecasts)
        coefficients.append(result.coefficients)
        edges.append(result.edge_history)
        tuning.append(result.tuning_history)
        diagnostics.append(result.origin_diagnostics)
        _write_indexed_table(run_dir, f"realized_variance_{spec_name}.csv", vintage.daily_variance)
        _write_indexed_table(run_dir, f"realized_log_variance_{spec_name}.csv", vintage.daily_log_variance)
        _write_table(
            run_dir,
            f"rv_metadata_{spec_name}.csv",
            vintage.rv_metadata.assign(spec_name=spec_name),
        )
    forecast_result = ForecastRunResult(
        forecasts=pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame(),
        coefficients=pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame(),
        edge_history=pd.concat(edges, ignore_index=True) if edges else pd.DataFrame(),
        tuning_history=pd.concat(tuning, ignore_index=True) if tuning else pd.DataFrame(),
        origin_diagnostics=pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame(),
        status=(
            "awaiting_future_data"
            if not any(not frame.empty for frame in forecasts)
            else "issued_pending_outcomes"
        ),
    )
    if forecast_result.forecasts.empty:
        forecast_result = replace(
            forecast_result,
            forecasts=pd.DataFrame(
                columns=[
                    "protocol_version",
                    "run_id",
                    "spec_name",
                    "model",
                    "stock",
                    "forecast_origin",
                    "as_of",
                    "target_date",
                    "target_open",
                    "target_outcome_available_at",
                    "training_label_cutoff",
                    "training_label_available_at",
                    "factor_fit_id",
                    "target_data_present_at_issue",
                    "predicted_log_variance",
                    "predicted_variance",
                ]
            ),
        )
    for frame in (forecast_result.forecasts,):
        validate_forecast_ledger(frame)
    latest = dict(frozen)
    latest.update(
        {
            "run_id": run_id,
            "mode": "prospective",
            "status": forecast_result.status,
            "forecast_ledger_immutable_before_scoring": True,
            "issued_at_utc": issuance_timestamp,
            "n_forecast_rows": int(len(forecast_result.forecasts)),
        }
    )
    _write_table(run_dir, "forecast_ledger.csv", forecast_result.forecasts)
    latest["forecast_ledger_sha256"] = file_sha256(run_dir / "forecast_ledger.csv")
    write_json(run_dir / "manifest.json", latest)
    _write_table(run_dir, "har_coefficients.csv", forecast_result.coefficients)
    _write_table(run_dir, "edge_history.csv", forecast_result.edge_history)
    _write_table(run_dir, "tuning_history.csv", forecast_result.tuning_history)
    _write_table(run_dir, "forecast_diagnostics.csv", forecast_result.origin_diagnostics)
    print(f"[stage16] prospective state: {forecast_result.status}; {run_dir}", flush=True)
    return run_dir


def run_report(*, run_id: str | None = None) -> Path:
    """Regenerate the durable report from an existing run directory."""

    if run_id is None:
        directories = sorted([path for path in RUNS_DIR.iterdir() if path.is_dir()]) if RUNS_DIR.exists() else []
        if not directories:
            raise FileNotFoundError("No Stage 16 run directories are available.")
        run_dir = directories[-1]
    else:
        run_dir = _run_directory(run_id)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    forecast = pd.read_csv(run_dir / "forecast_ledger.csv")
    score_path = run_dir / "score_ledger.csv"
    if score_path.exists():
        scored = pd.read_csv(score_path)
        model_summary = pd.read_csv(run_dir / "model_summary.csv")
        stock_summary = pd.read_csv(run_dir / "stock_summary.csv")
        comparison = pd.read_csv(run_dir / "comparison_summary.csv")
        hac = pd.read_csv(run_dir / "hac_inference.csv")
        bootstrap = pd.read_csv(run_dir / "bootstrap_inference.csv")
        edge = pd.read_csv(run_dir / "edge_stability.csv")
    else:
        scored = pd.DataFrame()
        model_summary = stock_summary = comparison = hac = bootstrap = edge = pd.DataFrame()
    write_results_markdown(
        run_dir / "OOS_RESULTS.md",
        manifest=manifest,
        model_summary=model_summary,
        stock_summary=stock_summary,
        comparison=comparison,
        hac=hac,
        bootstrap=bootstrap,
        edge_stability=edge,
        forecast_status=manifest.get("status", "unknown"),
    )
    print(f"[stage16] report regenerated: {run_dir / 'OOS_RESULTS.md'}", flush=True)
    return run_dir / "OOS_RESULTS.md"


def run_score(*, run_id: str | None = None, n_jobs: int = 1) -> Path:
    """Join stored prospective outcomes to an immutable forecast ledger."""

    if run_id is None:
        directories = sorted([path for path in RUNS_DIR.iterdir() if path.is_dir()]) if RUNS_DIR.exists() else []
        if not directories:
            raise FileNotFoundError("No Stage 16 run directories are available.")
        run_dir = directories[-1]
    else:
        run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Stage 16 run directory does not exist: {run_dir}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mode") != "prospective":
        raise ValueError("Score mode is reserved for an issued prospective run.")
    forecast_path = run_dir / "forecast_ledger.csv"
    expected_forecast_hash = manifest.get("forecast_ledger_sha256")
    if expected_forecast_hash and file_sha256(forecast_path) != expected_forecast_hash:
        raise ValueError("The immutable forecast ledger has changed since issuance.")
    forecast = pd.read_csv(forecast_path)
    validate_forecast_ledger(forecast)
    stored_config = deserialize_config(manifest.get("configuration", {}))
    config = replace(stored_config, har=replace(stored_config.har, n_jobs=n_jobs))
    specs = sorted(str(value) for value in forecast["spec_name"].dropna().unique())
    realized: dict[str, pd.DataFrame] = {}
    metadata: dict[str, pd.DataFrame] = {}
    if specs:
        calendar = _load_calendar()
        _, _, complete, _ = _load_strict_complete_panel()
        spec_windows = {
            spec_name: int(window)
            for window, _, spec_name in _historical_specs(config)
        }
        unknown = sorted(set(specs).difference(spec_windows))
        if unknown:
            raise ValueError(f"Forecast ledger contains unknown specifications: {unknown}")
        factor_runs: dict[int, FactorVintageRun] = {}
        for window in sorted({spec_windows[name] for name in specs}):
            factor_runs[window] = build_factor_vintage_run(
                complete,
                calendar,
                factor_window_sessions=window,
                config=config,
                spec_name=f"factor_adjusted_{window}",
            )
        for spec_name in specs:
            vintage = factor_runs[spec_windows[spec_name]]
            realized[spec_name] = vintage.daily_variance
            metadata[spec_name] = vintage.rv_metadata.assign(spec_name=spec_name)
            _write_indexed_table(
                run_dir,
                f"realized_variance_{spec_name}.csv",
                vintage.daily_variance,
            )
            _write_table(run_dir, f"rv_metadata_{spec_name}.csv", metadata[spec_name])
    scored = score_forecasts(forecast, realized, metadata, config=config)
    _write_table(run_dir, "score_ledger.csv", scored)
    n_scored = int((scored["score_status"] == "scored").sum())
    manifest["status"] = "confirmation_in_progress" if n_scored else "awaiting_future_data"
    manifest["n_scored_rows"] = n_scored
    manifest["scored_at_utc"] = utc_now_iso()
    manifest["outcome_data_provenance"] = _data_provenance()
    write_json(run_dir / "manifest.json", manifest)
    if forecast.empty:
        write_results_markdown(
            run_dir / "OOS_RESULTS.md",
            manifest=manifest,
            model_summary=pd.DataFrame(),
            stock_summary=pd.DataFrame(),
            comparison=pd.DataFrame(),
            hac=pd.DataFrame(),
            bootstrap=pd.DataFrame(),
            edge_stability=pd.DataFrame(),
            forecast_status=manifest["status"],
        )
        return run_dir / "OOS_RESULTS.md"
    forecast_result = ForecastRunResult(
        forecasts=forecast,
        coefficients=(
            pd.read_csv(run_dir / "har_coefficients.csv")
            if (run_dir / "har_coefficients.csv").exists()
            else pd.DataFrame()
        ),
        edge_history=(
            pd.read_csv(run_dir / "edge_history.csv")
            if (run_dir / "edge_history.csv").exists()
            else pd.DataFrame()
        ),
        tuning_history=(
            pd.read_csv(run_dir / "tuning_history.csv")
            if (run_dir / "tuning_history.csv").exists()
            else pd.DataFrame()
        ),
        origin_diagnostics=(
            pd.read_csv(run_dir / "forecast_diagnostics.csv")
            if (run_dir / "forecast_diagnostics.csv").exists()
            else pd.DataFrame()
        ),
        status="complete",
    )
    factor_tables = []
    for name in (
        "factor_fit_metadata",
        "rv_vintage_metadata",
        "factor_diagnostics",
        "factor_loadings",
        "benchmark_coefficients",
        "factor_metrics",
    ):
        path = run_dir / f"{name}.csv"
        if path.exists():
            factor_tables.append({name: pd.read_csv(path)})
    edge_bootstrap_path = run_dir / "conditional_edge_bootstrap.csv"
    edge_bootstrap = pd.read_csv(edge_bootstrap_path) if edge_bootstrap_path.exists() else None
    _write_run_results(
        run_dir,
        manifest=manifest,
        forecast_result=forecast_result,
        scored=scored,
        factor_tables=factor_tables,
        config=config,
        run_id=str(manifest.get("run_id", run_dir.name)),
        edge_bootstrap=edge_bootstrap,
        make_figures=False,
        write_forecast_ledger=False,
    )
    print(f"[stage16] outcomes scored: {n_scored} rows; {run_dir}", flush=True)
    return run_dir / "OOS_RESULTS.md"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("audit", "smoke", "freeze", "prospective", "score", "report"),
        required=True,
        help="Stage 16 operation to run.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Existing run id for resume/report operations.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel target workers for factor/Lasso routines.",
    )
    parser.add_argument(
        "--edge-bootstrap",
        action="store_true",
        help="Run the optional 200-replication conditional edge bootstrap.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip report figures.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume compatible forecast checkpoints.",
    )
    args = parser.parse_args(argv)
    if args.n_jobs < 1:
        parser.error("--n-jobs must be at least one")
    ensure_project_directories()
    base = OOSConfig()
    config = _effective_config(base, smoke=args.mode == "smoke", n_jobs=args.n_jobs)
    if args.mode == "freeze":
        run_freeze(config=config)
    elif args.mode == "prospective":
        run_prospective(config=config, make_figures=not args.no_figures)
    elif args.mode == "score":
        run_score(run_id=args.run_id, n_jobs=args.n_jobs)
    elif args.mode == "report":
        run_report(run_id=args.run_id)
    else:
        run_historical(
            config=config,
            run_id=args.run_id,
            smoke=args.mode == "smoke",
            make_figures=not args.no_figures,
            run_edge_bootstrap=args.edge_bootstrap,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
