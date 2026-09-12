r"""Stage 17: OOS predictive validation of common and sparse local factors.

Examples from the repository root::

    .\.venv\Scripts\python.exe src\17_oos_factor_augmented_har.py --mode smoke --n-jobs 2
    .\.venv\Scripts\python.exe src\17_oos_factor_augmented_har.py --mode audit --n-jobs 12
    .\.venv\Scripts\python.exe src\17_oos_factor_augmented_har.py --mode freeze
    .\.venv\Scripts\python.exe src\17_oos_factor_augmented_har.py --mode prospective --n-jobs 12
    .\.venv\Scripts\python.exe src\17_oos_factor_augmented_har.py --mode score --run-id RUN_ID
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import (
    CALENDAR_FILE,
    CLOSE_MATRIX_FILE,
    MISSING_MASK_FILE,
    REPORTS_DIR,
    ensure_project_directories,
)
from reporting.factor_har import (
    build_hac_comparisons,
    summarize_models,
    write_results_markdown,
)
from utils.factor_har_oos import (
    FACTOR_HAR_MODELS,
    issue_factor_har_forecasts,
    matched_qlike_comparisons,
)
from utils.network_har import HARConfig
from utils.oos_common import (
    file_sha256,
    normalize_calendar,
    stable_hash,
    strict_clean_return_panel,
    utc_now_iso,
    write_json,
)
from utils.oos_har_network import (
    OOSConfig,
    FactorVintageRun,
    ForecastRunResult,
    build_factor_vintage_run,
    deserialize_config,
    score_forecasts,
    serialize_config,
    validate_forecast_ledger,
)


OUT_DIR = REPORTS_DIR / "oos_factor_augmented_har"
RUNS_DIR = OUT_DIR / "runs"
PROTOCOL_VERSION = "oos-factor-har-v1.0.0"
PRIMARY_SPEC = "benchmark_factor_120_expanding"


def _specifications(config: OOSConfig) -> tuple[tuple[int, str | int, str], ...]:
    if config.factor_windows == (60,) and config.factor_window_primary == 60:
        return ((60, "expanding", "benchmark_factor_60_expanding_smoke"),)
    return (
        (config.factor_window_primary, "expanding", PRIMARY_SPEC),
        (config.factor_window_primary, 252, "benchmark_factor_120_rolling252"),
        (60, "expanding", "benchmark_factor_60_expanding"),
    )


def _effective_config(*, n_jobs: int, smoke: bool = False) -> OOSConfig:
    har = HARConfig(n_jobs=n_jobs, random_seed=17017)
    value = OOSConfig(
        protocol_version=PROTOCOL_VERSION,
        factor_seed=17017,
        har=har,
    )
    if not smoke:
        return value
    return replace(
        value,
        factor_windows=(60,),
        factor_window_primary=60,
        factor_l1_starts=8,
        min_valid_training_observations=40,
        har=replace(
            har,
            min_training_observations=40,
            rolling_training_observations=60,
            tuning_frequency=10,
            cv_splits=2,
            alpha_fractions=(0.1, 0.3, 1.0),
        ),
    )


def _load_inputs(smoke: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar = pd.read_csv(CALENDAR_FILE)
    prices = pd.read_parquet(CLOSE_MATRIX_FILE)
    missing = pd.read_parquet(MISSING_MASK_FILE)
    _, complete, _ = strict_clean_return_panel(prices, missing)
    if smoke:
        calendar_value = normalize_calendar(calendar)
        dates = pd.DatetimeIndex(calendar_value["session_date"].tail(180))
        source_dates = pd.DatetimeIndex(complete.index)
        if source_dates.tz is not None:
            source_dates = source_dates.tz_convert("America/New_York").tz_localize(None)
        keep = source_dates.normalize().isin(dates)
        complete = complete.loc[keep]
        raw_dates = pd.DatetimeIndex(pd.to_datetime(calendar["date"])).normalize()
        calendar = calendar.loc[raw_dates.isin(dates)]
    return complete, calendar


def _code_provenance() -> dict[str, str]:
    source = Path(__file__).parent
    files = (
        Path(__file__),
        source / "utils" / "oos_common.py",
        source / "utils" / "factor_har_oos.py",
        source / "utils" / "oos_har_network.py",
        source / "utils" / "factor_adjusted_residuals.py",
        source / "utils" / "network_har.py",
        source / "utils" / "realized_volatility.py",
        source / "reporting" / "factor_har.py",
    )
    return {path.name: file_sha256(path) for path in files}


def _manifest(config: OOSConfig, *, mode: str, freeze: str, status: str) -> dict[str, object]:
    configuration = serialize_config(config)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "status": status,
        "freeze_timestamp_utc": freeze,
        "protocol_hash": stable_hash(configuration),
        "configuration": configuration,
        "code_provenance": _code_provenance(),
        "response": "next-session benchmark-residual log realized variance",
        "models": list(FACTOR_HAR_MODELS),
        "primary_specification": PRIMARY_SPEC,
        "primary_comparison": "aggregate_factor_har minus own_har",
        "primary_loss": "QLIKE on raw positive realized variance",
        "factor_information_rule": "factor fit and score at d use information available no later than close(d)",
        "forecast_information_rule": "target d+1 is never read during issuance",
        "matched_sample_rule": "all six models use identical training and evaluation keys",
    }


def _run_directory(run_id: str) -> Path:
    value = RUNS_DIR / run_id
    value.mkdir(parents=True, exist_ok=True)
    return value


def _run_id(mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{mode}_{PROTOCOL_VERSION}_{stamp}"


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_vintage(run_dir: Path, window: int, vintage: FactorVintageRun) -> None:
    for name, frame in (
        ("benchmark_log_variance", vintage.benchmark_daily_log_variance),
        ("benchmark_variance", vintage.benchmark_daily_variance),
        ("aggregate_factor_log_variance", vintage.aggregate_factor_daily_log_variance),
        ("local_factor_log_variance", vintage.local_factor_daily_log_variance),
    ):
        frame.to_parquet(run_dir / f"{name}_{window}.parquet")
    _write_frame(run_dir / f"factor_metadata_{window}.csv", vintage.factor_metadata)
    _write_frame(run_dir / f"rv_metadata_{window}.csv", vintage.rv_metadata)


def _read_vintage(run_dir: Path, window: int) -> dict[str, pd.DataFrame] | None:
    paths = {
        "benchmark_log": run_dir / f"benchmark_log_variance_{window}.parquet",
        "benchmark_rv": run_dir / f"benchmark_variance_{window}.parquet",
        "aggregate_log": run_dir / f"aggregate_factor_log_variance_{window}.parquet",
        "local_log": run_dir / f"local_factor_log_variance_{window}.parquet",
        "factor_metadata": run_dir / f"factor_metadata_{window}.csv",
        "rv_metadata": run_dir / f"rv_metadata_{window}.csv",
    }
    if not all(path.exists() for path in paths.values()):
        return None
    return {
        name: pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        for name, path in paths.items()
    }


def _vintage_tables(vintage: FactorVintageRun) -> dict[str, pd.DataFrame]:
    return {
        "benchmark_log": vintage.benchmark_daily_log_variance,
        "benchmark_rv": vintage.benchmark_daily_variance,
        "aggregate_log": vintage.aggregate_factor_daily_log_variance,
        "local_log": vintage.local_factor_daily_log_variance,
        "factor_metadata": vintage.factor_metadata,
        "rv_metadata": vintage.rv_metadata,
    }


def _empty_forecast_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "protocol_version", "run_id", "spec_name", "model", "stock",
            "forecast_origin", "as_of", "target_date", "target_open",
            "target_outcome_available_at", "training_label_cutoff",
            "target_data_present_at_issue", "predicted_log_variance", "predicted_variance",
        ]
    )


def _fit_specs(
    complete: pd.DataFrame,
    calendar: pd.DataFrame,
    config: OOSConfig,
    run_dir: Path,
    *,
    run_id: str,
    mode: str,
    freeze_timestamp: str | None = None,
    resume: bool = False,
) -> tuple[ForecastRunResult, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    specs = _specifications(config)
    vintages: dict[int, dict[str, pd.DataFrame]] = {}
    for window in sorted({window for window, _, _ in specs}):
        stored = _read_vintage(run_dir, window) if resume else None
        if stored is not None:
            print(f"[stage17] loaded factor/RV vintages: {window} sessions", flush=True)
            vintages[window] = stored
            continue
        print(f"[stage17] causal factor/RV vintages: {window} sessions", flush=True)
        vintage = build_factor_vintage_run(
            complete,
            calendar,
            factor_window_sessions=window,
            config=config,
            spec_name=f"factor_{window}",
        )
        _write_vintage(run_dir, window, vintage)
        vintages[window] = _vintage_tables(vintage)

    result_frames: dict[str, list[pd.DataFrame]] = {
        "forecasts": [], "coefficients": [], "edges": [], "tuning": [], "diagnostics": []
    }
    realized: dict[str, pd.DataFrame] = {}
    metadata: dict[str, pd.DataFrame] = {}
    for window, har_window, spec_name in specs:
        vintage = vintages[window]
        realized[spec_name] = vintage["benchmark_rv"]
        metadata[spec_name] = vintage["rv_metadata"].assign(spec_name=spec_name)
        stored_forecast = run_dir / f"forecast_ledger_{spec_name}.parquet"
        if resume and stored_forecast.exists():
            print(f"[stage17] loaded completed forecasts: {spec_name}", flush=True)
            for key, stem in (
                ("forecasts", "forecast_ledger"),
                ("coefficients", "coefficients"),
                ("edges", "edges"),
                ("tuning", "tuning"),
                ("diagnostics", "diagnostics"),
            ):
                path = run_dir / f"{stem}_{spec_name}.parquet"
                result_frames[key].append(pd.read_parquet(path) if path.exists() else pd.DataFrame())
            continue
        print(f"[stage17] target-free issuance: {spec_name}", flush=True)
        issued = issue_factor_har_forecasts(
            vintage["benchmark_log"],
            vintage["aggregate_log"],
            vintage["local_log"],
            calendar,
            spec_name=spec_name,
            config=config,
            run_id=run_id,
            factor_metadata=vintage["factor_metadata"],
            har_window=har_window,
            mode=mode,
            freeze_timestamp=freeze_timestamp,
        )
        for key, stem, frame in (
            ("forecasts", "forecast_ledger", issued.forecasts),
            ("coefficients", "coefficients", issued.coefficients),
            ("edges", "edges", issued.edge_history),
            ("tuning", "tuning", issued.tuning_history),
            ("diagnostics", "diagnostics", issued.origin_diagnostics),
        ):
            result_frames[key].append(frame)
            frame.to_parquet(run_dir / f"{stem}_{spec_name}.parquet", index=False)
    combined = {
        key: (
            pd.concat(frames, ignore_index=True)
            if frames and any(not frame.empty for frame in frames)
            else pd.DataFrame()
        )
        for key, frames in result_frames.items()
    }
    forecasts = (
        combined["forecasts"]
        if not combined["forecasts"].empty
        else _empty_forecast_ledger()
    )
    return (
        ForecastRunResult(
            forecasts=forecasts,
            coefficients=combined["coefficients"],
            edge_history=combined["edges"],
            tuning_history=combined["tuning"],
            origin_diagnostics=combined["diagnostics"],
            status="complete" if mode == "historical_pseudo_oos" else (
                "issued_pending_outcomes" if not forecasts.empty else "awaiting_future_data"
            ),
        ),
        realized,
        metadata,
    )


def run_audit(*, config: OOSConfig, smoke: bool, run_id: str | None, resume: bool) -> Path:
    run_id = run_id or _run_id("smoke" if smoke else "historical_audit")
    run_dir = _run_directory(run_id)
    complete, calendar = _load_inputs(smoke)
    manifest = _manifest(
        config,
        mode="historical_pseudo_oos",
        freeze=utc_now_iso(),
        status="running",
    )
    manifest["run_id"] = run_id
    write_json(run_dir / "manifest.json", manifest)
    result, realized, metadata = _fit_specs(
        complete, calendar, config, run_dir, run_id=run_id,
        mode="historical_pseudo_oos", resume=resume,
    )
    scored = score_forecasts(
        result.forecasts, realized, metadata, config=config, models=FACTOR_HAR_MODELS
    )
    comparisons = matched_qlike_comparisons(scored)
    pairs = list(comparisons[["model_a", "model_b"]].drop_duplicates().itertuples(index=False, name=None))
    model_summary = summarize_models(scored)
    hac = build_hac_comparisons(scored, pairs)
    manifest["status"] = "historical_pseudo_oos_complete"
    manifest["n_forecast_rows"] = len(result.forecasts)
    manifest["n_scored_rows"] = int((scored["score_status"] == "scored").sum())
    write_json(run_dir / "manifest.json", manifest)
    for name, frame in (
        ("forecast_ledger.csv", result.forecasts),
        ("score_ledger.csv", scored),
        ("model_summary.csv", model_summary),
        ("matched_comparisons.csv", comparisons),
        ("hac_comparisons.csv", hac),
        ("coefficients.csv", result.coefficients),
        ("edges.csv", result.edge_history),
        ("tuning.csv", result.tuning_history),
        ("diagnostics.csv", result.origin_diagnostics),
    ):
        _write_frame(run_dir / name, frame)
    write_results_markdown(
        run_dir / "FACTOR_HAR_RESULTS.md",
        manifest=manifest,
        model_summary=model_summary,
        comparisons=comparisons,
        hac=hac,
    )
    print(f"[stage17] completed: {run_dir}", flush=True)
    return run_dir


def run_freeze(*, config: OOSConfig) -> Path:
    run_id = _run_id("protocol_freeze")
    run_dir = _run_directory(run_id)
    manifest = _manifest(
        config,
        mode="protocol_freeze",
        freeze=utc_now_iso(),
        status="ready / awaiting_future_data",
    )
    manifest["run_id"] = run_id
    write_json(run_dir / "manifest.json", manifest)
    write_json(OUT_DIR / "latest_protocol_manifest.json", manifest)
    print(f"[stage17] protocol frozen: {run_dir}", flush=True)
    return run_dir


def run_prospective(*, config: OOSConfig) -> Path:
    frozen_path = OUT_DIR / "latest_protocol_manifest.json"
    if not frozen_path.exists():
        raise FileNotFoundError("Run --mode freeze before prospective issuance.")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("code_provenance") != _code_provenance():
        raise ValueError("Stage 17 code changed after protocol freeze; freeze again.")
    frozen_config = deserialize_config(frozen["configuration"])
    config = replace(
        frozen_config,
        har=replace(frozen_config.har, n_jobs=config.har.n_jobs),
    )
    freeze = str(frozen["freeze_timestamp_utc"])
    run_id = _run_id("prospective")
    run_dir = _run_directory(run_id)
    complete, calendar = _load_inputs(False)
    result, _, _ = _fit_specs(
        complete, calendar, config, run_dir, run_id=run_id,
        mode="prospective", freeze_timestamp=freeze,
    )
    manifest = dict(frozen)
    manifest.update(
        run_id=run_id,
        mode="prospective",
        status=result.status,
        issued_at_utc=utc_now_iso(),
        n_forecast_rows=len(result.forecasts),
    )
    _write_frame(run_dir / "forecast_ledger.csv", result.forecasts)
    manifest["forecast_ledger_sha256"] = file_sha256(run_dir / "forecast_ledger.csv")
    write_json(run_dir / "manifest.json", manifest)
    print(f"[stage17] prospective state: {result.status}; {run_dir}", flush=True)
    return run_dir


def run_score(*, config: OOSConfig, run_id: str) -> Path:
    run_dir = RUNS_DIR / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Unknown Stage 17 run: {run_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("code_provenance") != _code_provenance():
        raise ValueError(
            "Stage 17 code changed after forecast issuance; "
            "do not score this ledger with a different protocol."
        )
    ledger_path = run_dir / "forecast_ledger.csv"
    if file_sha256(ledger_path) != manifest.get("forecast_ledger_sha256"):
        raise ValueError("The immutable prospective forecast ledger hash changed.")
    forecasts = pd.read_csv(ledger_path)
    validate_forecast_ledger(forecasts)
    frozen_config = deserialize_config(manifest["configuration"])
    config = replace(
        frozen_config,
        har=replace(frozen_config.har, n_jobs=config.har.n_jobs),
    )
    complete, calendar = _load_inputs(False)
    realized: dict[str, pd.DataFrame] = {}
    metadata: dict[str, pd.DataFrame] = {}
    vintages: dict[int, FactorVintageRun] = {}
    for window in sorted({window for window, _, _ in _specifications(config)}):
        print(f"[stage17] rebuilding causal outcomes: {window} sessions", flush=True)
        vintages[window] = build_factor_vintage_run(
            complete,
            calendar,
            factor_window_sessions=window,
            config=config,
            spec_name=f"factor_{window}",
        )
    for window, _, spec_name in _specifications(config):
        realized[spec_name] = vintages[window].benchmark_daily_variance
        metadata[spec_name] = vintages[window].rv_metadata.assign(spec_name=spec_name)
    scored = score_forecasts(
        forecasts, realized, metadata, config=config, models=FACTOR_HAR_MODELS
    )
    comparisons = matched_qlike_comparisons(scored)
    pairs = list(comparisons[["model_a", "model_b"]].drop_duplicates().itertuples(index=False, name=None))
    summary = summarize_models(scored)
    hac = build_hac_comparisons(scored, pairs)
    n_scored = int((scored["score_status"] == "scored").sum())
    manifest["status"] = "confirmation_in_progress" if n_scored else "awaiting_future_data"
    manifest["n_scored_rows"] = n_scored
    manifest["scored_at_utc"] = utc_now_iso()
    write_json(manifest_path, manifest)
    for name, frame in (
        ("score_ledger.csv", scored), ("model_summary.csv", summary),
        ("matched_comparisons.csv", comparisons), ("hac_comparisons.csv", hac),
    ):
        _write_frame(run_dir / name, frame)
    write_results_markdown(
        run_dir / "FACTOR_HAR_RESULTS.md",
        manifest=manifest,
        model_summary=summary,
        comparisons=comparisons,
        hac=hac,
    )
    return run_dir


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "smoke", "freeze", "prospective", "score"), required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    ensure_project_directories()
    config = _effective_config(n_jobs=args.n_jobs, smoke=args.mode == "smoke")
    if args.mode == "freeze":
        run_freeze(config=config)
    elif args.mode == "prospective":
        run_prospective(config=config)
    elif args.mode == "score":
        if not args.run_id:
            parser.error("--mode score requires --run-id")
        run_score(config=config, run_id=args.run_id)
    else:
        run_audit(
            config=config,
            smoke=args.mode == "smoke",
            run_id=args.run_id,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
