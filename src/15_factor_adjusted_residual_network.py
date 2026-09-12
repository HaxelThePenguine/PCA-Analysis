"""Stage 15: factor-adjusted realized-volatility network HAR analysis."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from config import (
    BENCHMARKS,
    CALENDAR_FILE,
    CORE_UNIVERSE,
    REPORTS_DIR,
    RETURN_MATRIX_CLEAN_FILE,
    ensure_project_directories,
)
from reporting.network import make_figures as render_figures, print_summary
from utils.data import load_panel, validate_panel
from utils.factor_adjusted_residuals import FactorConfig, run_factor_adjustment
from utils.network_har import (
    HARConfig,
    bootstrap_edge_selection,
    build_hac_tests,
    descriptive_network_edges,
    edge_stability,
    group_connectivity,
    network_centrality,
    network_density,
    summarize_forecasts,
    walk_forward_forecasts,
)
from utils.realized_volatility import (
    RealizedVarianceConfig,
    aggregate_realized_variance,
)


OUT_DIR = REPORTS_DIR / "factor_adjusted_residual_network"
PRIMARY_K = 3
FACTOR_WINDOWS = (120, 60)
GROUPS = {
    "MS_C": ("MS", "C"),
    "large_banks": ("JPM", "BAC", "WFC", "C"),
    "regional_banks": ("USB", "TFC", "KEY", "RF", "FITB", "CFG", "HBAN"),
}


@dataclass(frozen=True)
class Stage15Config:
    """Small immutable configuration object for deterministic production runs."""

    factor_windows: tuple[int, ...] = FACTOR_WINDOWS
    factor_update_step: int = 5
    factor_components: int = PRIMARY_K
    factor_l1_starts: int = 40
    factor_seed: int = 15015
    realized_variance: RealizedVarianceConfig = field(default_factory=RealizedVarianceConfig)
    har: HARConfig = field(default_factory=HARConfig)


def _panel(panel: pd.DataFrame | None) -> pd.DataFrame:
    if panel is None:
        return load_panel(
            RETURN_MATRIX_CLEAN_FILE,
            [*CORE_UNIVERSE, *BENCHMARKS],
            context="Factor-adjusted residual network panel",
            drop_incomplete=True,
        )
    required = [*CORE_UNIVERSE, *BENCHMARKS]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise ValueError(f"Factor-adjusted network panel is missing columns: {missing}")
    value = panel.loc[:, required].copy()
    validate_panel(value, context="Factor-adjusted residual network panel", require_complete=True)
    return value


def _long_variance(
    variance_panels: dict[int, Any],
    attribute: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for window, result in variance_panels.items():
        for control, frame in getattr(result, attribute).items():
            long = frame.rename_axis("session_date").stack().rename(attribute).reset_index()
            long.columns = ["session_date", "stock", attribute]
            long.insert(0, "factor_window_sessions", window)
            long.insert(1, "control", control)
            rows.append(long)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _calendar_period_summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame()
    result = forecasts.copy()
    dates = pd.to_datetime(result["target_date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    result["calendar_period"] = dates.dt.to_period("Q").astype(str)
    return summarize_forecasts(result, ("spec_name", "model", "calendar_period"))


def _stress_summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame(
            [{"stress_period": "2023-03-01_to_2023-05-31", "available": False, "n_forecasts": 0}]
        )
    dates = pd.to_datetime(forecasts["target_date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    selected = forecasts[(dates >= pd.Timestamp("2023-03-01")) & (dates <= pd.Timestamp("2023-05-31"))]
    if selected.empty:
        return pd.DataFrame(
            [{"stress_period": "2023-03-01_to_2023-05-31", "available": False, "n_forecasts": 0}]
        )
    result = summarize_forecasts(selected, ("spec_name", "model"))
    result.insert(0, "stress_period", "2023-03-01_to_2023-05-31")
    result.insert(1, "available", True)
    return result


def _forecast_comparison(forecast_summary: pd.DataFrame) -> pd.DataFrame:
    if forecast_summary.empty:
        return pd.DataFrame()
    keys = ["spec_name", "factor_window_sessions", "har_window"]
    own = forecast_summary[forecast_summary["model"] == "own_har"].rename(
        columns={
            "qlike": "own_qlike",
            "log_mse": "own_log_mse",
            "log_mae": "own_log_mae",
            "directional_accuracy": "own_directional_accuracy",
        }
    )
    network = forecast_summary[forecast_summary["model"] == "network_har_l1_1se"].rename(
        columns={
            "qlike": "network_qlike",
            "log_mse": "network_log_mse",
            "log_mae": "network_log_mae",
            "directional_accuracy": "network_directional_accuracy",
        }
    )
    own_columns = keys + [
        "own_qlike",
        "own_log_mse",
        "own_log_mae",
        "own_directional_accuracy",
    ]
    network_columns = keys + [
        "network_qlike",
        "network_log_mse",
        "network_log_mae",
        "network_directional_accuracy",
    ]
    result = own[own_columns].merge(
        network[network_columns], on=keys, how="inner", suffixes=("", "_duplicate")
    )
    result["qlike_improvement_pct"] = 100.0 * (result["own_qlike"] - result["network_qlike"]) / result["own_qlike"]
    result["log_mse_improvement_pct"] = 100.0 * (result["own_log_mse"] - result["network_log_mse"]) / result["own_log_mse"]
    return result


def _write_outputs(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(out_dir / filename, index=False)


def run_analysis(
    panel: pd.DataFrame | None = None,
    *,
    calendar: pd.DataFrame | None = None,
    config: Stage15Config = Stage15Config(),
    make_figures: bool = True,
    write_files: bool = True,
    run_bootstrap: bool = True,
    out_dir: Path = OUT_DIR,
) -> dict[str, Any]:
    """Run stage 15 and optionally write its separate report directory."""

    stage_started = time.perf_counter()
    data = _panel(panel)
    print(f"[stage15] panel loaded: {len(data):,} rows", flush=True)
    market_calendar = pd.read_csv(CALENDAR_FILE) if calendar is None else calendar.copy()
    factor_results: dict[int, Any] = {}
    variance_results: dict[int, Any] = {}
    factor_diagnostics: list[pd.DataFrame] = []
    factor_metrics: list[pd.DataFrame] = []
    factor_loadings: list[pd.DataFrame] = []
    benchmark_coefficients: list[pd.DataFrame] = []
    interval_diagnostics: list[pd.DataFrame] = []
    floors: list[pd.DataFrame] = []
    daily_logs: dict[tuple[int, str], pd.DataFrame] = {}
    for window in config.factor_windows:
        print(
            f"[stage15] factor adjustment {window} sessions started with {config.har.n_jobs} workers",
            flush=True,
        )
        factor_config = FactorConfig(
            n_jobs=config.har.n_jobs,
            window_sessions=int(window),
            update_step_sessions=config.factor_update_step,
            n_components=config.factor_components,
            l1_starts=config.factor_l1_starts,
            random_seed=config.factor_seed + int(window),
        )
        factor_result = run_factor_adjustment(
            data,
            CORE_UNIVERSE,
            BENCHMARKS,
            factor_config,
            spec_name=f"factor_adjusted_{window}",
        )
        variance_result = aggregate_realized_variance(
            {
                "raw": factor_result.raw_returns,
                "benchmark_residual": factor_result.benchmark_residuals,
                "factor_adjusted": factor_result.factor_adjusted_residuals,
            },
            market_calendar,
            config.realized_variance,
        )
        factor_results[int(window)] = factor_result
        variance_results[int(window)] = variance_result
        for control, frame in variance_result.log_daily_ivar.items():
            daily_logs[(int(window), control)] = frame
        for table, destination in (
            (factor_result.diagnostics, factor_diagnostics),
            (factor_result.factor_metrics, factor_metrics),
            (factor_result.factor_loadings, factor_loadings),
            (factor_result.benchmark_coefficients, benchmark_coefficients),
        ):
            copy = table.copy()
            copy.insert(0, "factor_window_sessions", int(window))
            destination.append(copy)
        interval = variance_result.interval_diagnostics.reset_index()
        interval.insert(0, "factor_window_sessions", int(window))
        interval_diagnostics.append(interval)
        floor = variance_result.floor_diagnostics.copy()
        floor.insert(0, "factor_window_sessions", int(window))
        floors.append(floor)

    primary_window = int(config.factor_windows[0])
    robustness_window = int(config.factor_windows[1] if len(config.factor_windows) > 1 else config.factor_windows[0])
    forecast_specs = (
        (f"factor_adjusted_{primary_window}_expanding", primary_window, "expanding"),
        (f"factor_adjusted_{robustness_window}_expanding", robustness_window, "expanding"),
        (f"factor_adjusted_{primary_window}_rolling252", primary_window, config.har.rolling_training_observations),
    )
    forecast_results: list[Any] = []
    for spec_name, window, har_window in forecast_specs:
        print(f"[stage15] forecast {spec_name} started", flush=True)
        result = walk_forward_forecasts(
            daily_logs[(window, "factor_adjusted")],
            spec_name=spec_name,
            factor_window_sessions=window,
            har_window=har_window,
            config=config.har,
        )
        forecast_results.append(result)
    forecasts = pd.concat([result.forecasts for result in forecast_results], ignore_index=True)
    har_coefficients = pd.concat([result.coefficients for result in forecast_results], ignore_index=True)
    edge_history = pd.concat([result.edge_history for result in forecast_results], ignore_index=True)
    tuning_history = pd.concat([result.tuning_history for result in forecast_results], ignore_index=True)
    forecast_summary = summarize_forecasts(
        forecasts,
        ("spec_name", "factor_window_sessions", "har_window", "model"),
    )
    stock_summary = summarize_forecasts(
        forecasts,
        ("spec_name", "factor_window_sessions", "har_window", "model", "stock"),
    )
    period_summary = _calendar_period_summary(forecasts)
    stress_summary = _stress_summary(forecasts)
    hac_tests = build_hac_tests(forecasts, max_lag=config.har.hac_lag)
    edge_stability_table = edge_stability(edge_history, model="network_har_l1_1se")
    edge_density = network_density(edge_history)
    centrality = network_centrality(edge_stability_table)
    forecast_groups = group_connectivity(edge_stability_table, GROUPS)

    descriptive_edges: list[pd.DataFrame] = []
    descriptive_tuning: list[pd.DataFrame] = []
    for window in config.factor_windows:
        for control in ("raw", "benchmark_residual", "factor_adjusted"):
            spec_name = f"descriptive_{control}_{window}"
            print(f"[stage15] descriptive network {spec_name} started", flush=True)
            edges, tuning = descriptive_network_edges(
                daily_logs[(int(window), control)],
                spec_name=spec_name,
                factor_window_sessions=int(window),
                har_window="expanding",
                config=config.har,
            )
            descriptive_edges.append(edges)
            descriptive_tuning.append(tuning)
    descriptive_edge_history = pd.concat(descriptive_edges, ignore_index=True)
    descriptive_tuning_history = pd.concat(descriptive_tuning, ignore_index=True)
    descriptive_stability = edge_stability(
        descriptive_edge_history,
        model="descriptive_network",
    )
    descriptive_density = network_density(descriptive_edge_history)
    descriptive_groups = group_connectivity(descriptive_stability, GROUPS)
    bootstrap = []
    if run_bootstrap:
        for window in config.factor_windows:
            print(f"[stage15] bootstrap {window} sessions started", flush=True)
            bootstrap.append(
                bootstrap_edge_selection(
                    daily_logs[(int(window), "factor_adjusted")],
                    spec_name=f"factor_adjusted_{window}_expanding",
                    factor_window_sessions=int(window),
                    har_window="expanding",
                    config=config.har,
                )
            )
    bootstrap_edges = pd.concat(bootstrap, ignore_index=True) if bootstrap else pd.DataFrame()

    factor_diagnostics_table = pd.concat(factor_diagnostics, ignore_index=True)
    factor_metrics_table = pd.concat(factor_metrics, ignore_index=True)
    factor_loadings_table = pd.concat(factor_loadings, ignore_index=True)
    benchmark_coefficients_table = pd.concat(benchmark_coefficients, ignore_index=True)
    interval_diagnostics_table = pd.concat(interval_diagnostics, ignore_index=True)
    floor_table = pd.concat(floors, ignore_index=True)
    daily_variance_table = _long_variance(variance_results, "daily_ivar")
    daily_log_table = _long_variance(variance_results, "log_daily_ivar")
    comparison = _forecast_comparison(forecast_summary)
    tables = {
        "15_daily_realized_variance.csv": daily_variance_table,
        "15_daily_log_realized_variance.csv": daily_log_table,
        "15_interval_diagnostics.csv": interval_diagnostics_table,
        "15_variance_floor_diagnostics.csv": floor_table,
        "15_factor_model_diagnostics.csv": factor_diagnostics_table,
        "15_factor_metrics.csv": factor_metrics_table,
        "15_factor_loadings.csv": factor_loadings_table,
        "15_benchmark_coefficients.csv": benchmark_coefficients_table,
        "15_walk_forward_forecasts.csv": forecasts,
        "15_forecast_summary.csv": forecast_summary,
        "15_stock_forecast_summary.csv": stock_summary,
        "15_calendar_period_summary.csv": period_summary,
        "15_stress_period_summary.csv": stress_summary,
        "15_hac_tests.csv": hac_tests,
        "15_har_coefficients.csv": har_coefficients,
        "15_edge_history.csv": edge_history,
        "15_edge_stability.csv": edge_stability_table,
        "15_network_density.csv": edge_density,
        "15_network_centrality.csv": centrality,
        "15_group_connectivity.csv": forecast_groups,
        "15_descriptive_edge_history.csv": descriptive_edge_history,
        "15_descriptive_tuning_history.csv": descriptive_tuning_history,
        "15_descriptive_edge_stability.csv": descriptive_stability,
        "15_descriptive_network_density.csv": descriptive_density,
        "15_descriptive_group_connectivity.csv": descriptive_groups,
        "15_bootstrap_edge_selection.csv": bootstrap_edges,
        "15_tuning_history.csv": tuning_history,
        "15_factor_window_comparison.csv": comparison,
    }
    if write_files:
        _write_outputs(tables, out_dir)
    if make_figures:
        render_figures(
            forecasts,
            edge_history,
            edge_stability_table,
            pd.concat([forecast_groups, descriptive_groups], ignore_index=True),
            factor_diagnostics_table,
            factor_loadings_table,
            out_dir=out_dir,
        )
    print(
        f"[stage15] completed in {time.perf_counter() - stage_started:.1f}s",
        flush=True,
    )
    return {
        "factor_results": factor_results,
        "variance_results": variance_results,
        "factor_diagnostics": factor_diagnostics_table,
        "factor_metrics": factor_metrics_table,
        "factor_loadings": factor_loadings_table,
        "benchmark_coefficients": benchmark_coefficients_table,
        "daily_variance": daily_variance_table,
        "daily_log_variance": daily_log_table,
        "interval_diagnostics": interval_diagnostics_table,
        "floor_diagnostics": floor_table,
        "forecasts": forecasts,
        "forecast_summary": forecast_summary,
        "stock_summary": stock_summary,
        "period_summary": period_summary,
        "stress_summary": stress_summary,
        "hac_tests": hac_tests,
        "har_coefficients": har_coefficients,
        "edge_history": edge_history,
        "edge_stability": edge_stability_table,
        "network_density": edge_density,
        "centrality": centrality,
        "group_connectivity": forecast_groups,
        "descriptive_edge_history": descriptive_edge_history,
        "descriptive_tuning_history": descriptive_tuning_history,
        "descriptive_edge_stability": descriptive_stability,
        "descriptive_network_density": descriptive_density,
        "descriptive_group_connectivity": descriptive_groups,
        "bootstrap_edges": bootstrap_edges,
        "tuning_history": tuning_history,
        "comparison": comparison,
    }


def main(argv: Iterable[str] | None = None) -> None:
    """Run stage 15 on the configured local data."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of worker processes used for parallel HAR target equations.",
    )
    args = parser.parse_args(argv)
    if args.n_jobs < 1:
        parser.error("--n-jobs must be at least one")
    ensure_project_directories()
    base_config = Stage15Config()
    config = replace(
        base_config,
        har=replace(base_config.har, n_jobs=args.n_jobs),
    )
    results = run_analysis(config=config)
    print_summary(
        results["forecast_summary"],
        results["hac_tests"],
        results["edge_stability"],
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
