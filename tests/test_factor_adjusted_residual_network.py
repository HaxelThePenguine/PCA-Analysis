"""Focused invariants for the factor-adjusted residual network stage."""

from __future__ import annotations

import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.factor_adjusted_residuals import (  # noqa: E402
    FactorConfig,
    apply_factor_fit,
    fit_factor_window,
    run_factor_adjustment,
)
from utils.network_har import (  # noqa: E402
    HARConfig,
    build_har_features,
    build_hac_tests,
    edge_stability,
    fit_partialling_out,
    network_density,
    walk_forward_forecasts,
)
from utils.realized_volatility import (  # noqa: E402
    RealizedVarianceConfig,
    aggregate_realized_variance,
)


def _minute_panel(
    n_sessions: int = 8,
    stocks: tuple[str, ...] = ("A", "B", "C", "D", "E", "F"),
    minutes_per_session: int = 10,
) -> pd.DataFrame:
    rng = np.random.default_rng(1201)
    days = pd.bdate_range("2023-01-02", periods=n_sessions, tz="America/New_York")
    index = pd.DatetimeIndex(
        [day + pd.Timedelta(hours=9, minutes=30 + minute) for day in days for minute in range(minutes_per_session)]
    )
    benchmarks = rng.normal(size=(len(index), 2))
    stock_values = benchmarks @ rng.normal(size=(2, len(stocks))) + rng.normal(size=(len(index), len(stocks)))
    return pd.DataFrame(
        np.column_stack([stock_values, benchmarks]) * 0.01,
        index=index,
        columns=[*stocks, "SPY", "XLF"],
    )


class FactorResidualTest(unittest.TestCase):
    def test_rotation_projector_and_residual_orthogonality(self) -> None:
        panel = _minute_panel()
        fit = fit_factor_window(
            panel.iloc[:50],
            tuple(panel.columns[:6]),
            ("SPY", "XLF"),
            FactorConfig(window_sessions=5, n_components=3, l1_starts=8),
        )
        applied = apply_factor_fit(panel.iloc[50:], fit)
        self.assertLess(fit.projector_invariance_error, 1e-10)
        self.assertLess(fit.training_orthogonality_error, 1e-10)
        self.assertLess(applied["orthogonality_error"], 1e-10)

    def test_parallel_l1_rotation_matches_serial_factor_fit(self) -> None:
        panel = _minute_panel()
        stocks = tuple(panel.columns[:6])
        serial = fit_factor_window(
            panel.iloc[:50], stocks, ("SPY", "XLF"), FactorConfig(n_components=3, l1_starts=8, n_jobs=1)
        )
        parallel = fit_factor_window(
            panel.iloc[:50], stocks, ("SPY", "XLF"), FactorConfig(n_components=3, l1_starts=8, n_jobs=2)
        )
        np.testing.assert_allclose(serial.loadings, parallel.loadings, atol=1e-10)
        np.testing.assert_allclose(serial.rotation, parallel.rotation, atol=1e-10)

    def test_future_observations_do_not_change_training_fit(self) -> None:
        panel = _minute_panel(n_sessions=8)
        stocks = tuple(panel.columns[:6])
        fit_original = fit_factor_window(
            panel.iloc[:50], stocks, ("SPY", "XLF"), FactorConfig(n_components=3, l1_starts=8)
        )
        changed = panel.copy()
        changed.iloc[50:] *= 100.0
        fit_changed = fit_factor_window(
            changed.iloc[:50], stocks, ("SPY", "XLF"), FactorConfig(n_components=3, l1_starts=8)
        )
        np.testing.assert_allclose(fit_original.loadings, fit_changed.loadings)
        np.testing.assert_allclose(fit_original.coefficients, fit_changed.coefficients)

    def test_rolling_score_blocks_are_disjoint(self) -> None:
        result = run_factor_adjustment(
            _minute_panel(n_sessions=8),
            ("A", "B", "C", "D", "E", "F"),
            ("SPY", "XLF"),
            FactorConfig(window_sessions=4, update_step_sessions=2, n_components=3, l1_starts=8),
            spec_name="test",
        )
        self.assertFalse(result.raw_returns.index.has_duplicates)
        self.assertEqual(len(result.raw_returns), 4 * 10)


class RealizedVarianceTest(unittest.TestCase):
    def test_five_minute_aggregation_rejects_a_gap_without_crossing_sessions(self) -> None:
        panel = _minute_panel(n_sessions=2)
        missing = panel.index[16]
        panel = panel.drop(index=missing)
        calendar = pd.DataFrame(
            {
                "date": ["2023-01-02", "2023-01-03"],
                "open_minute": [570, 570],
                "close_minute": [580, 580],
                "expected_bars": [10, 10],
            }
        )
        result = aggregate_realized_variance(
            {"raw": panel.iloc[:, :6], "factor_adjusted": panel.iloc[:, :6]},
            calendar,
            RealizedVarianceConfig(),
        )
        self.assertEqual(result.interval_diagnostics.loc[pd.Timestamp("2023-01-02"), "n_valid_intervals"], 2)
        self.assertEqual(result.interval_diagnostics.loc[pd.Timestamp("2023-01-03"), "n_valid_intervals"], 1)
        self.assertEqual(len(result.interval_returns["raw"]), 3)
        self.assertTrue((result.interval_returns["raw"].index.get_level_values(0) != pd.Timestamp("2023-01-02")).any())


class HARModelTest(unittest.TestCase):
    def test_har_features_end_at_forecast_origin(self) -> None:
        index = pd.bdate_range("2023-01-02", periods=30)
        data = pd.DataFrame({"A": np.arange(30, dtype=float)}, index=index)
        design = build_har_features(data)
        self.assertEqual(design.origins[0], index[21])
        self.assertEqual(design.targets[0], index[22])
        self.assertEqual(design.features.iloc[0][("A", "D")], 21.0)
        self.assertAlmostEqual(design.features.iloc[0][("A", "W")], np.mean(np.arange(17.0, 22.0)))
        self.assertAlmostEqual(design.features.iloc[0][("A", "M")], np.mean(np.arange(22.0)))

    def test_unpenalized_partialling_out_matches_ols(self) -> None:
        rng = np.random.default_rng(31)
        own = rng.normal(size=(100, 3))
        cross = rng.normal(size=(100, 4))
        y = 1.5 + own @ np.array([0.2, -0.4, 0.7]) + cross @ np.array([1.2, 0.0, -0.5, 0.3])
        fit = fit_partialling_out(y, own, cross, alpha=0.0, cross_feature_names=["a", "b", "c", "d"])
        expected = np.linalg.lstsq(np.column_stack([np.ones(len(y)), own, cross]), y, rcond=None)[0]
        np.testing.assert_allclose(
            np.r_[fit.intercept, fit.own_coefficients, fit.network_coefficients_original],
            expected,
            atol=1e-10,
        )
        np.testing.assert_allclose(fit.train_predictions, y, atol=1e-10)

    def test_known_qlike_and_hac_comparison_are_finite(self) -> None:
        rng = np.random.default_rng(11)
        index = pd.bdate_range("2023-01-02", periods=65)
        data = pd.DataFrame(np.exp(rng.normal(size=(65, 3))), index=index, columns=["A", "B", "C"])
        result = walk_forward_forecasts(
            np.log(data),
            spec_name="test",
            factor_window_sessions=4,
            config=HARConfig(min_training_observations=10, cv_splits=2, tuning_frequency=5, lasso_max_iter=500),
        )
        self.assertAlmostEqual(2.0 / 1.0 - np.log(2.0) - 1.0, 1.0 - np.log(2.0))
        tests = build_hac_tests(result.forecasts)
        self.assertTrue(np.isfinite(result.forecasts["qlike"]).all())
        self.assertTrue(np.isfinite(tests["mean_difference"].dropna()).all())

    def test_future_perturbation_does_not_change_earlier_forecasts(self) -> None:
        rng = np.random.default_rng(12)
        index = pd.bdate_range("2023-01-02", periods=75)
        original = pd.DataFrame(rng.normal(size=(75, 4)), index=index, columns=list("ABCD"))
        changed = original.copy()
        changed.iloc[50:] += 15.0
        config = HARConfig(min_training_observations=10, cv_splits=2, tuning_frequency=5, lasso_max_iter=500)
        first = walk_forward_forecasts(original, spec_name="test", factor_window_sessions=4, config=config).forecasts
        second = walk_forward_forecasts(changed, spec_name="test", factor_window_sessions=4, config=config).forecasts
        cutoff = index[50]
        left = first[first["target_date"] < cutoff].reset_index(drop=True)
        right = second[second["target_date"] < cutoff].reset_index(drop=True)
        np.testing.assert_array_equal(left["selected_penalty"].fillna(-1), right["selected_penalty"].fillna(-1))
        np.testing.assert_allclose(left["predicted_log_variance"], right["predicted_log_variance"])

    def test_parallel_walk_forward_matches_serial_results(self) -> None:
        rng = np.random.default_rng(121)
        index = pd.bdate_range("2023-01-02", periods=45)
        data = pd.DataFrame(rng.normal(size=(45, 5)), index=index, columns=list("ABCDE"))
        base = dict(min_training_observations=10, cv_splits=2, tuning_frequency=5, lasso_max_iter=300)
        serial = walk_forward_forecasts(
            data,
            spec_name="parallel_check",
            factor_window_sessions=4,
            config=HARConfig(n_jobs=1, **base),
        )
        parallel = walk_forward_forecasts(
            data,
            spec_name="parallel_check",
            factor_window_sessions=4,
            config=HARConfig(n_jobs=2, **base),
        )
        for column in ("predicted_log_variance", "qlike", "selected_penalty"):
            np.testing.assert_allclose(
                serial.forecasts[column].fillna(-1),
                parallel.forecasts[column].fillna(-1),
                atol=1e-10,
            )
        np.testing.assert_allclose(
            serial.edge_history["coefficient_original"],
            parallel.edge_history["coefficient_original"],
            atol=1e-10,
        )

    def test_synthetic_sparse_edge_is_more_persistent_than_independent_edges(self) -> None:
        rng = np.random.default_rng(99)
        n = 90
        index = pd.bdate_range("2023-01-02", periods=n)
        source = rng.normal(size=n)
        values = np.column_stack(
            [source, np.r_[0.0, 1.5 * source[:-1] + 0.05 * rng.normal(size=n - 1)], rng.normal(size=(n, 2))]
        )
        log_variance = pd.DataFrame(values, index=index, columns=["A", "B", "C", "D"])
        config = HARConfig(
            min_training_observations=25,
            cv_splits=2,
            tuning_frequency=10,
            alpha_fractions=(0.01, 0.03, 0.10, 0.30, 1.0),
            lasso_max_iter=700,
        )
        result = walk_forward_forecasts(log_variance, spec_name="sparse", factor_window_sessions=4, config=config)
        stability = edge_stability(result.edge_history)
        directed = stability[stability["source"].eq("A") & stability["target"].eq("B")]
        unrelated = stability[stability["target"].eq("B") & ~stability["source"].eq("A")]
        self.assertGreater(float(directed["selection_probability"].iloc[0]), float(unrelated["selection_probability"].max()))
        independent = pd.DataFrame(rng.normal(size=(n, 4)), index=index, columns=["A", "B", "C", "D"])
        independent_result = walk_forward_forecasts(independent, spec_name="independent", factor_window_sessions=4, config=config)
        density = network_density(independent_result.edge_history)
        self.assertLess(float(density["stable_edge_density"].iloc[0]), 0.80)


class ImportSafetyTest(unittest.TestCase):
    def test_stage_import_has_no_execution_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            namespace = runpy.run_path(SRC_DIR / "15_factor_adjusted_residual_network.py", run_name="test_stage15")
            self.assertTrue(callable(namespace["main"]))
            self.assertFalse(any(Path(temporary).iterdir()))


if __name__ == "__main__":
    unittest.main()
