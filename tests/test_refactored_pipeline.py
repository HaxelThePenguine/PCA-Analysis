"""Contracts at the boundary between research stages and reusable calculations."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from config import BENCHMARKS, CORE_UNIVERSE
from utils.data import normalize_intraday_volatility
from utils.kalman import fit_window, make_windows
from utils.l1_rotation import fit_l1_rotation
from utils.local_factor import (
    bootstrap_correlation,
    bootstrap_local_factors,
    build_session_moments,
)
from utils.pca import fit_pca, pca_diagnostics, reconstruction_by_stock
from utils.preprocessing import build_contaminated_mask, compute_intraday_returns
from utils.rolling_pca import collect_rolling_results
from utils.variance import build_variance_ledger


def sample_panel(n_sessions=65, stocks=CORE_UNIVERSE):
    generator = np.random.default_rng(117)
    days = pd.bdate_range("2023-01-02", periods=n_sessions, tz="America/New_York")
    index = pd.DatetimeIndex(
        [
            day + pd.Timedelta(hours=9, minutes=31 + minute)
            for day in days
            for minute in range(4)
        ]
    )
    benchmarks = generator.normal(size=(len(index), 2))
    values = benchmarks @ generator.normal(size=(2, len(stocks)))
    values += generator.normal(size=values.shape)
    return pd.DataFrame(
        np.column_stack([values, benchmarks]) * 0.001,
        index=index,
        columns=[*stocks, *BENCHMARKS],
    )


def load_stage(filename):
    name = "regression_" + filename.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class NumericalBoundaryTest(unittest.TestCase):
    def test_reconstruction_ledger_matches_pca_variance_share(self):
        panel = sample_panel().loc[:, list(CORE_UNIVERSE)]
        for method in ("covariance", "correlation"):
            result = fit_pca(panel, method)
            matrix_error, score_error = pca_diagnostics(result)
            self.assertLess(matrix_error, 1e-12)
            self.assertLess(score_error, 1e-12)
            ledger = reconstruction_by_stock(result, 3)
            retained = (
                1 - ledger.residual_variance.sum() / ledger.original_variance.sum()
            )
            self.assertAlmostEqual(retained, result.explained[:3].sum())
            full = reconstruction_by_stock(result, panel.shape[1])
            np.testing.assert_allclose(full.explained_pct, 100, atol=1e-10)

    def test_variance_ledger_reconciles_benchmarks_and_components(self):
        ledger = build_variance_ledger(sample_panel(), CORE_UNIVERSE)["ledger"]
        np.testing.assert_allclose(ledger.total_pct, 100, atol=1e-10)
        np.testing.assert_allclose(ledger.variance_identity_error, 0, atol=1e-12)

    def test_rolling_uses_requested_stocks_step_and_last_window(self):
        stocks = ("A", "B", "C")
        panel = sample_panel(12, stocks)
        normalized = normalize_intraday_volatility(
            panel.loc[:, list(stocks)]
        ).normalized_returns
        metrics, loadings, diagnostics = collect_rolling_results(
            panel,
            normalized,
            stocks=stocks,
            window_sizes=(5,),
            step_sessions=4,
            methods=("correlation",),
        )
        # Starts 0,4,7; the last window is retained even off the four-session grid.
        expected_ends = panel.index.normalize().unique()[[4, 8, 11]]
        actual = pd.to_datetime(metrics.window_end.unique(), utc=True)
        np.testing.assert_array_equal(actual, expected_ends.tz_convert("UTC"))
        self.assertEqual(set(loadings.stock), set(stocks))
        self.assertEqual(len(metrics), 9)
        self.assertEqual(set(metrics.n_observations), {20})
        self.assertLess(diagnostics.max_abs_residual_benchmark_corr.max(), 1e-12)

    def test_rolling_rejects_misaligned_normalized_panel(self):
        panel = sample_panel(8)
        normalized = panel.loc[:, list(CORE_UNIVERSE)].iloc[::-1]
        with self.assertRaisesRegex(ValueError, "identical row indices"):
            collect_rolling_results(panel, normalized, window_sizes=(5,))

    def test_session_moments_match_explicit_resampling(self):
        panel = sample_panel(5).loc[:, list(CORE_UNIVERSE)]
        moments = build_session_moments(panel)
        draw = np.array([2, 0, 2, 4, 1])
        groups = [group for _, group in panel.groupby(panel.index.normalize())]
        explicit = pd.concat([groups[i] for i in draw]).corr()
        np.testing.assert_allclose(
            bootstrap_correlation(moments, draw), explicit, atol=1e-12
        )

    def test_bootstrap_supports_a_different_universe_and_factor_count(self):
        panel = sample_panel(8, ("A", "B", "C", "D")).iloc[:, :4]
        reference = fit_l1_rotation(
            fit_pca(panel, "correlation"), n_components=2, n_starts=8, random_state=15
        )
        probabilities, stability = bootstrap_local_factors(
            panel,
            reference,
            replications=2,
            n_starts=6,
            random_state=16,
        )
        self.assertEqual(set(probabilities.stock), set(panel.columns))
        self.assertEqual(len(stability), 4)
        np.testing.assert_allclose(
            probabilities.active_probability + probabilities.small_probability, 1
        )
        with self.assertRaisesRegex(ValueError, "stock order"):
            bootstrap_local_factors(panel.iloc[:, ::-1], reference, replications=2)

    def test_kalman_training_fit_does_not_use_holdout_values(self):
        panel = sample_panel().loc[:, list(CORE_UNIVERSE)]
        sessions = panel.groupby(panel.index.normalize()).sum()
        window = make_windows(sessions)[0]
        original = fit_window(sessions, window, k=3, starts=8, seed=42)
        changed = sessions.copy()
        changed.iloc[window.test] *= 50
        perturbed = fit_window(changed, window, k=3, starts=8, seed=42)
        np.testing.assert_allclose(original["H"], perturbed["H"])
        np.testing.assert_allclose(
            original["train_filter"]["filt"], perturbed["train_filter"]["filt"]
        )
        np.testing.assert_allclose(
            original["dynamic_train"], perturbed["dynamic_train"]
        )

    def test_preprocessing_excludes_overnight_and_imputation_returns(self):
        index = pd.to_datetime(
            [
                "2023-01-02 09:30",
                "2023-01-02 09:31",
                "2023-01-02 09:32",
                "2023-01-03 09:30",
            ]
        )
        prices = pd.DataFrame({"A": [100.0, np.nan, 102.0, 150.0]}, index=index)
        returns = compute_intraday_returns(prices.ffill())
        self.assertTrue(returns.iloc[[0, 3]].isna().all().all())
        mask = build_contaminated_mask(prices.isna(), returns)
        self.assertEqual(mask.A.tolist(), [False, True, True, False])


class PipelineIntegrationTest(unittest.TestCase):
    def test_rolling_charts_accept_daylight_saving_offsets(self):
        from reporting.rolling_pca import plot_benchmark_variance_removed

        panel = sample_panel(65)
        normalized = normalize_intraday_volatility(
            panel.loc[:, list(CORE_UNIVERSE)]
        ).normalized_returns
        metrics, _, _ = collect_rolling_results(panel, normalized)
        self.assertTrue(metrics.window_end.str.endswith("-05:00").any())
        self.assertTrue(metrics.window_end.str.endswith("-04:00").any())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plot_benchmark_variance_removed(metrics, out_dir=directory)
            self.assertTrue((directory / "11_benchmark_variance_removed.png").is_file())

    def test_numbered_stages_do_not_depend_on_matplotlib(self):
        for path in SRC_DIR.glob("[0-9][0-9]*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = [
                n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
            ]
            imports += [
                a.name
                for n in ast.walk(tree)
                if isinstance(n, ast.Import)
                for a in n.names
            ]
            self.assertFalse(
                any(name and name.startswith("matplotlib") for name in imports),
                path.name,
            )

    def test_baseline_stages_write_expected_artifacts(self):
        panel = sample_panel()
        counts = {
            "07_baseline_pca.py": (10, 3),
            "08_intraday_normalization.py": (6, 1),
            "09_benchmark_residualization.py": (7, 0),
            "10_variance_decomposition.py": (5, 2),
        }
        for filename, (tables, charts) in counts.items():
            with (
                self.subTest(stage=filename),
                tempfile.TemporaryDirectory() as temporary,
            ):
                module = load_stage(filename)
                directory = Path(temporary)
                with (
                    patch.object(module, "OUT_DIR", directory),
                    patch.object(module, "ensure_project_directories"),
                    patch("pandas.read_parquet", return_value=panel.copy()),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    module.main()
                self.assertEqual(len(list(directory.glob("*.csv"))), tables)
                self.assertEqual(len(list(directory.glob("*.png"))), charts)

    def test_dynamic_stage_can_compute_without_writing_files(self):
        module = load_stage("14_dynamic_local_factor_regimes.py")
        panel = sample_panel(125)
        before = panel.copy()
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(module, "OUT_DIR", Path(temporary) / "unused"),
        ):
            results = module.run_analysis(
                panel,
                step_sessions=60,
                primary_starts=6,
                sensitivity_starts=8,
                full_sample_starts=10,
                make_figures=False,
                write_files=False,
            )
            self.assertFalse(module.OUT_DIR.exists())
        assert_frame_equal(panel, before)
        self.assertFalse(results["loadings"].empty)
        self.assertEqual(
            set(results["stability"].alignment_method),
            {"past_only", "ex_post_full_sample_alignment"},
        )


if __name__ == "__main__":
    unittest.main()
