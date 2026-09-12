"""Regression tests for the frozen Stage 16 OOS protocol."""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.network_har import (  # noqa: E402
    HARConfig,
    _alpha_max,
    build_hac_tests,
    fit_partialling_out,
)
from utils.oos_har_network import (  # noqa: E402
    FORECAST_MODELS,
    OOSConfig,
    _session_date_index,
    _write_forecast_checkpoint,
    build_calendar_har_features,
    build_hac_inference,
    draw_shared_date_indices,
    issue_target_free_forecasts,
    paired_loss_differentials,
    score_forecasts,
    strict_clean_return_panel,
    validate_identical_model_keys,
    validate_forecast_ledger,
)
from utils.preprocessing import (  # noqa: E402
    compute_intraday_returns,
    compute_strict_intraday_returns,
)


NY_TZ = "America/New_York"


def _calendar(n_sessions: int = 70) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_sessions)
    opens = dates.tz_localize(NY_TZ) + pd.Timedelta(hours=9, minutes=30)
    closes = dates.tz_localize(NY_TZ) + pd.Timedelta(hours=16)
    return pd.DataFrame(
        {
            "date": dates,
            "session_open": opens,
            "session_close": closes,
            "open_minute": 570,
            "close_minute": 960,
            "expected_bars": 390,
        }
    )


def _log_variance(n_sessions: int = 70) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar = _calendar(n_sessions)
    dates = pd.DatetimeIndex(calendar["date"])
    rng = np.random.default_rng(16016)
    values = rng.normal(loc=-0.1, scale=0.25, size=(n_sessions, 3))
    panel = pd.DataFrame(values, index=dates, columns=["A", "B", "C"])
    return panel, calendar


def _small_config() -> OOSConfig:
    har = HARConfig(
        n_jobs=1,
        min_training_observations=10,
        rolling_training_observations=20,
        tuning_frequency=5,
        cv_splits=2,
        alpha_fractions=(0.03, 0.10, 0.30, 1.0),
        lasso_max_iter=300,
        lasso_tolerance=1e-6,
        hac_lag=5,
    )
    return OOSConfig(
        protocol_version="test-oos-v2",
        factor_windows=(4,),
        factor_window_primary=4,
        factor_l1_starts=4,
        min_valid_training_observations=10,
        checkpoint_frequency_origins=2,
        loss_bootstrap_repetitions=20,
        loss_bootstrap_block_lengths=(5,),
        edge_bootstrap_repetitions=10,
        edge_bootstrap_checkpoint_step=10,
        har=har,
    )


class StrictPreprocessingTest(unittest.TestCase):
    def test_strict_returns_reject_a_removed_minute(self) -> None:
        index = pd.DatetimeIndex(
            [
                "2026-01-02 09:30",
                "2026-01-02 09:31",
                "2026-01-02 09:33",
            ]
        ).tz_localize(NY_TZ)
        prices = pd.DataFrame({"A": [100.0, 101.0, 103.0]}, index=index)

        legacy = compute_intraday_returns(prices)
        strict = compute_strict_intraday_returns(prices)

        self.assertTrue(np.isfinite(legacy.iloc[2, 0]))
        self.assertTrue(pd.isna(strict.iloc[2, 0]))

    def test_strict_panel_preserves_contamination_information(self) -> None:
        index = pd.DatetimeIndex(
            [
                "2026-01-02 09:30",
                "2026-01-02 09:31",
                "2026-01-02 09:33",
                "2026-01-02 09:34",
            ]
        ).tz_localize(NY_TZ)
        prices = pd.DataFrame({"A": [100.0, 101.0, 103.0, 104.0]}, index=index)
        missing = pd.DataFrame(False, index=index, columns=["A"])
        cleaned, complete, contaminated = strict_clean_return_panel(prices, missing)

        self.assertTrue(pd.isna(cleaned.iloc[2, 0]))
        self.assertFalse(bool(contaminated.iloc[2, 0]))
        self.assertFalse(bool(complete.index.isin([index[2]]).any()))
        self.assertTrue(bool(complete.index.isin([index[3]]).any()))


class CalendarAndCausalityTest(unittest.TestCase):
    def test_session_dates_accept_mixed_checkpoint_iso_formats(self) -> None:
        dates = _session_date_index(
            ["2026-09-04", "2026-09-08 00:00:00", "2026-09-09T00:00:00"]
        )
        self.assertTrue(
            dates.equals(
                pd.DatetimeIndex(["2026-09-04", "2026-09-08", "2026-09-09"])
            )
        )

    def test_checkpoint_normalizes_mixed_legacy_labels_for_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            _write_forecast_checkpoint(
                checkpoint,
                forecasts=[{"har_window": 252}, {"har_window": "252"}],
                coefficients=[],
                edges=[],
                tuning_rows=[],
                origin_rows=[],
                state={"table_format": "parquet"},
            )
            stored = pd.read_parquet(checkpoint / "forecast_ledger.parquet")
        self.assertEqual(stored["har_window"].tolist(), ["252", "252"])

    def test_har_features_use_exchange_positions_and_keep_missing_sessions(self) -> None:
        panel, calendar = _log_variance()
        missing_date = panel.index[30]
        panel.loc[missing_date, "A"] = np.nan
        design = build_calendar_har_features(panel, calendar)

        self.assertEqual(len(design.calendar_dates), len(calendar))
        self.assertIn(missing_date, design.targets)
        origin_after_gap = panel.index[31]
        row = design.features.loc[origin_after_gap, ("A", "M")]
        self.assertTrue(pd.isna(row))

    def test_issuance_does_not_depend_on_the_current_target(self) -> None:
        original, calendar = _log_variance()
        changed = original.copy()
        changed.loc[original.index[50]:, :] += 7.0
        config = _small_config()

        first = issue_target_free_forecasts(
            original,
            calendar,
            spec_name="test_spec",
            config=config,
            run_id="first",
        ).forecasts
        second = issue_target_free_forecasts(
            changed,
            calendar,
            spec_name="test_spec",
            config=config,
            run_id="second",
        ).forecasts
        cutoff = original.index[50]
        first = first[first["target_date"] < cutoff].sort_values(
            ["model", "stock", "forecast_origin"]
        )
        second = second[second["target_date"] < cutoff].sort_values(
            ["model", "stock", "forecast_origin"]
        )
        self.assertEqual(len(first), len(second))
        np.testing.assert_allclose(
            first["predicted_log_variance"].to_numpy(dtype=float),
            second["predicted_log_variance"].to_numpy(dtype=float),
        )
        self.assertTrue((first["target_data_present_at_issue"] == False).all())  # noqa: E712
        self.assertGreater(len(validate_identical_model_keys(first)), 0)
        validate_forecast_ledger(first)
        for forbidden in ("actual_rv", "actual_log_variance", "qlike", "loss_difference"):
            self.assertNotIn(forbidden, first.columns)

    def test_checkpoint_resume_matches_uninterrupted_issuance(self) -> None:
        panel, calendar = _log_variance()
        config = _small_config()
        full = issue_target_free_forecasts(
            panel,
            calendar,
            spec_name="resume_spec",
            config=config,
            run_id="full",
        ).forecasts
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            partial = issue_target_free_forecasts(
                panel,
                calendar,
                spec_name="resume_spec",
                config=config,
                run_id="resume",
                checkpoint_dir=checkpoint,
                stop_after_origins=3,
            )
            self.assertEqual(partial.status, "checkpointed_partial")
            resumed = issue_target_free_forecasts(
                panel,
                calendar,
                spec_name="resume_spec",
                config=replace(config, har=replace(config.har, n_jobs=2)),
                run_id="resume",
                checkpoint_dir=checkpoint,
                resume=True,
            ).forecasts
        key = ["spec_name", "model", "stock", "forecast_origin", "target_date"]
        left = full.sort_values(key).reset_index(drop=True)
        right = resumed.sort_values(key).reset_index(drop=True)
        self.assertEqual(left[key].astype(str).to_dict("list"), right[key].astype(str).to_dict("list"))
        np.testing.assert_allclose(
            left["predicted_log_variance"].to_numpy(dtype=float),
            right["predicted_log_variance"].to_numpy(dtype=float),
        )

    def test_resume_rejects_model_changes(self) -> None:
        panel, calendar = _log_variance()
        config = _small_config()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            issue_target_free_forecasts(
                panel,
                calendar,
                spec_name="strict_spec",
                config=config,
                run_id="strict",
                checkpoint_dir=checkpoint,
                stop_after_origins=3,
            )
            changed = replace(
                config,
                har=replace(config.har, alpha_fractions=(0.10, 0.30, 1.0)),
            )
            with self.assertRaisesRegex(ValueError, "forecast specification"):
                issue_target_free_forecasts(
                    panel,
                    calendar,
                    spec_name="strict_spec",
                    config=changed,
                    run_id="strict",
                    checkpoint_dir=checkpoint,
                    resume=True,
                )

    def test_prospective_mode_never_backfills_opened_targets(self) -> None:
        panel, calendar = _log_variance()
        result = issue_target_free_forecasts(
            panel,
            calendar,
            spec_name="prospective_spec",
            config=_small_config(),
            run_id="prospective",
            mode="prospective",
            freeze_timestamp="2023-12-31T00:00:00Z",
            issuance_timestamp="2025-01-01T00:00:00Z",
        )
        self.assertTrue(result.forecasts.empty)
        self.assertTrue(
            result.origin_diagnostics["status"]
            .eq("not_issued_target_already_opened")
            .all()
        )


class ScoringAndInferenceTest(unittest.TestCase):
    def _forecast_ledger(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        target_dates = pd.DatetimeIndex(["2024-03-01", "2024-03-04"])
        rows: list[dict[str, object]] = []
        for target_date in target_dates:
            origin = target_date - pd.offsets.BDay(1)
            for stock in ("A", "B"):
                for model in FORECAST_MODELS:
                    predicted = 1.0 if model == "own_har" else 1.5
                    rows.append(
                        {
                            "protocol_version": "test-oos-v2",
                            "run_id": "score",
                            "spec_name": "score_spec",
                            "model": model,
                            "stock": stock,
                            "forecast_origin": origin,
                            "as_of": origin + pd.Timedelta(hours=16),
                            "target_date": target_date,
                            "target_data_present_at_issue": False,
                            "predicted_log_variance": np.log(predicted),
                            "predicted_variance": predicted,
                        }
                    )
        forecasts = pd.DataFrame(rows)
        realized = pd.DataFrame(
            {"A": [2.0, 0.0], "B": [2.0, 2.0]},
            index=target_dates,
        )
        metadata = pd.DataFrame(
            {
                "session_date": target_dates,
                "target_quality_eligible": True,
                "target_quality_reason": "eligible_by_predeclared_rule",
            }
        )
        return forecasts, realized, metadata

    def test_model_key_validation_accepts_pandas_string_dtype(self) -> None:
        forecasts, _, _ = self._forecast_ledger()
        forecasts["model"] = forecasts["model"].astype("string")
        rows = validate_identical_model_keys(forecasts)
        self.assertTrue(rows["n_models"].eq(len(FORECAST_MODELS)).all())
        self.assertEqual(rows["models"].nunique(), 1)
        with self.assertRaisesRegex(ValueError, "identical forecast keys"):
            validate_identical_model_keys(forecasts.iloc[1:].copy())

    def test_scoring_uses_raw_positive_variance_and_keeps_nonpositive_status(self) -> None:
        forecasts, realized, metadata = self._forecast_ledger()
        scored = score_forecasts(
            forecasts,
            {"score_spec": realized},
            {"score_spec": metadata},
            config=_small_config(),
        )
        positive = scored[(scored["target_date"] == pd.Timestamp("2024-03-01")) & (scored["stock"] == "A")]
        self.assertTrue((positive["score_status"] == "scored").all())
        expected = 2.0 / 1.0 - np.log(2.0) - 1.0
        self.assertAlmostEqual(float(positive.loc[positive["model"] == "own_har", "qlike"].iloc[0]), expected)
        nonpositive = scored[(scored["target_date"] == pd.Timestamp("2024-03-04")) & (scored["stock"] == "A")]
        self.assertTrue((nonpositive["score_status"] == "nonpositive_target").all())
        self.assertFalse(nonpositive["comparison_eligible"].any())

    def test_hac_aggregates_the_stock_panel_by_date(self) -> None:
        forecasts, realized, metadata = self._forecast_ledger()
        scored = score_forecasts(
            forecasts,
            {"score_spec": realized},
            {"score_spec": metadata},
            config=_small_config(),
        )
        differences = paired_loss_differentials(scored)
        self.assertEqual(len(differences), 3)
        hac = build_hac_inference(scored, max_lags=(1,))
        pooled = hac[hac["level"] == "pooled_date_aggregated"]
        self.assertEqual(int(pooled["n_dates"].iloc[0]), 2)
        self.assertEqual(int(pooled["n_stock_date"].iloc[0]), 3)

    def test_shared_moving_block_draws_are_reproducible(self) -> None:
        first = draw_shared_date_indices(20, 8, 5, 16016)
        second = draw_shared_date_indices(20, 8, 5, 16016)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (8, 20))

    def test_standardized_alpha_max_has_zero_solution_at_threshold(self) -> None:
        rng = np.random.default_rng(7)
        own = rng.normal(size=(120, 3))
        cross = rng.normal(size=(120, 5))
        cross[:, 1] = 8.0 * cross[:, 0] + 0.1 * rng.normal(size=120)
        y = 1.0 + own @ np.array([0.2, -0.4, 0.7]) + cross @ np.array([0.8, 0.0, -0.4, 0.2, 0.1])
        alpha = _alpha_max(y, own, cross)
        fit = fit_partialling_out(
            y,
            own,
            cross,
            alpha=alpha,
            cross_feature_names=[f"x{index}" for index in range(cross.shape[1])],
            config=HARConfig(lasso_max_iter=500),
        )
        np.testing.assert_allclose(fit.network_coefficients_standardized, 0.0, atol=1e-12)
        self.assertTrue(fit.lasso_converged)
        self.assertLessEqual(fit.lasso_kkt_max_violation, 1e-10)


if __name__ == "__main__":
    unittest.main()
