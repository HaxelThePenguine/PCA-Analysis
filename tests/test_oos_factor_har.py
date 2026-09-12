"""Regression tests for Stage 17 factor-augmented OOS forecasts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.factor_adjusted_residuals import FactorFit, apply_factor_fit  # noqa: E402
from utils.factor_har_oos import (  # noqa: E402
    FACTOR_HAR_MODELS,
    build_factor_har_design,
    issue_factor_har_forecasts,
)
from utils.network_har import HARConfig  # noqa: E402
from utils.oos_har_network import OOSConfig, validate_identical_model_keys  # noqa: E402


NY_TZ = "America/New_York"


def _calendar(n: int = 75) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "session_open": dates.tz_localize(NY_TZ) + pd.Timedelta(hours=9, minutes=30),
            "session_close": dates.tz_localize(NY_TZ) + pd.Timedelta(hours=16),
            "open_minute": 570,
            "close_minute": 960,
            "expected_bars": 390,
        }
    )


def _panels(n: int = 75) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calendar = _calendar(n)
    dates = pd.DatetimeIndex(calendar["date"])
    rng = np.random.default_rng(17017)
    common = rng.normal(scale=0.18, size=n)
    local = np.column_stack(
        [common + rng.normal(scale=0.08, size=n) for _ in range(3)]
    )
    banks = np.column_stack(
        [
            -0.2 + 0.35 * common + rng.normal(scale=0.12, size=n),
            -0.1 - 0.20 * common + rng.normal(scale=0.13, size=n),
            -0.3 + 0.15 * common + rng.normal(scale=0.11, size=n),
        ]
    )
    return (
        pd.DataFrame(banks, index=dates, columns=["A", "B", "C"]),
        pd.DataFrame(common, index=dates, columns=["COMMON"]),
        pd.DataFrame(local, index=dates, columns=["LF1", "LF2", "LF3"]),
        calendar,
    )


def _config() -> OOSConfig:
    har = HARConfig(
        n_jobs=1,
        min_training_observations=10,
        rolling_training_observations=30,
        tuning_frequency=5,
        cv_splits=2,
        alpha_fractions=(0.1, 0.3, 1.0),
        lasso_max_iter=500,
    )
    return OOSConfig(
        protocol_version="test-factor-har-v1",
        min_valid_training_observations=10,
        har=har,
    )


class FactorScoreTest(unittest.TestCase):
    def test_oblique_local_scores_reconstruct_the_pca_projection(self) -> None:
        q, _ = np.linalg.qr(np.array([[1.0, 0.2], [0.1, 1.0], [0.7, -0.4]]))
        rotation = np.array([[1.0, 0.35], [-0.2, 1.1]])
        loadings = q @ rotation
        stocks = ("A", "B", "C")
        factors = ("LF1", "LF2")
        fit = FactorFit(
            stocks=stocks,
            benchmarks=("SPY", "XLF"),
            coefficients=pd.DataFrame(
                0.0,
                index=stocks,
                columns=["alpha", "beta_SPY", "beta_XLF"],
            ),
            residual_means=pd.Series(0.0, index=stocks),
            residual_scales=pd.Series(1.0, index=stocks),
            pca_explained=np.array([0.6, 0.3]),
            pca_eigenvalues=np.array([1.8, 0.9]),
            pca_weights=pd.DataFrame(q, index=stocks, columns=["PC1", "PC2"]),
            factor_projector=pd.DataFrame(q @ q.T, index=stocks, columns=stocks),
            loadings=pd.DataFrame(loadings, index=stocks, columns=factors),
            rotation=pd.DataFrame(rotation),
            rotation_condition_number=float(np.linalg.cond(rotation)),
            projector_invariance_error=0.0,
            training_orthogonality_error=0.0,
            training_factor_variance_removed_pct=0.0,
            training_benchmark_variance_removed_pct=0.0,
            training_max_abs_residual_benchmark_corr=0.0,
            l1_rotation_status="ok",
            l1_optimizer_success_rate=1.0,
        )
        values = np.array([[0.2, -0.3, 0.4], [0.8, 0.1, -0.2]])
        panel = pd.DataFrame(
            np.column_stack([values, np.zeros((2, 2))]),
            index=pd.date_range("2026-01-02 09:31", periods=2, freq="min", tz=NY_TZ),
            columns=[*stocks, "SPY", "XLF"],
        )
        applied = apply_factor_fit(panel, fit)
        reconstructed = (
            applied["local_factor_scores"].to_numpy(dtype=float) @ loadings.T
        )
        np.testing.assert_allclose(reconstructed, values @ (q @ q.T), atol=1e-12)


class FactorHARForecastTest(unittest.TestCase):
    def test_designs_share_calendar_and_retain_missing_factor_history(self) -> None:
        banks, aggregate, local, calendar = _panels()
        local.loc[local.index[30], "LF2"] = np.nan
        design = build_factor_har_design(banks, aggregate, local, calendar)
        self.assertTrue(design.origins.equals(design.aggregate_features.index))
        self.assertTrue(design.origins.equals(design.local_features.index))
        self.assertTrue(pd.isna(design.local_features.loc[local.index[31], ("LF2", "M")]))

    def test_current_target_is_not_used_and_all_models_share_keys(self) -> None:
        banks, aggregate, local, calendar = _panels()
        target_date = banks.index[55]
        changed = banks.copy()
        changed.loc[target_date, :] += 25.0
        first = issue_factor_har_forecasts(
            banks,
            aggregate,
            local,
            calendar,
            spec_name="factor_test",
            config=_config(),
            run_id="first",
        ).forecasts
        second = issue_factor_har_forecasts(
            changed,
            aggregate,
            local,
            calendar,
            spec_name="factor_test",
            config=_config(),
            run_id="second",
        ).forecasts
        first = first[first["target_date"] == target_date].sort_values(["model", "stock"])
        second = second[second["target_date"] == target_date].sort_values(["model", "stock"])
        self.assertEqual(len(first), len(second))
        self.assertGreater(len(first), 0)
        np.testing.assert_allclose(
            first["predicted_log_variance"].to_numpy(dtype=float),
            second["predicted_log_variance"].to_numpy(dtype=float),
            atol=1e-12,
        )
        self.assertTrue((first["target_data_present_at_issue"] == False).all())  # noqa: E712
        self.assertGreater(len(validate_identical_model_keys(first, FACTOR_HAR_MODELS)), 0)

    def test_prospective_mode_refuses_historical_backfill(self) -> None:
        banks, aggregate, local, calendar = _panels()
        result = issue_factor_har_forecasts(
            banks,
            aggregate,
            local,
            calendar,
            spec_name="factor_test",
            config=_config(),
            mode="prospective",
            freeze_timestamp="2030-01-01T00:00:00Z",
            issuance_timestamp="2030-01-01T00:00:01Z",
        )
        self.assertTrue(result.forecasts.empty)
        self.assertEqual(result.status, "awaiting_future_data")


if __name__ == "__main__":
    unittest.main()
