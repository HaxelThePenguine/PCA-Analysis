"""Regression tests for the rolling dynamic local-factor stage."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.dynamic_local_factor_regimes import (  # noqa: E402
    AlignmentResult,
    WindowDefinition,
    WindowFit,
    _align_fit,
    _instability_reasons,
    _jaccard_support,
    build_window_definitions,
    classify_window_regime,
    deterministic_seed,
)
from utils.l1_rotation import fit_l1_rotation  # noqa: E402
from utils.pca import fit_pca  # noqa: E402


def _intraday_panel(n_sessions: int = 130) -> pd.DataFrame:
    """Create a small complete panel with two observations per session."""

    sessions = pd.bdate_range("2022-01-03", periods=n_sessions)
    index = pd.DatetimeIndex(
        [
            timestamp + pd.Timedelta(minutes=offset)
            for timestamp in sessions
            for offset in (1, 2)
        ]
    )
    values = np.arange(len(index), dtype=float)[:, None]
    return pd.DataFrame(values, index=index, columns=["A"])


class DynamicWindowTest(unittest.TestCase):
    """Check exact trailing-session construction and causal window metadata."""

    def test_windows_use_exact_sessions_and_never_look_forward(self) -> None:
        panel = _intraday_panel()
        definitions = build_window_definitions(
            panel, window_sizes=(60, 120), step_sessions=5
        )

        self.assertEqual(
            {definition.window_sessions for definition in definitions}, {60, 120}
        )
        for definition in definitions:
            selected = panel.iloc[definition.positions]
            selected_sessions = pd.DatetimeIndex(selected.index.normalize()).unique()
            self.assertEqual(len(selected_sessions), definition.window_sessions)
            self.assertEqual(selected_sessions[0], definition.window_start)
            self.assertEqual(selected_sessions[-1], definition.window_end)
            self.assertLessEqual(
                selected.index.normalize().max(), definition.window_end
            )

    def test_regime_classifier_retains_mixed_window_flag(self) -> None:
        sessions = pd.to_datetime(
            ["2023-02-28", "2023-03-15", "2023-05-15", "2023-06-01"]
        )
        regime, sequence, mixed = classify_window_regime(sessions)
        self.assertTrue(mixed)
        self.assertIn("pre_banking_crisis", sequence)
        self.assertIn("banking_crisis_mar_may_2023", sequence)
        self.assertIn(regime, sequence.split(";"))


class DynamicAlignmentTest(unittest.TestCase):
    """Ensure the same factor permutation/sign is applied to all outputs."""

    def test_alignment_applies_permutation_and_sign_to_scores(self) -> None:
        stocks = [f"S{i}" for i in range(12)]
        reference_values = np.zeros((12, 3))
        reference_values[:4, 0] = 1.0
        reference_values[4:8, 1] = 1.0
        reference_values[8:, 2] = 1.0
        reference = pd.DataFrame(
            reference_values, index=stocks, columns=["LF1", "LF2", "LF3"]
        )
        order = [2, 0, 1]
        signs = np.array([-1.0, 1.0, -1.0])
        estimate_values = reference_values[:, order] * signs[None, :]
        base_score_values = np.arange(36, dtype=float).reshape(12, 3)
        score_values = base_score_values[:, order] * signs[None, :]
        definition = WindowDefinition(
            window_id="60_2023-01-01",
            window_sessions=60,
            start_index=0,
            end_index=59,
            window_start=pd.Timestamp("2022-10-01"),
            window_end=pd.Timestamp("2023-01-01"),
            positions=np.arange(60),
            n_observations=60,
            regime="pre_banking_crisis",
            regime_sequence="pre_banking_crisis",
            crosses_regime_boundary=False,
            is_quarterly_window=False,
        )
        fit = WindowFit(
            definition=definition,
            structural_loadings=pd.DataFrame(
                estimate_values, index=stocks, columns=["LF1", "LF2", "LF3"]
            ),
            score_loadings=pd.DataFrame(
                score_values, index=stocks, columns=["LF1", "LF2", "LF3"]
            ),
            score_correlation=pd.DataFrame(
                np.eye(3), index=["LF1", "LF2", "LF3"], columns=["LF1", "LF2", "LF3"]
            ),
            eigenvalues=np.array([3.0, 2.0, 1.0, 0.5]),
            explained=np.array([0.45, 0.30, 0.15, 0.10]),
            small_loading_threshold=0.1,
            gamma_n=2,
            small_counts=np.array([8, 8, 8]),
            local_factor_flags=np.array([True, True, True]),
            has_any_local_factors=True,
            l1_norms=np.array([4.0, 4.0, 4.0]),
            solution_frequencies=np.array([1, 1, 1]),
            sources=("LF1", "LF2", "LF3"),
            optimizer_success_rate=1.0,
            rotation_condition_number=1.0,
            reconstruction_pct=90.0,
            reconstruction_max_abs_error=1e-12,
            benchmark_variance_removed_pct=10.0,
            max_abs_residual_benchmark_corr=1e-12,
            n_starts=20,
            random_seed=123,
        )

        aligned = _align_fit(fit, reference)

        np.testing.assert_allclose(
            aligned.structural_loadings.to_numpy(), reference_values
        )
        np.testing.assert_allclose(aligned.score_loadings.to_numpy(), base_score_values)
        self.assertIsInstance(aligned, AlignmentResult)


class DynamicNumericalTest(unittest.TestCase):
    """Check deterministic seeds, subspace preservation, and support flags."""

    def test_seed_is_deterministic_and_start_count_specific(self) -> None:
        start = pd.Timestamp("2024-01-02")
        end = pd.Timestamp("2024-03-25")
        first = deterministic_seed(60, start, end, 200)
        self.assertEqual(first, deterministic_seed(60, start, end, 200))
        self.assertNotEqual(first, deterministic_seed(60, start, end, 500))
        self.assertNotEqual(first, deterministic_seed(120, start, end, 200))

    def test_rotation_preserves_selected_pca_subspace(self) -> None:
        generator = np.random.default_rng(19)
        data = pd.DataFrame(
            generator.normal(size=(220, 8)),
            columns=[f"X{i}" for i in range(8)],
        )
        pca = fit_pca(data, method="correlation")
        result = fit_l1_rotation(pca, n_components=3, n_starts=40, random_state=27)
        self.assertLess(result.reconstruction_max_abs_error, 1e-10)
        self.assertAlmostEqual(
            result.reconstruction_pct,
            100.0 * pca.explained[:3].sum(),
            places=10,
        )

    def test_synthetic_support_shift_is_detected_without_stable_false_change(
        self,
    ) -> None:
        stable = np.zeros((12, 3))
        stable[:4, 0] = 1.0
        stable[4:8, 1] = 1.0
        stable[8:, 2] = 1.0
        shifted = stable.copy()
        shifted[:4, 0] = 0.0
        shifted[4:8, 0] = 1.0

        stable_jaccard = _jaccard_support(stable, stable, threshold=0.5)
        shifted_jaccard = _jaccard_support(shifted, stable, threshold=0.5)
        np.testing.assert_allclose(stable_jaccard, 1.0)
        self.assertLess(shifted_jaccard[0], 0.5)
        self.assertTrue(
            _instability_reasons(0.95, 0.95, float(shifted_jaccard[0]), False, 1.0)
        )
        self.assertEqual(_instability_reasons(0.95, 0.95, 1.0, False, 1.0), [])


if __name__ == "__main__":
    unittest.main()
