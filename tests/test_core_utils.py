"""PCA reconstruction, benchmark residuals, scaling, and stage imports."""

from __future__ import annotations

import runpy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.benchmark import residualize_against_benchmarks  # noqa: E402
from utils.data import normalize_intraday_volatility  # noqa: E402
from utils.pca import fit_pca, fit_varimax, pca_diagnostics, reconstruction_by_stock  # noqa: E402


class ScriptStructureTest(unittest.TestCase):
    """Numbered stages should import without running the pipeline."""

    def test_numbered_scripts_are_import_safe(self) -> None:
        for path in sorted(SRC_DIR.glob("[0-9][0-9]*.py")):
            with self.subTest(script=path.name):
                namespace = runpy.run_path(path, run_name=f"test_{path.stem}")
                self.assertTrue(callable(namespace.get("main")))


class PCAUtilitiesTest(unittest.TestCase):
    """PCA variance, reconstruction, and rotation checks."""

    def setUp(self) -> None:
        generator = np.random.default_rng(42)
        common = generator.normal(size=(250, 1))
        noise = generator.normal(scale=0.3, size=(250, 4))
        self.data = pd.DataFrame(common @ np.array([[1.0, 0.8, -0.5, 0.3]]) + noise, columns=list("ABCD"))

    def test_pca_matrices_reconstruction_and_variance_ledger(self) -> None:
        for method in ("covariance", "correlation"):
            with self.subTest(method=method):
                result = fit_pca(self.data, method=method)
                expected = self.data.cov() if method == "covariance" else self.data.corr()
                assert_frame_equal(result.matrix, expected, atol=1e-12)
                reconstructed = result.scores.to_numpy() @ result.eigenvectors.T
                np.testing.assert_allclose(reconstructed, result.analysis_data.to_numpy(), atol=1e-12)
                self.assertAlmostEqual(float(result.explained.sum()), 1.0)
                self.assertLess(max(pca_diagnostics(result)), 1e-12)
                ledger = reconstruction_by_stock(result, 3)
                retained = 1 - ledger.residual_variance.sum() / ledger.original_variance.sum()
                self.assertAlmostEqual(retained, result.explained[:3].sum())
                full = reconstruction_by_stock(result, self.data.shape[1])
                np.testing.assert_allclose(full.explained_pct, 100, atol=1e-10)
                if method == "correlation":
                    loadings = pd.concat([self.data, result.scores], axis=1).corr()
                    np.testing.assert_allclose(
                        result.loadings, loadings.loc[self.data.columns, result.scores.columns], atol=1e-12,
                    )

    def test_varimax_preserves_selected_subspace(self) -> None:
        result = fit_pca(self.data, method="correlation")
        rotated = fit_varimax(result, n_components=3)
        np.testing.assert_allclose(
            rotated.rotation.to_numpy().T @ rotated.rotation.to_numpy(), np.eye(3), atol=1e-12,
        )


class DataUtilitiesTest(unittest.TestCase):
    """SPY/XLF projection and intraday scaling checks."""

    def test_residuals_are_orthogonal_to_benchmarks(self) -> None:
        generator = np.random.default_rng(7)
        benchmarks = generator.normal(size=(100, 2))
        stocks = benchmarks @ np.array([[1.2, -0.4], [0.3, 0.8]])
        stocks += generator.normal(scale=0.1, size=stocks.shape)
        panel = pd.DataFrame(np.column_stack([stocks, benchmarks]), columns=["A", "B", "SPY", "XLF"])

        result = residualize_against_benchmarks(panel, ["A", "B"])
        for benchmark in ("SPY", "XLF"):
            correlations = result["residual_returns"].corrwith(panel[benchmark])
            np.testing.assert_allclose(correlations, 0.0, atol=1e-12)

    def test_intraday_normalization_has_unit_minute_volatility(self) -> None:
        index = pd.to_datetime(
            [
                "2026-01-02 09:31", "2026-01-02 09:32", "2026-01-05 09:31", "2026-01-05 09:32",
                "2026-01-06 09:31", "2026-01-06 09:32",
            ]
        ).tz_localize("America/New_York")
        returns = pd.DataFrame({"A": [1.0, 2.0, 2.0, 4.0, 4.0, 8.0]}, index=index)
        normalized = normalize_intraday_volatility(returns).normalized_returns
        minute = normalized.index.hour * 60 + normalized.index.minute
        np.testing.assert_allclose(normalized.groupby(minute).std(ddof=1).to_numpy(), 1.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
