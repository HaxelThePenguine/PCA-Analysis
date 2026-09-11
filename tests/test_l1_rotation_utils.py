"""Regression tests for sparse local-factor identification utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from l1_rotation_utils import (  # noqa: E402
    align_loading_columns,
    cartesian_to_spherical,
    fit_l1_rotation,
    l1_rotate_basis,
    local_factor_test,
    spherical_to_cartesian,
)
from pca_utils import fit_pca  # noqa: E402


class L1RotationUtilitiesTest(unittest.TestCase):
    """Check geometric, identification, and reconstruction invariants."""

    def test_spherical_round_trip(self) -> None:
        direction = np.array([-0.25, 0.45, -0.62, 0.59])
        direction /= np.linalg.norm(direction)
        recovered = spherical_to_cartesian(cartesian_to_spherical(direction))
        np.testing.assert_allclose(recovered, direction, atol=1e-12)

    def test_rotation_recovers_disjoint_sparse_directions(self) -> None:
        n_variables = 12
        n_factors = 3
        true_loadings = np.zeros((n_variables, n_factors))
        for factor in range(n_factors):
            true_loadings[4 * factor : 4 * (factor + 1), factor] = np.sqrt(3.0)

        generator = np.random.default_rng(4)
        scramble, _ = np.linalg.qr(generator.normal(size=(n_factors, n_factors)))
        initial = true_loadings @ scramble
        result = l1_rotate_basis(initial, n_starts=300, random_state=12)
        _, _, similarities = align_loading_columns(
            true_loadings,
            result.rotated_loadings,
        )

        self.assertGreater(float(similarities.min()), 0.999)
        self.assertLessEqual(
            float(np.abs(result.rotated_loadings).sum()),
            float(np.abs(initial).sum()),
        )

    def test_local_factor_test_detects_many_small_loadings(self) -> None:
        loadings = np.zeros((12, 2))
        loadings[:4, 0] = np.sqrt(3.0)
        loadings[4:, 1] = np.sqrt(1.5)
        diagnostic = local_factor_test(loadings)
        self.assertTrue(diagnostic.has_local_factors)
        self.assertGreater(int(diagnostic.small_counts.max()), diagnostic.gamma_n)

    def test_oblique_rotation_preserves_pca_reconstruction(self) -> None:
        generator = np.random.default_rng(21)
        factors = generator.normal(size=(600, 3))
        structural_loadings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.7, 0.0],
                [0.0, 0.4, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.7],
            ]
        )
        data = factors @ structural_loadings.T
        data += generator.normal(scale=0.08, size=data.shape)
        frame = pd.DataFrame(data, columns=[f"X{i}" for i in range(data.shape[1])])
        pca = fit_pca(frame, method="correlation")
        result = fit_l1_rotation(
            pca,
            n_components=3,
            n_starts=250,
            random_state=31,
        )

        self.assertLess(result.reconstruction_max_abs_error, 1e-10)
        self.assertAlmostEqual(
            result.reconstruction_pct,
            100.0 * pca.explained[:3].sum(),
            places=10,
        )
        self.assertGreater(abs(np.linalg.det(result.rotation.to_numpy())), 1e-4)

    def test_alignment_handles_permutation_and_sign(self) -> None:
        reference = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.2]])
        estimate = np.column_stack([-reference[:, 1], reference[:, 0]])
        aligned, order, similarities = align_loading_columns(reference, estimate)
        np.testing.assert_allclose(aligned, reference)
        np.testing.assert_array_equal(order, [1, 0])
        np.testing.assert_allclose(similarities, 1.0)


if __name__ == "__main__":
    unittest.main()
