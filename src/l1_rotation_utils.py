"""L1-rotation utilities for identifying sparse local factor loadings.

The implementation follows the computational structure of Freyaldenhoven's
``l1rotation`` package: optimize the entrywise L1 norm over many directions on
the unit sphere, consolidate repeated local minima, and retain a nonsingular
basis.  Unlike Sparse PCA, this is a rotation of an already estimated loading
space and therefore does not shrink coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, log, sqrt

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import norm

from pca_utils import PCAResult


@dataclass(frozen=True)
class LocalFactorTest:
    """Small-loading diagnostic used by the reference R implementation."""

    h_n: float
    gamma_n: int
    small_counts: pd.Series
    has_local_factors: bool


@dataclass(frozen=True)
class L1BasisRotation:
    """Numerical result of rotating one orthonormal loading-space basis."""

    initial_loadings: np.ndarray
    rotated_loadings: np.ndarray
    rotation: np.ndarray
    l1_norms: np.ndarray
    small_counts: np.ndarray
    solution_frequencies: np.ndarray
    sources: tuple[str, ...]
    optimizer_success_rate: float


@dataclass(frozen=True)
class L1RotationResult:
    """Full L1-rotation fit, including scores and identification diagnostics."""

    initial_loadings: pd.DataFrame
    rotated_loadings: pd.DataFrame
    rotation: pd.DataFrame
    scores: pd.DataFrame
    score_loadings: pd.DataFrame
    l1_norms: pd.Series
    solution_frequencies: pd.Series
    sources: pd.Series
    local_factor_test: LocalFactorTest
    reconstruction_pct: float
    reconstruction_max_abs_error: float
    score_correlation: pd.DataFrame
    optimizer_success_rate: float


def spherical_to_cartesian(theta: np.ndarray) -> np.ndarray:
    """Map ``r - 1`` hyperspherical angles to a unit vector in ``R^r``."""

    angles = np.asarray(theta, dtype=float)
    if angles.ndim != 1 or len(angles) < 1:
        raise ValueError("At least one spherical angle is required.")

    dimension = len(angles) + 1
    direction = np.zeros(dimension)
    direction[0] = np.cos(angles[0])
    sine_product = 1.0
    for coordinate in range(1, dimension - 1):
        sine_product *= np.sin(angles[coordinate - 1])
        direction[coordinate] = sine_product * np.cos(angles[coordinate])
    direction[-1] = sine_product * np.sin(angles[-1])
    return direction


def cartesian_to_spherical(direction: np.ndarray) -> np.ndarray:
    """Map a non-zero Cartesian vector to ``r - 1`` hyperspherical angles."""

    vector = np.asarray(direction, dtype=float)
    if vector.ndim != 1 or len(vector) < 2:
        raise ValueError("A Cartesian vector with at least two entries is required.")
    length = np.linalg.norm(vector)
    if length <= 0 or not np.isfinite(length):
        raise ValueError("The Cartesian direction must be finite and non-zero.")
    vector = vector / length

    angles = np.zeros(len(vector) - 1)
    for coordinate in range(len(vector) - 2):
        trailing_norm = np.linalg.norm(vector[coordinate + 1 :])
        angles[coordinate] = np.arctan2(trailing_norm, vector[coordinate])
    angles[-1] = np.arctan2(vector[-1], vector[-2])
    return angles


def _canonical_direction(direction: np.ndarray) -> np.ndarray:
    """Remove sign indeterminacy using the first material coordinate."""

    vector = np.asarray(direction, dtype=float).copy()
    material = np.flatnonzero(np.abs(vector) > 1e-12)
    if len(material) and vector[material[0]] < 0:
        vector *= -1
    return vector


def _grid_size(n_factors: int) -> int:
    if n_factors == 2:
        return 500
    if n_factors == 3:
        return 1000
    if n_factors == 4:
        return 2000
    if n_factors == 5:
        return 4000
    if 5 < n_factors < 9:
        return 6000
    return 10000


def local_factor_test(
    loadings: pd.DataFrame | np.ndarray,
    *,
    alpha: float = 0.05,
    gamma_0: float = 0.03,
) -> LocalFactorTest:
    """Test whether at least one column contains unusually many small loadings.

    This reproduces the threshold and critical count in the official
    ``test_local_factors`` implementation.  Loading columns must use the
    normalization ``Lambda.T @ Lambda / n = I`` before an oblique rotation.
    """

    values = np.asarray(loadings, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError("Local-factor testing requires an n-by-r loading matrix.")
    if not np.isfinite(values).all():
        raise ValueError("The loading matrix contains non-finite values.")

    n_variables = values.shape[0]
    h_n = 1.0 / log(n_variables)
    expected_small = 2.0 * norm.cdf(h_n) - 1.0
    gamma = (
        gamma_0
        + expected_small
        + norm.ppf(1.0 - alpha / 2.0)
        * sqrt(expected_small * (1.0 - expected_small) / n_variables)
    )
    gamma_n = floor(gamma * n_variables)
    counts = (np.abs(values) < h_n).sum(axis=0)
    labels = (
        list(loadings.columns)
        if isinstance(loadings, pd.DataFrame)
        else [f"LF{i}" for i in range(1, values.shape[1] + 1)]
    )
    count_series = pd.Series(counts, index=labels, name="n_small", dtype=int)
    return LocalFactorTest(
        h_n=h_n,
        gamma_n=gamma_n,
        small_counts=count_series,
        has_local_factors=bool(counts.max() > gamma_n),
    )


def _select_independent_directions(
    initial_loadings: np.ndarray,
    representatives: list[dict[str, object]],
    n_factors: int,
    eigenvalues: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Greedily keep sparse local minima that form a well-conditioned basis."""

    n_variables = initial_loadings.shape[0]
    h_n = 1.0 / log(n_variables)
    ordered = sorted(
        representatives,
        key=lambda item: (
            -int(np.sum(np.abs(initial_loadings @ item["direction"]) < h_n)),
            float(item["objective"]),
            -int(item["frequency"]),
        ),
    )

    chosen: list[np.ndarray] = []
    l1_norms: list[float] = []
    frequencies: list[int] = []
    sources: list[str] = []
    minimum_allowed = sqrt(1.0 / n_factors) / 3.0

    for item in ordered:
        direction = np.asarray(item["direction"], dtype=float)
        candidate = np.column_stack([*chosen, direction]) if chosen else direction[:, None]
        minimum_eigenvalue = float(np.linalg.eigvalsh(candidate.T @ candidate).min())
        previous_minimum = (
            float(np.linalg.eigvalsh(np.column_stack(chosen).T @ np.column_stack(chosen)).min())
            if chosen
            else 1.0
        )
        if (
            minimum_eigenvalue > minimum_allowed
            and minimum_eigenvalue > previous_minimum / 4.0
        ):
            chosen.append(direction)
            l1_norms.append(float(item["objective"]))
            frequencies.append(int(item["frequency"]))
            sources.append("local_minimum")
        if len(chosen) == n_factors:
            break

    strengths = (
        np.sqrt(np.maximum(np.asarray(eigenvalues)[:n_factors], 0.0))
        if eigenvalues is not None
        else np.ones(n_factors)
    )
    identity = np.eye(n_factors)
    while len(chosen) < n_factors:
        best_index = None
        best_score = -np.inf
        for index in range(n_factors):
            direction = identity[:, index]
            candidate = (
                np.column_stack([*chosen, direction])
                if chosen
                else direction[:, None]
            )
            minimum_eigenvalue = float(
                np.linalg.eigvalsh(candidate.T @ candidate).min()
            )
            score = minimum_eigenvalue * strengths[index]
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is None or best_score <= 1e-12:
            raise RuntimeError("Could not construct a nonsingular rotated basis.")
        direction = identity[:, best_index]
        chosen.append(direction)
        l1_norms.append(float(np.abs(initial_loadings @ direction).sum()))
        frequencies.append(0)
        sources.append("pc_fallback")

    rotation = np.column_stack(chosen)
    rotated = initial_loadings @ rotation
    for factor in range(n_factors):
        pivot = int(np.argmax(np.abs(rotated[:, factor])))
        if rotated[pivot, factor] < 0:
            rotated[:, factor] *= -1
            rotation[:, factor] *= -1

    return (
        rotation,
        np.asarray(l1_norms),
        np.asarray(frequencies, dtype=int),
        tuple(sources),
    )


def l1_rotate_basis(
    initial_loadings: pd.DataFrame | np.ndarray,
    *,
    n_starts: int | None = None,
    random_state: int = 916,
    max_iterations: int | None = None,
    cluster_tolerance: float = 0.05,
    minimum_frequency: float = 0.005,
    eigenvalues: np.ndarray | None = None,
) -> L1BasisRotation:
    """Find a sparse, nonsingular rotation of an orthonormal loading basis."""

    loadings = np.asarray(initial_loadings, dtype=float)
    if loadings.ndim != 2 or loadings.shape[1] < 2:
        raise ValueError("L1 rotation requires an n-by-r matrix with r >= 2.")
    if loadings.shape[0] < loadings.shape[1] or not np.isfinite(loadings).all():
        raise ValueError("The initial loading basis has invalid dimensions or values.")

    n_variables, n_factors = loadings.shape
    scaled_gram = loadings.T @ loadings / n_variables
    if not np.allclose(scaled_gram, np.eye(n_factors), atol=1e-6):
        raise ValueError(
            "Initial loadings must satisfy Lambda.T @ Lambda / n = I."
        )

    starts = _grid_size(n_factors) if n_starts is None else int(n_starts)
    if starts < n_factors:
        raise ValueError("n_starts must be at least the number of factors.")
    iterations = 200 * (n_factors - 1) if max_iterations is None else max_iterations
    generator = np.random.default_rng(random_state)
    initial_directions = generator.normal(size=(starts, n_factors))
    initial_directions /= np.linalg.norm(initial_directions, axis=1, keepdims=True)

    solutions = []
    successes = 0
    for initial_direction in initial_directions:
        initial_angles = cartesian_to_spherical(initial_direction)
        result = minimize(
            lambda angles: float(
                np.abs(loadings @ spherical_to_cartesian(angles)).sum()
            ),
            initial_angles,
            method="Nelder-Mead",
            options={
                "maxiter": iterations,
                "xatol": 1e-7,
                "fatol": 1e-7,
                "disp": False,
            },
        )
        direction = _canonical_direction(spherical_to_cartesian(result.x))
        objective = float(np.abs(loadings @ direction).sum())
        if np.isfinite(objective):
            solutions.append((objective, direction, bool(result.success)))
            successes += int(result.success)
    if not solutions:
        raise RuntimeError("Every L1-rotation optimization failed.")

    clusters: list[dict[str, object]] = []
    for objective, direction, success in sorted(solutions, key=lambda item: item[0]):
        matched = None
        for cluster in clusters:
            distance = np.linalg.norm(direction - cluster["direction"]) / sqrt(n_factors)
            if distance < cluster_tolerance:
                matched = cluster
                break
        if matched is None:
            clusters.append(
                {
                    "objective": objective,
                    "direction": direction,
                    "frequency": 1,
                    "successes": int(success),
                }
            )
        else:
            matched["frequency"] = int(matched["frequency"]) + 1
            matched["successes"] = int(matched["successes"]) + int(success)

    frequency_cutoff = max(1, ceil(minimum_frequency * len(solutions)))
    representatives = [
        cluster
        for cluster in clusters
        if int(cluster["frequency"]) >= frequency_cutoff
    ]
    if not representatives:
        representatives = [clusters[0]]

    rotation, l1_norms, frequencies, sources = _select_independent_directions(
        loadings,
        representatives,
        n_factors,
        eigenvalues,
    )
    rotated = loadings @ rotation
    test = local_factor_test(rotated)
    return L1BasisRotation(
        initial_loadings=loadings,
        rotated_loadings=rotated,
        rotation=rotation,
        l1_norms=l1_norms,
        small_counts=test.small_counts.to_numpy(),
        solution_frequencies=frequencies,
        sources=sources,
        optimizer_success_rate=successes / len(solutions),
    )


def fit_l1_rotation(
    pca_result: PCAResult,
    *,
    n_components: int = 3,
    n_starts: int | None = None,
    random_state: int = 916,
) -> L1RotationResult:
    """Rotate a PCA loading space and compute observationally equivalent scores."""

    if n_components < 2 or n_components > pca_result.eigenvectors.shape[1]:
        raise ValueError("Invalid number of components for L1 rotation.")
    n_variables = pca_result.eigenvectors.shape[0]
    initial = sqrt(n_variables) * pca_result.eigenvectors[:, :n_components]
    basis_result = l1_rotate_basis(
        initial,
        n_starts=n_starts,
        random_state=random_state,
        eigenvalues=pca_result.eigenvalues[:n_components],
    )

    labels = [f"LF{i}" for i in range(1, n_components + 1)]
    pc_labels = [f"PC{i}" for i in range(1, n_components + 1)]
    rotated_frame = pd.DataFrame(
        basis_result.rotated_loadings,
        index=pca_result.weights.index,
        columns=labels,
    )
    initial_frame = pd.DataFrame(
        initial,
        index=pca_result.weights.index,
        columns=pc_labels,
    )
    rotation_frame = pd.DataFrame(
        basis_result.rotation,
        index=pc_labels,
        columns=labels,
    )

    x = pca_result.analysis_data.to_numpy(dtype=float)
    loadings = basis_result.rotated_loadings
    score_values = x @ loadings @ np.linalg.inv(loadings.T @ loadings)
    scores = pd.DataFrame(score_values, index=pca_result.analysis_data.index, columns=labels)
    covariance = x.T @ score_values / (len(x) - 1)
    score_std = score_values.std(axis=0, ddof=1)
    score_loadings = pd.DataFrame(
        covariance / score_std[None, :],
        index=pca_result.weights.index,
        columns=labels,
    )

    reconstructed = score_values @ loadings.T
    reference = (
        x
        @ pca_result.eigenvectors[:, :n_components]
        @ pca_result.eigenvectors[:, :n_components].T
    )
    reconstruction_error = float(np.max(np.abs(reconstructed - reference)))
    reconstruction_pct = 100.0 * (
        1.0 - np.square(x - reconstructed).sum() / np.square(x).sum()
    )
    diagnostic = local_factor_test(rotated_frame)

    return L1RotationResult(
        initial_loadings=initial_frame,
        rotated_loadings=rotated_frame,
        rotation=rotation_frame,
        scores=scores,
        score_loadings=score_loadings,
        l1_norms=pd.Series(basis_result.l1_norms, index=labels, name="l1_norm"),
        solution_frequencies=pd.Series(
            basis_result.solution_frequencies,
            index=labels,
            name="solution_frequency",
        ),
        sources=pd.Series(basis_result.sources, index=labels, name="source"),
        local_factor_test=diagnostic,
        reconstruction_pct=float(reconstruction_pct),
        reconstruction_max_abs_error=reconstruction_error,
        score_correlation=scores.corr(),
        optimizer_success_rate=basis_result.optimizer_success_rate,
    )


def align_loading_columns(
    reference: pd.DataFrame | np.ndarray,
    estimate: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align an estimated loading matrix to a reference by permutation and sign."""

    reference_values = np.asarray(reference, dtype=float)
    estimate_values = np.asarray(estimate, dtype=float)
    if reference_values.shape != estimate_values.shape:
        raise ValueError("Reference and estimate loading matrices must have equal shape.")
    reference_norm = reference_values / np.linalg.norm(
        reference_values, axis=0, keepdims=True
    )
    estimate_norm = estimate_values / np.linalg.norm(
        estimate_values, axis=0, keepdims=True
    )
    similarity = reference_norm.T @ estimate_norm
    reference_index, estimate_index = linear_sum_assignment(-np.abs(similarity))
    order = estimate_index[np.argsort(reference_index)]
    aligned = estimate_values[:, order].copy()
    matched_similarity = np.empty(reference_values.shape[1])
    signs = np.ones(reference_values.shape[1])
    for factor, source in enumerate(order):
        signed_similarity = similarity[factor, source]
        signs[factor] = 1.0 if signed_similarity >= 0 else -1.0
        aligned[:, factor] *= signs[factor]
        matched_similarity[factor] = abs(signed_similarity)
    return aligned, order, matched_similarity
