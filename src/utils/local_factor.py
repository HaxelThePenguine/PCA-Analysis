"""L1 identification diagnostics and whole-session bootstrap calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import CORE_UNIVERSE
from utils.l1_rotation import (
    L1RotationResult,
    align_loading_columns,
    l1_rotate_basis,
    local_factor_test,
)
from utils.pca import PCAResult, orient_eigenvectors

STOCKS = list(CORE_UNIVERSE)
N_COMPONENTS = 3
PRIMARY_STARTS = 1000
SENSITIVITY_STARTS = 200
BOOTSTRAP_REPLICATIONS = 100
BOOTSTRAP_STARTS = 100
RANDOM_STATE = 916
MAX_FACTOR_COUNT = 5


@dataclass(frozen=True)
class SessionMoments:
    """Sufficient statistics for resampling full trading sessions."""

    labels: pd.Index
    counts: np.ndarray
    sums: np.ndarray
    cross_products: np.ndarray


def build_session_moments(data: pd.DataFrame) -> SessionMoments:
    """Compress each session into moments needed for a block bootstrap."""

    labels = []
    counts = []
    sums = []
    cross_products = []
    sessions = data.groupby(data.index.normalize(), sort=True)
    for session, frame in sessions:
        values = frame.to_numpy(dtype=float)
        labels.append(session)
        counts.append(len(values))
        sums.append(values.sum(axis=0))
        cross_products.append(values.T @ values)
    return SessionMoments(
        labels=pd.Index(labels, name="session"),
        counts=np.asarray(counts, dtype=int),
        sums=np.asarray(sums),
        cross_products=np.asarray(cross_products),
    )


def bootstrap_correlation(
    moments: SessionMoments,
    draw: np.ndarray,
) -> np.ndarray:
    """Build a correlation matrix from a with-replacement session draw."""

    weights = np.bincount(draw, minlength=len(moments.labels)).astype(float)
    observations = float(weights @ moments.counts)
    total = weights @ moments.sums
    cross = np.tensordot(weights, moments.cross_products, axes=(0, 0))
    covariance = (cross - np.outer(total, total) / observations) / (observations - 1.0)
    standard_deviation = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(standard_deviation, standard_deviation)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    return correlation


def correlation_eigendecomposition(
    correlation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return descending eigenvalues and reproducibly oriented eigenvectors."""

    values, vectors = np.linalg.eigh(correlation)
    order = np.argsort(values)[::-1]
    return np.maximum(values[order], 0.0), orient_eigenvectors(vectors[:, order])


def loading_rows(
    transformation: str,
    result: L1RotationResult,
) -> list[dict[str, Any]]:
    """Return initial and rotated structural loadings in tidy form."""

    rows = []
    matrices = (
        ("principal_component_basis", result.initial_loadings),
        ("l1_rotation", result.rotated_loadings),
    )
    for method, matrix in matrices:
        test = local_factor_test(matrix)
        for factor_number, factor in enumerate(matrix.columns, start=1):
            for stock in matrix.index:
                value = float(matrix.loc[stock, factor])
                rows.append(
                    {
                        "transformation": transformation,
                        "method": method,
                        "factor": factor,
                        "factor_number": factor_number,
                        "stock": stock,
                        "loading": value,
                        "absolute_loading": abs(value),
                        "small_loading_threshold": test.h_n,
                        "is_small": abs(value) < test.h_n,
                        "is_active": abs(value) >= test.h_n,
                    }
                )
    return rows


def diagnostic_rows(
    transformation: str,
    result: L1RotationResult,
) -> list[dict[str, Any]]:
    """Summarize the local-factor decision and rotation diagnostics."""

    initial_test = local_factor_test(result.initial_loadings)
    rotated_test = result.local_factor_test
    rows = []
    for factor_number, factor in enumerate(result.rotated_loadings.columns):
        rows.append(
            {
                "transformation": transformation,
                "factor": factor,
                "factor_number": factor_number + 1,
                "l1_norm": result.l1_norms.iloc[factor_number],
                "solution_frequency": result.solution_frequencies.iloc[factor_number],
                "candidate_source": result.sources.iloc[factor_number],
                "pc_small_count": initial_test.small_counts.iloc[factor_number],
                "rotated_small_count": rotated_test.small_counts.iloc[factor_number],
                "h_n": rotated_test.h_n,
                "gamma_n": rotated_test.gamma_n,
                "has_local_factors": rotated_test.has_local_factors,
                "optimizer_success_rate": result.optimizer_success_rate,
                "reconstruction_pct": result.reconstruction_pct,
                "reconstruction_max_abs_error": result.reconstruction_max_abs_error,
                "rotation_condition_number": np.linalg.cond(result.rotation.to_numpy()),
            }
        )
    return rows


def factor_count_rows(
    transformation: str,
    pca: PCAResult,
) -> list[dict[str, Any]]:
    """Report the eigenvalue-ratio diagnostic without treating it as certainty."""

    upper = min(MAX_FACTOR_COUNT, len(pca.eigenvalues) - 1)
    ratios = pca.eigenvalues[:upper] / pca.eigenvalues[1 : upper + 1]
    selected = int(np.argmax(ratios) + 1)
    rows = []
    for index, eigenvalue in enumerate(pca.eigenvalues, start=1):
        rows.append(
            {
                "transformation": transformation,
                "component": index,
                "eigenvalue": eigenvalue,
                "explained_pct": 100.0 * pca.explained[index - 1],
                "cumulative_pct": 100.0 * pca.explained[:index].sum(),
                "eigenvalue_ratio": ratios[index - 1] if index <= upper else np.nan,
                "ratio_search_upper_bound": upper,
                "eigenvalue_ratio_selection": selected,
                "is_primary_specification": index == N_COMPONENTS,
            }
        )
    return rows


def k_sensitivity_rows(pca: PCAResult) -> list[dict[str, Any]]:
    """Refit the residual loading space for K=2,...,5."""

    rows = []
    n_variables = pca.eigenvectors.shape[0]
    for n_factors in range(2, MAX_FACTOR_COUNT + 1):
        initial = np.sqrt(n_variables) * pca.eigenvectors[:, :n_factors]
        fit = l1_rotate_basis(
            initial,
            n_starts=SENSITIVITY_STARTS,
            random_state=RANDOM_STATE + 100 * n_factors,
            eigenvalues=pca.eigenvalues[:n_factors],
        )
        test = local_factor_test(fit.rotated_loadings)
        rows.append(
            {
                "n_factors": n_factors,
                "total_l1_norm": np.abs(fit.rotated_loadings).sum(),
                "minimum_small_count": test.small_counts.min(),
                "maximum_small_count": test.small_counts.max(),
                "gamma_n": test.gamma_n,
                "has_local_factors": test.has_local_factors,
                "pc_fallback_count": sum(
                    source == "pc_fallback" for source in fit.sources
                ),
                "rotation_condition_number": np.linalg.cond(fit.rotation),
                "optimizer_success_rate": fit.optimizer_success_rate,
                "cumulative_explained_pct": 100.0 * pca.explained[:n_factors].sum(),
            }
        )
    return rows


def bootstrap_local_factors(
    data: pd.DataFrame,
    reference: L1RotationResult,
    *,
    replications: int = BOOTSTRAP_REPLICATIONS,
    n_starts: int = BOOTSTRAP_STARTS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate loading support and direction stability by session bootstrap."""

    if replications < 1:
        raise ValueError("replications must be positive.")
    if list(data.columns) != list(reference.rotated_loadings.index):
        raise ValueError("Bootstrap data must match the reference stock order.")
    n_components = reference.rotated_loadings.shape[1]
    stocks = list(data.columns)
    moments = build_session_moments(data)
    generator = np.random.default_rng(random_state)
    reference_loadings = reference.rotated_loadings.to_numpy()
    reference_test = reference.local_factor_test
    active_counts = np.zeros_like(reference_loadings, dtype=float)
    stability_rows = []

    for replication in range(replications):
        draw = generator.integers(0, len(moments.labels), size=len(moments.labels))
        correlation = bootstrap_correlation(moments, draw)
        eigenvalues, eigenvectors = correlation_eigendecomposition(correlation)
        initial = np.sqrt(len(stocks)) * eigenvectors[:, :n_components]
        fit = l1_rotate_basis(
            initial,
            n_starts=n_starts,
            random_state=random_state + replication + 1,
            eigenvalues=eigenvalues[:n_components],
        )
        aligned, _, similarities = align_loading_columns(
            reference_loadings,
            fit.rotated_loadings,
        )
        active = np.abs(aligned) >= reference_test.h_n
        active_counts += active

        for factor in range(n_components):
            reference_support = (
                np.abs(reference_loadings[:, factor]) >= reference_test.h_n
            )
            intersection = np.logical_and(reference_support, active[:, factor]).sum()
            union = np.logical_or(reference_support, active[:, factor]).sum()
            jaccard = 1.0 if union == 0 else intersection / union
            stability_rows.append(
                {
                    "replication": replication + 1,
                    "factor": f"LF{factor + 1}",
                    "cosine_similarity": similarities[factor],
                    "support_jaccard": jaccard,
                    "active_count": int(active[:, factor].sum()),
                    "small_count": int((~active[:, factor]).sum()),
                    "has_local_factor": int((~active[:, factor]).sum())
                    > reference_test.gamma_n,
                    "rotation_condition_number": np.linalg.cond(fit.rotation),
                }
            )

    probability_rows = []
    probabilities = active_counts / replications
    for stock_index, stock in enumerate(stocks):
        for factor in range(n_components):
            probability_rows.append(
                {
                    "stock": stock,
                    "factor": f"LF{factor + 1}",
                    "reference_loading": reference_loadings[stock_index, factor],
                    "reference_active": abs(reference_loadings[stock_index, factor])
                    >= reference_test.h_n,
                    "active_probability": probabilities[stock_index, factor],
                    "small_probability": 1.0 - probabilities[stock_index, factor],
                    "small_loading_threshold": reference_test.h_n,
                    "bootstrap_replications": replications,
                }
            )
    return pd.DataFrame(probability_rows), pd.DataFrame(stability_rows)
