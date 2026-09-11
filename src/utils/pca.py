"""Reusable PCA, Varimax, and Elastic-Net Sparse PCA utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PCAResult:
    """Container for one covariance or correlation PCA fit."""

    method: str
    analysis_data: pd.DataFrame
    matrix: pd.DataFrame
    means: pd.Series
    scales: pd.Series
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    explained: np.ndarray
    summary: pd.DataFrame
    weights: pd.DataFrame
    loadings: pd.DataFrame
    scores: pd.DataFrame


def pca_diagnostics(result: PCAResult) -> tuple[float, float]:
    """Check the estimated matrix and score variances against pandas."""
    matrix_error = float(
        (result.matrix - result.analysis_data.cov()).abs().to_numpy().max()
    )
    score_error = float(
        np.max(np.abs(result.eigenvalues - result.scores.var(ddof=1).to_numpy()))
    )
    return matrix_error, score_error


def reconstruction_by_stock(result: PCAResult, n_components: int = 3) -> pd.DataFrame:
    """Variance retained by a truncated PCA in the fitted analysis scale."""
    if not 1 <= n_components <= len(result.eigenvalues):
        raise ValueError("n_components must lie within the fitted PCA dimension.")
    reconstruction = (
        result.scores.iloc[:, :n_components].to_numpy()
        @ result.eigenvectors[:, :n_components].T
    )
    residuals = result.analysis_data - reconstruction
    summary = pd.DataFrame(
        {
            "original_variance": result.analysis_data.var(ddof=1),
            "residual_variance": residuals.var(ddof=1),
        }
    )
    summary["explained_pct"] = 100 * (
        1 - summary["residual_variance"] / summary["original_variance"]
    )
    return summary


@dataclass(frozen=True)
class VarimaxResult:
    """Container for an orthogonal rotation of selected PCA components."""

    rotated_loadings: pd.DataFrame
    rotated_weights: pd.DataFrame
    score_loadings: pd.DataFrame
    scores: pd.DataFrame
    rotation: pd.DataFrame
    iterations: int


@dataclass(frozen=True)
class SparsePCAResult:
    """Container for the Elastic-Net Sparse PCA fit."""

    l1_penalty: float
    l2_penalty: float
    penalized_coefficients: pd.DataFrame
    weights: pd.DataFrame
    score_loadings: pd.DataFrame
    scores: pd.DataFrame
    reconstruction_coefficients: pd.DataFrame
    reconstruction_pct: float
    score_correlation: pd.DataFrame
    weight_gram: pd.DataFrame
    iterations: int
    converged: bool


def _validate_data(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("PCA input must be a pandas DataFrame.")
    if data.empty or len(data) < 2:
        raise ValueError("PCA input must contain at least two observations.")
    if data.columns.has_duplicates:
        raise ValueError("PCA input contains duplicate column names.")
    if data.isna().any().any():
        raise ValueError("PCA input contains NaN values.")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("PCA input contains non-finite values.")


def orient_eigenvectors(vectors: np.ndarray) -> np.ndarray:
    """Apply a reproducible sign convention to eigenvector columns."""

    oriented = np.asarray(vectors, dtype=float).copy()
    for component in range(oriented.shape[1]):
        pivot = np.argmax(np.abs(oriented[:, component]))
        if oriented[pivot, component] < 0:
            oriented[:, component] *= -1
    return oriented


def fit_pca(data: pd.DataFrame, method: str = "covariance") -> PCAResult:
    """Fit covariance or correlation PCA and return a reusable result.

    ``weights`` are the normalized eigenvectors used to construct scores.
    ``loadings`` are the conventional eigenvector loadings, equal to
    ``weights * sqrt(eigenvalue)``.  For correlation PCA they are also the
    correlations between variables and principal-component scores.
    """

    _validate_data(data)
    if method not in {"covariance", "correlation"}:
        raise ValueError(f"Unknown PCA method: {method}")

    means = data.mean()
    centered = data.subtract(means, axis="columns")
    if method == "correlation":
        scales = centered.std(ddof=1)
        if scales.isna().any() or (scales <= 0).any():
            raise ValueError("Correlation PCA found a zero-variance variable.")
        analysis_data = centered.divide(scales, axis="columns")
    else:
        scales = pd.Series(1.0, index=data.columns)
        analysis_data = centered

    matrix = (analysis_data.T @ analysis_data) / (len(analysis_data) - 1)
    values, vectors = np.linalg.eigh(matrix.to_numpy(dtype=float))
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = orient_eigenvectors(vectors[:, order])

    if values[-1] < -1e-12:
        raise ValueError("PCA produced a materially negative eigenvalue.")
    values = np.maximum(values, 0.0)
    total_variance = values.sum()
    if total_variance <= 0:
        raise ValueError("PCA found zero total variance.")

    explained = values / total_variance
    labels = [f"PC{i}" for i in range(1, len(values) + 1)]
    summary = pd.DataFrame(
        {
            "eigenvalue": values,
            "explained_pct": explained * 100,
            "cumulative_pct": explained.cumsum() * 100,
        },
        index=labels,
    )
    weights = pd.DataFrame(vectors, index=data.columns, columns=labels)
    loadings = pd.DataFrame(
        vectors * np.sqrt(values)[None, :],
        index=data.columns,
        columns=labels,
    )
    scores = pd.DataFrame(
        analysis_data.to_numpy(dtype=float) @ vectors,
        index=data.index,
        columns=labels,
    )

    return PCAResult(
        method=method,
        analysis_data=analysis_data,
        matrix=matrix,
        means=means,
        scales=scales,
        eigenvalues=values,
        eigenvectors=vectors,
        explained=explained,
        summary=summary,
        weights=weights,
        loadings=loadings,
        scores=scores,
    )


def format_pca_summary(summary: pd.DataFrame) -> str:
    """Format a PCA summary for readable terminal output."""

    table = summary.copy()
    table["eigenvalue"] = table["eigenvalue"].map(lambda value: f"{value:.8e}")
    for column in ("explained_pct", "cumulative_pct"):
        table[column] = table[column].map(lambda value: f"{value:.4f}")
    return table.to_string()


def _correlation_loadings(
    data: pd.DataFrame,
    scores: pd.DataFrame | np.ndarray,
    columns: list[str],
) -> pd.DataFrame:
    """Return correlations between variables and arbitrary factor scores."""

    x = data.to_numpy(dtype=float)
    f = np.asarray(scores, dtype=float)
    covariance = (x.T @ f) / (len(data) - 1)
    score_std = np.std(f, axis=0, ddof=1)
    if (score_std <= 0).any():
        raise ValueError("A factor score has zero variance.")
    values = covariance / score_std[None, :]
    return pd.DataFrame(values, index=data.columns, columns=columns)


def varimax(
    loadings: pd.DataFrame | np.ndarray,
    gamma: float = 1.0,
    max_iterations: int = 1000,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Rotate a loading matrix toward a simpler, more interpretable structure."""

    phi = np.asarray(loadings, dtype=float)
    if phi.ndim != 2 or phi.shape[1] < 2:
        raise ValueError("Varimax requires a matrix with at least two factors.")
    if not np.isfinite(phi).all():
        raise ValueError("Varimax input contains non-finite values.")

    n_variables, n_factors = phi.shape
    rotation = np.eye(n_factors)
    previous_objective = -np.inf

    for iteration in range(1, max_iterations + 1):
        rotated = phi @ rotation
        squared = rotated**2
        diagonal = np.diag(squared.sum(axis=0))
        target = phi.T @ (rotated**3 - (gamma / n_variables) * rotated @ diagonal)
        left, singular_values, right_transpose = np.linalg.svd(
            target,
            full_matrices=False,
        )
        new_rotation = left @ right_transpose
        objective = singular_values.sum()
        rotation = new_rotation

        if iteration > 1 and abs(objective - previous_objective) <= tolerance * max(
            1.0, abs(previous_objective)
        ):
            break
        previous_objective = objective

    return phi @ rotation, rotation, iteration


def fit_varimax(
    pca_result: PCAResult,
    n_components: int = 3,
    gamma: float = 1.0,
) -> VarimaxResult:
    """Rotate the first PCA loadings and return scores plus diagnostics."""

    if n_components < 2 or n_components > pca_result.loadings.shape[1]:
        raise ValueError("Invalid number of components for Varimax.")

    selected_labels = [f"VF{i}" for i in range(1, n_components + 1)]
    base_loadings = pca_result.loadings.iloc[:, :n_components].to_numpy()
    rotated_loadings, rotation, iterations = varimax(
        base_loadings,
        gamma=gamma,
    )
    rotated_weights = pca_result.eigenvectors[:, :n_components] @ rotation

    # Keep signs consistent between rotated loadings, weights, and scores.
    for component in range(n_components):
        pivot = np.argmax(np.abs(rotated_loadings[:, component]))
        if rotated_loadings[pivot, component] < 0:
            rotated_loadings[:, component] *= -1
            rotated_weights[:, component] *= -1
            rotation[:, component] *= -1

    scores = pd.DataFrame(
        pca_result.analysis_data.to_numpy() @ rotated_weights,
        index=pca_result.analysis_data.index,
        columns=selected_labels,
    )
    return VarimaxResult(
        rotated_loadings=pd.DataFrame(
            rotated_loadings,
            index=pca_result.loadings.index,
            columns=selected_labels,
        ),
        rotated_weights=pd.DataFrame(
            rotated_weights,
            index=pca_result.weights.index,
            columns=selected_labels,
        ),
        score_loadings=_correlation_loadings(
            pca_result.analysis_data,
            scores,
            selected_labels,
        ),
        scores=scores,
        rotation=pd.DataFrame(
            rotation,
            index=[f"PC{i}" for i in range(1, n_components + 1)],
            columns=selected_labels,
        ),
        iterations=iterations,
    )


def _soft_threshold(value: float, penalty: float) -> float:
    return np.sign(value) * max(abs(value) - penalty, 0.0)


def _elastic_net_update(
    gram: np.ndarray,
    target: np.ndarray,
    l1_penalty: float,
    l2_penalty: float,
    initial: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Solve one small Elastic-Net coordinate-descent subproblem."""

    coefficients = initial.copy()
    target_cross = gram @ target
    diagonal = np.diag(gram)

    for _ in range(max_iterations):
        previous = coefficients.copy()
        for variable in range(len(coefficients)):
            partial = target_cross[variable] - (
                gram[variable] @ coefficients
                - diagonal[variable] * coefficients[variable]
            )
            coefficients[variable] = _soft_threshold(
                partial,
                l1_penalty,
            ) / (diagonal[variable] + l2_penalty)
        if np.max(np.abs(coefficients - previous)) <= tolerance:
            break
    return coefficients


def fit_elastic_net_sparse_pca(
    data: pd.DataFrame,
    n_components: int = 3,
    l1_penalty: float = 0.10,
    l2_penalty: float = 0.10,
    initial_pca: PCAResult | None = None,
    max_iterations: int = 1000,
    coordinate_max_iterations: int = 1000,
    tolerance: float = 1e-7,
    coordinate_tolerance: float = 1e-9,
    zero_tolerance: float = 1e-8,
) -> SparsePCAResult:
    """Fit Sparse PCA with an Elastic-Net penalty using alternating updates.

    The implementation follows the regression formulation of Zou, Hastie,
    and Tibshirani (2006).  ``l1_penalty`` creates exact zeros and
    ``l2_penalty`` stabilizes groups of correlated variables.
    """

    _validate_data(data)
    if n_components < 1 or n_components > data.shape[1]:
        raise ValueError("Invalid number of sparse components.")
    if l1_penalty < 0 or l2_penalty < 0:
        raise ValueError("Sparse PCA penalties must be non-negative.")

    if initial_pca is None:
        initial_pca = fit_pca(data, method="covariance")
    if initial_pca.analysis_data.shape != data.shape:
        raise ValueError("The initial PCA fit does not match the input data.")

    x = data.to_numpy(dtype=float)
    gram = (x.T @ x) / (len(data) - 1)
    a_matrix = initial_pca.eigenvectors[:, :n_components].copy()
    coefficients = a_matrix.copy()
    converged = False

    for iteration in range(1, max_iterations + 1):
        previous = coefficients.copy()
        for component in range(n_components):
            coefficients[:, component] = _elastic_net_update(
                gram=gram,
                target=a_matrix[:, component],
                l1_penalty=l1_penalty,
                l2_penalty=l2_penalty,
                initial=coefficients[:, component],
                max_iterations=coordinate_max_iterations,
                tolerance=coordinate_tolerance,
            )

        # Polar update: A remains orthonormal while B stays sparse.
        left, _, right_transpose = np.linalg.svd(
            gram @ coefficients,
            full_matrices=False,
        )
        a_matrix = left[:, :n_components] @ right_transpose[:n_components, :]

        if np.max(np.abs(coefficients - previous)) <= tolerance:
            converged = True
            break

    weights = coefficients.copy()
    for component in range(n_components):
        norm = np.linalg.norm(weights[:, component])
        if norm <= zero_tolerance:
            raise ValueError(
                "Sparse PCA produced an empty component; reduce the L1 penalty."
            )
        weights[:, component] /= norm

        pivot = np.argmax(np.abs(weights[:, component]))
        if weights[pivot, component] < 0:
            weights[:, component] *= -1
            coefficients[:, component] *= -1

    labels = [f"SPC{i}" for i in range(1, n_components + 1)]
    weight_frame = pd.DataFrame(weights, index=data.columns, columns=labels)
    coefficient_frame = pd.DataFrame(
        coefficients,
        index=data.columns,
        columns=labels,
    )
    scores = pd.DataFrame(
        x @ weights,
        index=data.index,
        columns=labels,
    )

    # Use least-squares reconstruction only to measure the information kept
    # by the sparse score space; this is not an economic regression model.
    reconstruction_coefficients = np.linalg.lstsq(
        scores.to_numpy(),
        x,
        rcond=None,
    )[0]
    reconstructed = scores.to_numpy() @ reconstruction_coefficients
    total_sum_of_squares = np.square(x).sum()
    reconstruction_error = np.square(x - reconstructed).sum()
    reconstruction_pct = 100 * (1 - reconstruction_error / total_sum_of_squares)

    return SparsePCAResult(
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
        penalized_coefficients=coefficient_frame,
        weights=weight_frame,
        score_loadings=_correlation_loadings(data, scores, labels),
        scores=scores,
        reconstruction_coefficients=pd.DataFrame(
            reconstruction_coefficients,
            index=labels,
            columns=data.columns,
        ),
        reconstruction_pct=float(reconstruction_pct),
        score_correlation=scores.corr(),
        weight_gram=pd.DataFrame(
            weights.T @ weights,
            index=labels,
            columns=labels,
        ),
        iterations=iteration,
        converged=converged,
    )
