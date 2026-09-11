"""Identify local banking factors with the L1-rotation criterion.

This stage compares the raw and SPY/XLF-residualized loading spaces, applies
Freyaldenhoven's L1 rotation without coefficient shrinkage, and measures
support stability with a trading-session bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from benchmark_utils import residualize_against_benchmarks
from config import (
    BENCHMARKS,
    CORE_UNIVERSE,
    REPORTS_DIR,
    RETURN_MATRIX_CLEAN_FILE,
    ensure_project_directories,
)
from l1_rotation_utils import (
    L1RotationResult,
    align_loading_columns,
    fit_l1_rotation,
    l1_rotate_basis,
    local_factor_test,
)
from pca_utils import PCAResult, fit_pca, orient_eigenvectors


OUT_DIR = REPORTS_DIR / "local_factor_identification"
STOCKS = list(CORE_UNIVERSE)
N_COMPONENTS = 3
PRIMARY_STARTS = 1000
SENSITIVITY_STARTS = 200
BOOTSTRAP_REPLICATIONS = 100
BOOTSTRAP_STARTS = 100
RANDOM_STATE = 916
MAX_FACTOR_COUNT = 5
GRID_COLOR = "#D9DEE5"
COLORS = {"raw": "#2F6690", "benchmark_residual": "#C26A2E"}


@dataclass(frozen=True)
class SessionMoments:
    """Sufficient statistics for resampling full trading sessions."""

    labels: pd.Index
    counts: np.ndarray
    sums: np.ndarray
    cross_products: np.ndarray


def style_axis(axis: plt.Axes) -> None:
    """Apply the restrained visual style used by the project."""

    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Save a high-resolution figure and release its resources."""

    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def load_complete_panel() -> pd.DataFrame:
    """Load the synchronized CORE stocks and benchmark returns."""

    required = STOCKS + list(BENCHMARKS)
    frame = pd.read_parquet(RETURN_MATRIX_CLEAN_FILE)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"L1 local-factor panel is missing columns: {missing}")
    panel = frame.loc[:, required].dropna(how="any")
    if panel.empty or panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError("L1 local-factor panel has invalid index structure.")
    if not np.isfinite(panel.to_numpy(dtype=float)).all():
        raise ValueError("L1 local-factor panel contains non-finite values.")
    return panel


def build_factor_panels(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return raw stock returns and benchmark-residualized returns."""

    residualization = residualize_against_benchmarks(panel, stocks=STOCKS)
    return {
        "raw": panel.loc[:, STOCKS],
        "benchmark_residual": residualization["residual_returns"],
    }


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
    covariance = (cross - np.outer(total, total) / observations) / (
        observations - 1.0
    )
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
                "rotation_condition_number": np.linalg.cond(
                    result.rotation.to_numpy()
                ),
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate loading support and direction stability by session bootstrap."""

    moments = build_session_moments(data)
    generator = np.random.default_rng(RANDOM_STATE)
    reference_loadings = reference.rotated_loadings.to_numpy()
    reference_test = reference.local_factor_test
    active_counts = np.zeros_like(reference_loadings, dtype=float)
    stability_rows = []

    for replication in range(BOOTSTRAP_REPLICATIONS):
        draw = generator.integers(0, len(moments.labels), size=len(moments.labels))
        correlation = bootstrap_correlation(moments, draw)
        eigenvalues, eigenvectors = correlation_eigendecomposition(correlation)
        initial = np.sqrt(len(STOCKS)) * eigenvectors[:, :N_COMPONENTS]
        fit = l1_rotate_basis(
            initial,
            n_starts=BOOTSTRAP_STARTS,
            random_state=RANDOM_STATE + replication + 1,
            eigenvalues=eigenvalues[:N_COMPONENTS],
        )
        aligned, _, similarities = align_loading_columns(
            reference_loadings,
            fit.rotated_loadings,
        )
        active = np.abs(aligned) >= reference_test.h_n
        active_counts += active

        for factor in range(N_COMPONENTS):
            reference_support = np.abs(reference_loadings[:, factor]) >= reference_test.h_n
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
    probabilities = active_counts / BOOTSTRAP_REPLICATIONS
    for stock_index, stock in enumerate(STOCKS):
        for factor in range(N_COMPONENTS):
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
                    "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
                }
            )
    return pd.DataFrame(probability_rows), pd.DataFrame(stability_rows)


def plot_loading_comparison(loadings: pd.DataFrame) -> None:
    """Compare principal-component and L1-rotated structural loadings."""

    panels = []
    maximum = 0.0
    for transformation in ("raw", "benchmark_residual"):
        for method in ("principal_component_basis", "l1_rotation"):
            subset = loadings[
                (loadings["transformation"] == transformation)
                & (loadings["method"] == method)
            ]
            pivot = subset.pivot(
                index="stock", columns="factor_number", values="loading"
            ).reindex(STOCKS)
            panels.append((transformation, method, pivot))
            maximum = max(maximum, float(np.abs(pivot.to_numpy()).max()))

    figure, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    norm_scale = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    image = None
    for axis, (transformation, method, pivot) in zip(axes.ravel(), panels):
        image = axis.imshow(pivot.to_numpy(), cmap="RdBu_r", norm=norm_scale, aspect="auto")
        axis.set_title(
            f"{transformation.replace('_', ' ').title()}\n"
            f"{'PC basis' if method == 'principal_component_basis' else 'L1 rotation'}"
        )
        axis.set_xticks(range(N_COMPONENTS), [f"F{i}" for i in range(1, N_COMPONENTS + 1)])
        axis.set_yticks(range(len(STOCKS)), STOCKS)
        axis.tick_params(axis="both", length=0)
        for row in range(len(STOCKS)):
            for column in range(N_COMPONENTS):
                axis.text(
                    column,
                    row,
                    f"{pivot.iloc[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#252525",
                )
        for spine in axis.spines.values():
            spine.set_visible(False)
    colorbar_axis = figure.add_axes([0.86, 0.16, 0.022, 0.68])
    figure.colorbar(image, cax=colorbar_axis, label="Structural loading")
    figure.suptitle("L1 rotation reveals local loading directions", x=0.06, ha="left", y=0.99)
    figure.text(
        0.06,
        0.95,
        "Columns have norm sqrt(n); L1 rotation changes coordinates, not the selected PCA subspace.",
        fontsize=9,
        color="#666666",
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.82,
        bottom=0.07,
        top=0.87,
        wspace=0.22,
        hspace=0.28,
    )
    save_figure(figure, OUT_DIR / "13_l1_loading_comparison.png")


def plot_bootstrap_support(probabilities: pd.DataFrame) -> None:
    """Plot session-bootstrap support probabilities for residual local factors."""

    pivot = probabilities.pivot(
        index="stock", columns="factor", values="active_probability"
    ).reindex(STOCKS)
    figure, axis = plt.subplots(figsize=(7.2, 7.0))
    image = axis.imshow(pivot.to_numpy(), cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(N_COMPONENTS), pivot.columns)
    axis.set_yticks(range(len(STOCKS)), STOCKS)
    axis.tick_params(axis="both", length=0)
    for row in range(len(STOCKS)):
        for column in range(N_COMPONENTS):
            probability = pivot.iloc[row, column]
            axis.text(
                column,
                row,
                f"{probability:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if probability > 0.58 else "#252525",
            )
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03, label="Active-loading probability")
    figure.suptitle("Residual local-factor support stability", x=0.08, ha="left", y=0.98)
    figure.text(
        0.08,
        0.93,
        f"{BOOTSTRAP_REPLICATIONS} whole-session bootstrap replications; factors aligned by cosine similarity.",
        fontsize=9,
        color="#666666",
    )
    figure.subplots_adjust(left=0.13, right=0.88, bottom=0.08, top=0.87)
    save_figure(figure, OUT_DIR / "13_bootstrap_support_probability.png")


def plot_factor_diagnostics(
    factor_counts: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    """Plot eigenvalue-ratio context and bootstrap factor-direction stability."""

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for transformation, color in COLORS.items():
        subset = factor_counts[factor_counts["transformation"] == transformation]
        axes[0].plot(
            subset["component"],
            subset["explained_pct"],
            marker="o",
            color=color,
            label=transformation.replace("_", " ").title(),
        )
    axes[0].axvline(N_COMPONENTS, color="#777777", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Factor-count context")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    groups = [
        stability.loc[stability["factor"] == factor, "cosine_similarity"].to_numpy()
        for factor in [f"LF{i}" for i in range(1, N_COMPONENTS + 1)]
    ]
    box = axes[1].boxplot(groups, patch_artist=True, tick_labels=["LF1", "LF2", "LF3"])
    for patch in box["boxes"]:
        patch.set_facecolor("#9EC5E5")
        patch.set_edgecolor("#2F6690")
    axes[1].axhline(0.90, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("Absolute loading cosine")
    axes[1].set_title("Session-bootstrap direction stability")
    style_axis(axes[1])
    figure.suptitle("Identification diagnostics", x=0.06, ha="left", y=0.98)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.83, wspace=0.25)
    save_figure(figure, OUT_DIR / "13_identification_diagnostics.png")


def main() -> None:
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_complete_panel()
    factor_panels = build_factor_panels(panel)
    fits: dict[str, L1RotationResult] = {}
    pca_fits: dict[str, PCAResult] = {}
    loading_records = []
    diagnostic_records = []
    factor_count_records = []
    score_frames = []
    score_loading_frames = []
    score_correlation_frames = []
    rotation_frames = []

    for offset, (transformation, data) in enumerate(factor_panels.items()):
        pca = fit_pca(data, method="correlation")
        result = fit_l1_rotation(
            pca,
            n_components=N_COMPONENTS,
            n_starts=PRIMARY_STARTS,
            random_state=RANDOM_STATE + offset,
        )
        pca_fits[transformation] = pca
        fits[transformation] = result
        loading_records.extend(loading_rows(transformation, result))
        diagnostic_records.extend(diagnostic_rows(transformation, result))
        factor_count_records.extend(factor_count_rows(transformation, pca))

        scores = result.scores.add_prefix(f"{transformation}_")
        score_frames.append(scores)
        score_loadings = result.score_loadings.copy()
        score_loadings.insert(0, "transformation", transformation)
        score_loading_frames.append(score_loadings.reset_index(names="stock"))
        score_correlations = (
            result.score_correlation.rename_axis("factor")
            .reset_index()
            .melt(
                id_vars="factor",
                var_name="other_factor",
                value_name="correlation",
            )
        )
        score_correlations.insert(0, "transformation", transformation)
        score_correlation_frames.append(score_correlations)
        rotation = result.rotation.copy()
        rotation.insert(0, "transformation", transformation)
        rotation_frames.append(rotation.reset_index(names="principal_component"))

    probabilities, stability = bootstrap_local_factors(
        factor_panels["benchmark_residual"],
        fits["benchmark_residual"],
    )
    loadings = pd.DataFrame(loading_records)
    diagnostics = pd.DataFrame(diagnostic_records)
    factor_counts = pd.DataFrame(factor_count_records)
    k_sensitivity = pd.DataFrame(
        k_sensitivity_rows(pca_fits["benchmark_residual"])
    )

    loadings.to_csv(OUT_DIR / "13_l1_loadings.csv", index=False)
    diagnostics.to_csv(OUT_DIR / "13_rotation_diagnostics.csv", index=False)
    factor_counts.to_csv(OUT_DIR / "13_factor_count_diagnostics.csv", index=False)
    k_sensitivity.to_csv(OUT_DIR / "13_k_sensitivity.csv", index=False)
    probabilities.to_csv(OUT_DIR / "13_bootstrap_support_probability.csv", index=False)
    stability.to_csv(OUT_DIR / "13_bootstrap_factor_stability.csv", index=False)
    pd.concat(rotation_frames, ignore_index=True).to_csv(
        OUT_DIR / "13_rotation_matrices.csv", index=False
    )
    pd.concat(score_loading_frames, ignore_index=True).to_csv(
        OUT_DIR / "13_factor_score_loadings.csv", index=False
    )
    pd.concat(score_correlation_frames, ignore_index=True).to_csv(
        OUT_DIR / "13_factor_score_correlations.csv", index=False
    )
    pd.concat(score_frames, axis=1).to_parquet(OUT_DIR / "13_factor_scores.parquet")

    plot_loading_comparison(loadings)
    plot_bootstrap_support(probabilities)
    plot_factor_diagnostics(factor_counts, stability)

    print("=== L1 LOCAL FACTOR IDENTIFICATION ===")
    print(f"Complete panel: {panel.shape}")
    print(f"Period: {panel.index[0]} -> {panel.index[-1]}")
    print(
        f"Primary K={N_COMPONENTS}; starts={PRIMARY_STARTS}; "
        f"session bootstraps={BOOTSTRAP_REPLICATIONS}"
    )
    for transformation, result in fits.items():
        test = result.local_factor_test
        print(f"\n--- {transformation.replace('_', ' ').title()} ---")
        print(
            f"Local factor detected: {test.has_local_factors} | "
            f"small counts={test.small_counts.tolist()} | critical count>{test.gamma_n}"
        )
        print(f"L1 norms: {result.l1_norms.round(4).tolist()}")
        print(
            f"PCA-subspace reconstruction: {result.reconstruction_pct:.4f}% | "
            f"max equivalence error={result.reconstruction_max_abs_error:.3e}"
        )
        print("Rotated structural loadings:")
        print(result.rotated_loadings.round(3).to_string())

    summary = stability.groupby("factor").agg(
        median_cosine=("cosine_similarity", "median"),
        fifth_pct_cosine=("cosine_similarity", lambda values: values.quantile(0.05)),
        median_jaccard=("support_jaccard", "median"),
        local_detection_rate=("has_local_factor", "mean"),
    )
    print("\n--- Residual session-bootstrap stability ---")
    print(summary.round(3).to_string())
    print("\nK sensitivity:")
    print(k_sensitivity.round(4).to_string(index=False))
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
