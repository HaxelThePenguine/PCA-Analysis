"""Rolling stability and regime diagnostics for the L1 local-factor stage.

The module keeps the rolling stage separate from the numbered entry point so
that its pure helpers can be tested without running the real-data analysis.
Each window re-estimates SPY/XLF residuals, correlation PCA, and L1 rotation
using only observations in that trailing window. Factor labels are aligned in
two distinct ways: a past-only anchor for stability and an ex-post full-sample
anchor for descriptive plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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
from data_utils import load_panel, validate_panel
from l1_rotation_utils import (
    L1RotationResult,
    align_loading_columns,
    fit_l1_rotation,
)
from pca_utils import fit_pca
from plotting_utils import save_figure, style_axis
from rolling_utils import rolling_starts, session_index, session_window_positions


OUT_DIR = REPORTS_DIR / "dynamic_local_factor_regimes"
STOCKS = list(CORE_UNIVERSE)
FACTORS = [f"LF{i}" for i in range(1, 4)]
WINDOWS = (60, 120)
STEP_SESSIONS = 5
N_COMPONENTS = 3
PRIMARY_STARTS = 200
SENSITIVITY_STARTS = 500
FULL_SAMPLE_STARTS = 1000
RANDOM_STATE = 14014
FULL_SAMPLE_RANDOM_STATE = 917
COSINE_INSTABILITY_THRESHOLD = 0.80
JACCARD_INSTABILITY_THRESHOLD = 0.50
CONDITION_NUMBER_THRESHOLD = 10.0
BANKING_CRISIS_START = pd.Timestamp("2023-03-01")
BANKING_CRISIS_END = pd.Timestamp("2023-05-31")
RECENT_PERIOD_START = pd.Timestamp("2026-01-01")
CRISIS_SHADE_START = pd.Timestamp("2023-03-01")
CRISIS_SHADE_END = pd.Timestamp("2023-06-01")
COLORS = {"LF1": "#2F6690", "LF2": "#D99A2B", "LF3": "#C26A2E"}
GRID_COLOR = "#D9DEE5"


@dataclass(frozen=True)
class WindowDefinition:
    """Metadata and row positions for one trailing session window."""

    window_id: str
    window_sessions: int
    start_index: int
    end_index: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    positions: np.ndarray
    n_observations: int
    regime: str
    regime_sequence: str
    crosses_regime_boundary: bool
    is_quarterly_window: bool


@dataclass(frozen=True)
class WindowFit:
    """Small, serializable summary of one window's numerical fit."""

    definition: WindowDefinition
    structural_loadings: pd.DataFrame
    score_loadings: pd.DataFrame
    score_correlation: pd.DataFrame
    eigenvalues: np.ndarray
    explained: np.ndarray
    small_loading_threshold: float
    gamma_n: int
    small_counts: np.ndarray
    local_factor_flags: np.ndarray
    has_any_local_factors: bool
    l1_norms: np.ndarray
    solution_frequencies: np.ndarray
    sources: tuple[str, ...]
    optimizer_success_rate: float
    rotation_condition_number: float
    reconstruction_pct: float
    reconstruction_max_abs_error: float
    benchmark_variance_removed_pct: float
    max_abs_residual_benchmark_corr: float
    n_starts: int
    random_seed: int


class PastOnlyAnchor:
    """Causal average of previously aligned loading directions."""

    def __init__(self) -> None:
        self._unit_sum: np.ndarray | None = None
        self._norm_sum: np.ndarray | None = None
        self.count = 0

    def matrix(self) -> np.ndarray | None:
        """Return the average loading direction available before this window."""

        if self.count == 0 or self._unit_sum is None or self._norm_sum is None:
            return None
        mean_unit = self._unit_sum / self.count
        norms = self._norm_sum / self.count
        unit_norms = np.linalg.norm(mean_unit, axis=0)
        if (unit_norms <= 1e-12).any():
            raise RuntimeError("Past-only factor anchor lost a material direction.")
        return mean_unit / unit_norms[None, :] * norms[None, :]

    def update(self, loadings: pd.DataFrame | np.ndarray) -> None:
        """Add the current aligned loadings after all current metrics are saved."""

        values = np.asarray(loadings, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Past-only anchor update requires finite 2D loadings.")
        norms = np.linalg.norm(values, axis=0)
        if (norms <= 1e-12).any():
            raise ValueError("Past-only anchor cannot contain zero loading columns.")
        unit = values / norms[None, :]
        if self._unit_sum is None:
            self._unit_sum = unit.copy()
            self._norm_sum = norms.copy()
        else:
            self._unit_sum += unit
            self._norm_sum += norms
        self.count += 1


@dataclass(frozen=True)
class AlignmentResult:
    """Aligned structural and score-loading matrices plus permutation metadata."""

    structural_loadings: pd.DataFrame
    score_loadings: pd.DataFrame
    score_correlation: pd.DataFrame
    source_factors: tuple[str, ...]
    order: np.ndarray
    signs: np.ndarray
    cosine_similarity: np.ndarray


def deterministic_seed(
    window_sessions: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    n_starts: int,
    base_seed: int = RANDOM_STATE,
) -> int:
    """Derive a reproducible 32-bit seed from window identity and start count."""

    token = "|".join(
        (
            str(base_seed),
            str(window_sessions),
            pd.Timestamp(window_start).date().isoformat(),
            pd.Timestamp(window_end).date().isoformat(),
            str(n_starts),
        )
    )
    digest = sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little") % 2_147_483_647


def _date_label(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


def _classify_session(value: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value).tz_localize(None)
    if timestamp < BANKING_CRISIS_START:
        return "pre_banking_crisis"
    if timestamp <= BANKING_CRISIS_END:
        return "banking_crisis_mar_may_2023"
    if timestamp < RECENT_PERIOD_START:
        return "post_banking_crisis"
    return "recent_2026"


def classify_window_regime(
    window_sessions: Iterable[pd.Timestamp],
) -> tuple[str, str, bool]:
    """Classify a window by majority regime and expose mixed-regime windows."""

    labels = [_classify_session(session) for session in window_sessions]
    if not labels:
        raise ValueError("At least one session is required for regime classification.")
    counts = pd.Series(labels).value_counts()
    tied = counts[counts == counts.max()].index.tolist()
    regime = labels[-1] if len(tied) > 1 and labels[-1] in tied else tied[0]
    sequence = list(dict.fromkeys(labels))
    return regime, ";".join(sequence), len(sequence) > 1


def is_quarterly_window(window_end: pd.Timestamp) -> bool:
    """Flag late-March, late-June, late-September, and late-December windows."""

    timestamp = pd.Timestamp(window_end)
    return timestamp.month in {3, 6, 9, 12} and timestamp.day >= 20


def build_window_definitions(
    panel: pd.DataFrame,
    window_sizes: Iterable[int] = WINDOWS,
    step_sessions: int = STEP_SESSIONS,
) -> list[WindowDefinition]:
    """Build exact-session trailing windows from the observations in ``panel``."""

    validate_panel(panel, context="Dynamic local-factor panel", require_complete=True)
    sessions, codes = session_index(panel.index)
    definitions: list[WindowDefinition] = []
    for window_size in window_sizes:
        for start in rolling_starts(len(sessions), window_size, step_sessions):
            end = start + window_size - 1
            positions = session_window_positions(codes, start, window_size)
            window_sessions = sessions[start : end + 1]
            if len(window_sessions) != window_size:
                raise AssertionError("A rolling window did not contain the requested sessions.")
            regime, sequence, crosses = classify_window_regime(window_sessions)
            window_start = pd.Timestamp(window_sessions[0])
            window_end = pd.Timestamp(window_sessions[-1])
            window_id = f"{window_size}_{_date_label(window_end)}"
            definitions.append(
                WindowDefinition(
                    window_id=window_id,
                    window_sessions=window_size,
                    start_index=start,
                    end_index=end,
                    window_start=window_start,
                    window_end=window_end,
                    positions=positions,
                    n_observations=len(positions),
                    regime=regime,
                    regime_sequence=sequence,
                    crosses_regime_boundary=crosses,
                    is_quarterly_window=is_quarterly_window(window_end),
                )
            )
    return definitions


def _factor_score_correlation_maximum(correlation: pd.DataFrame) -> float:
    """Return the largest absolute off-diagonal score correlation."""

    values = correlation.to_numpy(dtype=float)
    upper = values[np.triu_indices(values.shape[0], k=1)]
    return float(np.max(np.abs(upper))) if len(upper) else np.nan


def _eigenvalue_ratios(eigenvalues: np.ndarray) -> tuple[float, float, float]:
    """Return the first three adjacent eigenvalue ratios."""

    values = np.asarray(eigenvalues, dtype=float)
    ratios = values[:3] / values[1:4]
    return tuple(float(value) for value in ratios)


def fit_window(
    panel: pd.DataFrame,
    definition: WindowDefinition,
    *,
    n_starts: int,
    random_seed: int,
) -> WindowFit:
    """Fit benchmark residuals, correlation PCA, and L1 rotation in one window."""

    window_panel = panel.iloc[definition.positions]
    if window_panel.index.normalize().max() > definition.window_end:
        raise AssertionError("A rolling fit received an observation after window_end.")
    residualization = residualize_against_benchmarks(window_panel, stocks=STOCKS)
    residuals = residualization["residual_returns"]
    pca = fit_pca(residuals, method="correlation")
    rotation = fit_l1_rotation(
        pca,
        n_components=N_COMPONENTS,
        n_starts=n_starts,
        random_state=random_seed,
    )
    local_test = rotation.local_factor_test
    condition_number = float(np.linalg.cond(rotation.rotation.to_numpy()))
    max_residual_corr = float(
        residualization["diagnostics"][
            [f"corr_resid_{benchmark}" for benchmark in BENCHMARKS]
        ]
        .abs()
        .to_numpy()
        .max()
    )
    return WindowFit(
        definition=definition,
        structural_loadings=rotation.rotated_loadings.copy(),
        score_loadings=rotation.score_loadings.copy(),
        score_correlation=rotation.score_correlation.copy(),
        eigenvalues=pca.eigenvalues.copy(),
        explained=pca.explained.copy(),
        small_loading_threshold=float(local_test.h_n),
        gamma_n=int(local_test.gamma_n),
        small_counts=local_test.small_counts.to_numpy(dtype=int),
        local_factor_flags=(
            local_test.small_counts.to_numpy(dtype=int) > int(local_test.gamma_n)
        ),
        has_any_local_factors=bool(local_test.has_local_factors),
        l1_norms=rotation.l1_norms.to_numpy(dtype=float),
        solution_frequencies=rotation.solution_frequencies.to_numpy(dtype=int),
        sources=tuple(rotation.sources.astype(str).tolist()),
        optimizer_success_rate=float(rotation.optimizer_success_rate),
        rotation_condition_number=condition_number,
        reconstruction_pct=float(rotation.reconstruction_pct),
        reconstruction_max_abs_error=float(rotation.reconstruction_max_abs_error),
        benchmark_variance_removed_pct=float(
            residualization["diagnostics"]["variance_removed_pct"].mean()
        ),
        max_abs_residual_benchmark_corr=max_residual_corr,
        n_starts=int(n_starts),
        random_seed=int(random_seed),
    )


def fit_full_sample_reference(
    panel: pd.DataFrame,
    *,
    n_starts: int = FULL_SAMPLE_STARTS,
    random_seed: int = FULL_SAMPLE_RANDOM_STATE,
) -> pd.DataFrame:
    """Fit the ex-post full-sample residual reference used only for plots."""

    residualization = residualize_against_benchmarks(panel, stocks=STOCKS)
    pca = fit_pca(residualization["residual_returns"], method="correlation")
    result = fit_l1_rotation(
        pca,
        n_components=N_COMPONENTS,
        n_starts=n_starts,
        random_state=random_seed,
    )
    return result.rotated_loadings.copy()


def _column_signs(
    reference: np.ndarray,
    estimate: np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    """Recover the signs used by ``align_loading_columns`` for other matrices."""

    signs = np.ones(reference.shape[1], dtype=float)
    for factor, source in enumerate(order):
        signed_dot = float(reference[:, factor] @ estimate[:, source])
        signs[factor] = 1.0 if signed_dot >= 0 else -1.0
    return signs


def _align_fit(
    fit: WindowFit,
    reference: pd.DataFrame,
) -> AlignmentResult:
    """Align structural loadings and apply the same permutation/sign to scores."""

    reference_values = reference.to_numpy(dtype=float)
    estimate_values = fit.structural_loadings.to_numpy(dtype=float)
    aligned_values, order, similarities = align_loading_columns(
        reference_values,
        estimate_values,
    )
    signs = _column_signs(reference_values, estimate_values, order)
    score_values = fit.score_loadings.to_numpy(dtype=float)[:, order] * signs[None, :]
    source_factors = tuple(fit.structural_loadings.columns[index] for index in order)
    score_correlation = fit.score_correlation.to_numpy(dtype=float)
    score_correlation = score_correlation[np.ix_(order, order)]
    score_correlation = score_correlation * signs[:, None] * signs[None, :]
    return AlignmentResult(
        structural_loadings=pd.DataFrame(
            aligned_values,
            index=fit.structural_loadings.index,
            columns=FACTORS,
        ),
        score_loadings=pd.DataFrame(
            score_values,
            index=fit.score_loadings.index,
            columns=FACTORS,
        ),
        score_correlation=pd.DataFrame(
            score_correlation,
            index=FACTORS,
            columns=FACTORS,
        ),
        source_factors=source_factors,
        order=order,
        signs=signs,
        cosine_similarity=similarities,
    )


def _same_factor_cosines(
    current: pd.DataFrame,
    previous: pd.DataFrame | None,
) -> np.ndarray:
    """Return sign-invariant same-column cosines for aligned loading matrices."""

    if previous is None:
        return np.full(current.shape[1], np.nan)
    current_values = current.to_numpy(dtype=float)
    previous_values = previous.to_numpy(dtype=float)
    numerator = np.sum(current_values * previous_values, axis=0)
    denominator = np.linalg.norm(current_values, axis=0) * np.linalg.norm(
        previous_values,
        axis=0,
    )
    return np.abs(numerator / denominator)


def _jaccard_support(
    current: np.ndarray,
    anchor: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Compute support Jaccard similarity for each aligned factor."""

    current_active = np.abs(current) >= threshold
    anchor_active = np.abs(anchor) >= threshold
    scores = []
    for factor in range(current.shape[1]):
        intersection = np.logical_and(
            current_active[:, factor],
            anchor_active[:, factor],
        ).sum()
        union = np.logical_or(
            current_active[:, factor],
            anchor_active[:, factor],
        ).sum()
        scores.append(1.0 if union == 0 else float(intersection / union))
    return np.asarray(scores)


def _factor_metadata(fit: WindowFit) -> dict[str, Any]:
    """Return common fit metrics repeated in factor-level output tables."""

    ratios = _eigenvalue_ratios(fit.eigenvalues)
    return {
        "n_sessions": fit.definition.window_sessions,
        "n_observations": fit.definition.n_observations,
        "pc1_explained_pct": 100.0 * fit.explained[0],
        "cumulative_explained_pct_first3": 100.0 * fit.explained[:3].sum(),
        "eigenvalue_ratio_pc1_pc2": ratios[0],
        "eigenvalue_ratio_pc2_pc3": ratios[1],
        "eigenvalue_ratio_pc3_pc4": ratios[2],
        "benchmark_variance_removed_pct": fit.benchmark_variance_removed_pct,
        "max_abs_residual_benchmark_corr": fit.max_abs_residual_benchmark_corr,
        "has_any_local_factors": fit.has_any_local_factors,
        "reconstruction_pct": fit.reconstruction_pct,
        "reconstruction_max_abs_error": fit.reconstruction_max_abs_error,
        "n_starts": fit.n_starts,
        "random_seed": fit.random_seed,
        "optimizer_success_rate": fit.optimizer_success_rate,
        "rotation_condition_number": fit.rotation_condition_number,
        "max_abs_factor_score_correlation": _factor_score_correlation_maximum(
            fit.score_correlation
        ),
    }


def _base_window_metadata(definition: WindowDefinition) -> dict[str, Any]:
    """Return window metadata shared by all output tables."""

    return {
        "window_id": definition.window_id,
        "window_sessions": definition.window_sessions,
        "window_start": _date_label(definition.window_start),
        "window_end": _date_label(definition.window_end),
        "window_start_index": definition.start_index,
        "window_end_index": definition.end_index,
        "regime": definition.regime,
        "regime_sequence": definition.regime_sequence,
        "crosses_regime_boundary": definition.crosses_regime_boundary,
        "is_quarterly_window": definition.is_quarterly_window,
    }


def _instability_reasons(
    previous_cosine: float,
    anchor_cosine: float,
    anchor_jaccard: float,
    local_changed: bool,
    condition_number: float,
) -> list[str]:
    """Return descriptive flags without calling them formal break tests."""

    reasons: list[str] = []
    if np.isfinite(previous_cosine) and previous_cosine < COSINE_INSTABILITY_THRESHOLD:
        reasons.append("low_previous_cosine")
    if anchor_cosine < COSINE_INSTABILITY_THRESHOLD:
        reasons.append("low_anchor_cosine")
    if anchor_jaccard < JACCARD_INSTABILITY_THRESHOLD:
        reasons.append("low_anchor_jaccard")
    if local_changed:
        reasons.append("locality_decision_change")
    if condition_number > CONDITION_NUMBER_THRESHOLD:
        reasons.append("high_rotation_condition_number")
    return reasons


def _aligned_support_flags(
    loadings: pd.DataFrame,
    threshold: float,
) -> np.ndarray:
    return np.abs(loadings.to_numpy(dtype=float)) >= threshold


def run_rolling_fits(
    panel: pd.DataFrame,
    definitions: list[WindowDefinition],
    *,
    primary_starts: int = PRIMARY_STARTS,
    full_sample_reference: pd.DataFrame,
    base_seed: int = RANDOM_STATE,
) -> tuple[
    dict[str, WindowFit],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, Any]],
]:
    """Fit all primary windows and build aligned loading/diagnostic records."""

    fits: dict[str, WindowFit] = {}
    loading_rows: list[dict[str, Any]] = []
    score_loading_rows: list[dict[str, Any]] = []
    score_correlation_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    anchors = {window_size: PastOnlyAnchor() for window_size in WINDOWS}
    previous_aligned: dict[tuple[int, str], pd.DataFrame | None] = {}
    contexts: dict[str, dict[str, Any]] = {}

    for definition in definitions:
        seed = deterministic_seed(
            definition.window_sessions,
            definition.window_start,
            definition.window_end,
            primary_starts,
            base_seed,
        )
        fit = fit_window(
            panel,
            definition,
            n_starts=primary_starts,
            random_seed=seed,
        )
        fits[definition.window_id] = fit
        past_anchor = anchors[definition.window_sessions].matrix()
        if past_anchor is None:
            past_reference = fit.structural_loadings.copy()
        else:
            past_reference = pd.DataFrame(
                past_anchor,
                index=STOCKS,
                columns=FACTORS,
            )
        past_alignment = _align_fit(fit, past_reference)
        full_alignment = _align_fit(fit, full_sample_reference)
        contexts[definition.window_id] = {
            "definition": definition,
            "past_reference": past_reference.copy(),
            "primary_past_alignment": past_alignment,
        }

        for alignment_method, alignment, reference in (
            ("past_only", past_alignment, past_reference),
            ("ex_post_full_sample_alignment", full_alignment, full_sample_reference),
        ):
            mode_key = (definition.window_sessions, alignment_method)
            previous = previous_aligned.get(mode_key)
            previous_cosines = _same_factor_cosines(
                alignment.structural_loadings,
                previous,
            )
            current_values = alignment.structural_loadings.to_numpy(dtype=float)
            reference_values = reference.to_numpy(dtype=float)
            threshold = fit.small_loading_threshold
            current_active = _aligned_support_flags(
                alignment.structural_loadings,
                threshold,
            )
            reference_active = _aligned_support_flags(reference, threshold)
            small_counts = (~current_active).sum(axis=0).astype(int)
            active_counts = current_active.sum(axis=0).astype(int)
            local_flags = small_counts > fit.gamma_n
            anchor_cosines = alignment.cosine_similarity
            anchor_jaccards = _jaccard_support(
                current_values,
                reference_values,
                threshold,
            )
            past_alignment_values = contexts[definition.window_id][
                "primary_past_alignment"
            ].structural_loadings.to_numpy(dtype=float)
            past_anchor_values = contexts[definition.window_id][
                "past_reference"
            ].to_numpy(dtype=float)
            full_alignment_values = full_alignment.structural_loadings.to_numpy(
                dtype=float
            )
            past_cosines = _same_factor_cosines(
                past_alignment.structural_loadings,
                contexts[definition.window_id]["past_reference"],
            )
            full_cosines = full_alignment.cosine_similarity
            past_jaccards = _jaccard_support(
                past_alignment_values,
                past_anchor_values,
                threshold,
            )
            full_jaccards = _jaccard_support(
                full_alignment_values,
                full_sample_reference.to_numpy(dtype=float),
                threshold,
            )
            anchor_local_flags = (
                (~reference_active).sum(axis=0).astype(int) > fit.gamma_n
            )

            common = {
                **_base_window_metadata(definition),
                **_factor_metadata(fit),
                "alignment_method": alignment_method,
            }
            for factor_number, factor in enumerate(FACTORS):
                local_changed = bool(local_flags[factor_number] != anchor_local_flags[factor_number])
                reasons = _instability_reasons(
                    float(previous_cosines[factor_number]),
                    float(anchor_cosines[factor_number]),
                    float(anchor_jaccards[factor_number]),
                    local_changed,
                    fit.rotation_condition_number,
                )
                factor_common = {
                    **common,
                    "factor": factor,
                    "source_factor": alignment.source_factors[factor_number],
                    "cosine_similarity_previous": previous_cosines[factor_number],
                    "cosine_similarity_anchor": anchor_cosines[factor_number],
                    "cosine_similarity_past_only_anchor": past_cosines[factor_number],
                    "cosine_similarity_full_sample": full_cosines[factor_number],
                    "support_jaccard_anchor": anchor_jaccards[factor_number],
                    "support_jaccard_past_only_anchor": past_jaccards[factor_number],
                    "support_jaccard_full_sample": full_jaccards[factor_number],
                    "n_small_loadings": int(small_counts[factor_number]),
                    "n_active_loadings": int(active_counts[factor_number]),
                    "small_loading_threshold": threshold,
                    "gamma_n": fit.gamma_n,
                    "is_local_factor": bool(local_flags[factor_number]),
                    "anchor_is_local_factor": bool(anchor_local_flags[factor_number]),
                    "locality_decision_changed": local_changed,
                    "l1_norm": fit.l1_norms[factor_number],
                    "solution_frequency": fit.solution_frequencies[factor_number],
                    "candidate_source": fit.sources[factor_number],
                    "instability_flag": bool(reasons),
                    "instability_reasons": ";".join(reasons),
                }
                stability_rows.append(factor_common)

                for stock_number, stock in enumerate(STOCKS):
                    loading_rows.append(
                        {
                            **factor_common,
                            "stock": stock,
                            "structural_loading": current_values[
                                stock_number, factor_number
                            ],
                            "absolute_structural_loading": abs(
                                current_values[stock_number, factor_number]
                            ),
                            "anchor_structural_loading": reference_values[
                                stock_number, factor_number
                            ],
                            "is_active": bool(current_active[stock_number, factor_number]),
                            "anchor_is_active": bool(
                                reference_active[stock_number, factor_number]
                            ),
                            "support_match": bool(
                                current_active[stock_number, factor_number]
                                == reference_active[stock_number, factor_number]
                            ),
                        }
                    )
                    score_loading_rows.append(
                        {
                            **factor_common,
                            "stock": stock,
                            "stock_factor_score_loading": alignment.score_loadings.iloc[
                                stock_number, factor_number
                            ],
                        }
                    )
                    support_rows.append(
                        {
                            **factor_common,
                            "stock": stock,
                            "structural_loading": current_values[
                                stock_number, factor_number
                            ],
                            "is_active": bool(current_active[stock_number, factor_number]),
                            "anchor_is_active": bool(
                                reference_active[stock_number, factor_number]
                            ),
                            "support_match": bool(
                                current_active[stock_number, factor_number]
                                == reference_active[stock_number, factor_number]
                            ),
                        }
                    )

                for other_number, other_factor in enumerate(FACTORS):
                    score_correlation_rows.append(
                        {
                            **common,
                            "factor": factor,
                            "other_factor": other_factor,
                            "score_correlation": alignment.score_correlation.iloc[
                                factor_number, other_number
                            ],
                        }
                    )

            diagnostics_rows.append(
                {
                    **common,
                    "mean_cosine_similarity_anchor": float(np.nanmean(anchor_cosines)),
                    "minimum_cosine_similarity_anchor": float(np.nanmin(anchor_cosines)),
                    "mean_support_jaccard_anchor": float(np.nanmean(anchor_jaccards)),
                    "minimum_support_jaccard_anchor": float(np.nanmin(anchor_jaccards)),
                    "n_local_factors": int(local_flags.sum()),
                    "n_instability_flags": int(sum(bool(reasons) for reasons in [
                        _instability_reasons(
                            float(previous_cosines[index]),
                            float(anchor_cosines[index]),
                            float(anchor_jaccards[index]),
                            bool(local_flags[index] != anchor_local_flags[index]),
                            fit.rotation_condition_number,
                        )
                        for index in range(N_COMPONENTS)
                    ])),
                    "sensitivity_run": False,
                    "sensitivity_reason": "",
                    "sensitivity_conclusion_stable": np.nan,
                    "sensitivity_instability_flag": False,
                }
            )
            previous_aligned[mode_key] = alignment.structural_loadings.copy()

        anchors[definition.window_sessions].update(
            past_alignment.structural_loadings
        )

    return (
        fits,
        pd.DataFrame(loading_rows),
        pd.DataFrame(score_loading_rows),
        pd.DataFrame(support_rows),
        {
            "stability": pd.DataFrame(stability_rows),
            "score_correlations": pd.DataFrame(score_correlation_rows),
            "diagnostics": pd.DataFrame(diagnostics_rows),
            "contexts": contexts,
        },
    )


def _annotate_persistence(stability: pd.DataFrame) -> pd.DataFrame:
    """Require two consecutive rolling checkpoints before a regime candidate."""

    result = stability.copy()
    result["persistent_instability"] = False
    result["regime_candidate"] = False
    for keys, group in result.groupby(
        ["window_sessions", "alignment_method", "factor"],
        sort=False,
    ):
        ordered = group.sort_values("window_start_index")
        current = ordered["instability_flag"].astype(bool).to_numpy()
        starts = ordered["window_start_index"].to_numpy(dtype=int)
        consecutive = np.r_[False, np.diff(starts) == STEP_SESSIONS]
        persistent = current & np.roll(current, 1)
        persistent[0] = False
        persistent &= consecutive
        result.loc[ordered.index, "persistent_instability"] = persistent
        result.loc[ordered.index, "regime_candidate"] = persistent
    return result


def run_start_count_sensitivity(
    panel: pd.DataFrame,
    definitions_by_id: dict[str, WindowDefinition],
    fits: dict[str, WindowFit],
    contexts: dict[str, dict[str, Any]],
    candidate_ids: Iterable[str],
    *,
    sensitivity_starts: int = SENSITIVITY_STARTS,
    base_seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Re-fit quarterly or unstable windows with 500 starts."""

    rows: list[dict[str, Any]] = []
    for window_id in sorted(candidate_ids):
        definition = definitions_by_id[window_id]
        primary = fits[window_id]
        seed = deterministic_seed(
            definition.window_sessions,
            definition.window_start,
            definition.window_end,
            sensitivity_starts,
            base_seed,
        )
        sensitivity = fit_window(
            panel,
            definition,
            n_starts=sensitivity_starts,
            random_seed=seed,
        )
        anchor = contexts[window_id]["past_reference"]
        aligned_sensitivity = _align_fit(sensitivity, anchor)
        primary_aligned = contexts[window_id]["primary_past_alignment"]
        threshold = primary.small_loading_threshold
        primary_values = primary_aligned.structural_loadings.to_numpy(dtype=float)
        sensitivity_values = aligned_sensitivity.structural_loadings.to_numpy(dtype=float)
        primary_small = (np.abs(primary_values) < threshold).sum(axis=0).astype(int)
        sensitivity_small = (
            np.abs(sensitivity_values) < threshold
        ).sum(axis=0).astype(int)
        primary_local = primary_small > primary.gamma_n
        sensitivity_local = sensitivity_small > sensitivity.gamma_n
        supports = np.asarray(
            [
                _jaccard_support(
                    sensitivity_values[:, factor : factor + 1],
                    primary_values[:, factor : factor + 1],
                    threshold,
                )[0]
                for factor in range(N_COMPONENTS)
            ]
        )
        stable = (
            np.all(aligned_sensitivity.cosine_similarity >= COSINE_INSTABILITY_THRESHOLD)
            and np.all(supports >= JACCARD_INSTABILITY_THRESHOLD)
            and np.array_equal(primary_local, sensitivity_local)
        )
        for factor_number, factor in enumerate(FACTORS):
            rows.append(
                {
                    **_base_window_metadata(definition),
                    "alignment_method": "past_only",
                    "factor": factor,
                    "primary_n_starts": primary.n_starts,
                    "sensitivity_n_starts": sensitivity.n_starts,
                    "primary_seed": primary.random_seed,
                    "sensitivity_seed": sensitivity.random_seed,
                    "primary_vs_sensitivity_cosine": aligned_sensitivity.cosine_similarity[
                        factor_number
                    ],
                    "support_jaccard_primary_vs_sensitivity": supports[factor_number],
                    "primary_small_count": primary_small[factor_number],
                    "sensitivity_small_count": sensitivity_small[factor_number],
                    "primary_local_factor": bool(primary_local[factor_number]),
                    "sensitivity_local_factor": bool(sensitivity_local[factor_number]),
                    "local_factor_decision_unchanged": bool(
                        primary_local[factor_number] == sensitivity_local[factor_number]
                    ),
                    "primary_condition_number": primary.rotation_condition_number,
                    "sensitivity_condition_number": sensitivity.rotation_condition_number,
                    "primary_optimizer_success_rate": primary.optimizer_success_rate,
                    "sensitivity_optimizer_success_rate": sensitivity.optimizer_success_rate,
                    "primary_l1_norm": primary.l1_norms[factor_number],
                    "sensitivity_l1_norm": sensitivity.l1_norms[factor_number],
                    "sensitivity_conclusion_stable": bool(stable),
                    "sensitivity_instability_flag": bool(not stable),
                }
            )
    return pd.DataFrame(rows)


def _merge_sensitivity_annotations(
    frame: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Attach window-level sensitivity results to stability and diagnostics."""

    if sensitivity.empty:
        result = frame.copy()
        result["sensitivity_run"] = False
        result["sensitivity_reason"] = ""
        result["sensitivity_conclusion_stable"] = np.nan
        result["sensitivity_instability_flag"] = False
        return result
    summary = (
        sensitivity.groupby("window_id", as_index=False)
        .agg(
            sensitivity_conclusion_stable=("sensitivity_conclusion_stable", "first"),
            sensitivity_instability_flag=("sensitivity_instability_flag", "first"),
        )
        .assign(sensitivity_run=True)
    )
    summary["sensitivity_reason"] = "quarterly_or_primary_instability"
    result = frame.merge(summary, on="window_id", how="left", suffixes=("", "_sensitivity"))
    result["sensitivity_run"] = result["sensitivity_run"].fillna(False).astype(bool)
    result["sensitivity_reason"] = result["sensitivity_reason"].fillna("")
    result["sensitivity_instability_flag"] = (
        result["sensitivity_instability_flag"].fillna(False).astype(bool)
    )
    return result


def build_window_diagnostics(
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate factor-level diagnostics while retaining alignment method."""

    group_columns = ["window_id", "window_sessions", "window_start", "window_end", "alignment_method"]
    aggregations = stability.groupby(group_columns, as_index=False).agg(
        window_start_index=("window_start_index", "first"),
        window_end_index=("window_end_index", "first"),
        regime=("regime", "first"),
        regime_sequence=("regime_sequence", "first"),
        crosses_regime_boundary=("crosses_regime_boundary", "first"),
        is_quarterly_window=("is_quarterly_window", "first"),
        n_sessions=("n_sessions", "first"),
        n_observations=("n_observations", "first"),
        pc1_explained_pct=("pc1_explained_pct", "first"),
        cumulative_explained_pct_first3=("cumulative_explained_pct_first3", "first"),
        benchmark_variance_removed_pct=("benchmark_variance_removed_pct", "first"),
        max_abs_residual_benchmark_corr=("max_abs_residual_benchmark_corr", "first"),
        mean_cosine_similarity_anchor=("cosine_similarity_anchor", "mean"),
        minimum_cosine_similarity_anchor=("cosine_similarity_anchor", "min"),
        mean_support_jaccard_anchor=("support_jaccard_anchor", "mean"),
        minimum_support_jaccard_anchor=("support_jaccard_anchor", "min"),
        n_local_factors=("is_local_factor", "sum"),
        n_instability_flags=("instability_flag", "sum"),
        n_persistent_instability=("persistent_instability", "sum"),
        regime_candidate=("regime_candidate", "any"),
        sensitivity_run=("sensitivity_run", "any"),
        sensitivity_conclusion_stable=("sensitivity_conclusion_stable", "first"),
        sensitivity_instability_flag=("sensitivity_instability_flag", "any"),
    )
    return aggregations


def build_regime_summary(stability: pd.DataFrame) -> pd.DataFrame:
    """Summarize factor stability by window size, alignment mode, and regime."""

    return (
        stability.groupby(
            ["window_sessions", "alignment_method", "regime", "factor"],
            as_index=False,
        )
        .agg(
            n_windows=("window_id", "nunique"),
            n_mixed_regime_windows=("crosses_regime_boundary", "sum"),
            mean_cosine_similarity_previous=("cosine_similarity_previous", "mean"),
            mean_cosine_similarity_anchor=("cosine_similarity_anchor", "mean"),
            minimum_cosine_similarity_anchor=("cosine_similarity_anchor", "min"),
            mean_cosine_similarity_full_sample=("cosine_similarity_full_sample", "mean"),
            mean_support_jaccard_anchor=("support_jaccard_anchor", "mean"),
            minimum_support_jaccard_anchor=("support_jaccard_anchor", "min"),
            mean_n_small_loadings=("n_small_loadings", "mean"),
            local_factor_rate=("is_local_factor", "mean"),
            mean_rotation_condition_number=("rotation_condition_number", "mean"),
            mean_pc1_explained_pct=("pc1_explained_pct", "mean"),
            mean_first3_explained_pct=("cumulative_explained_pct_first3", "mean"),
            mean_benchmark_variance_removed_pct=(
                "benchmark_variance_removed_pct",
                "mean",
            ),
            n_regime_candidates=("regime_candidate", "sum"),
            n_sensitivity_runs=("sensitivity_run", "sum"),
            n_sensitivity_instabilities=("sensitivity_instability_flag", "sum"),
        )
    )


def build_unstable_windows(stability: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive instability candidates, not formal break tests."""

    result = stability[
        stability["regime_candidate"] | stability["sensitivity_instability_flag"]
    ].copy()
    if result.empty:
        return pd.DataFrame(
            columns=[
                *stability.columns,
                "candidate_type",
            ]
        )
    result["candidate_type"] = np.where(
        result["regime_candidate"] & result["sensitivity_instability_flag"],
        "persistent_instability_and_sensitivity_disagreement",
        np.where(
            result["regime_candidate"],
            "persistent_instability",
            "sensitivity_disagreement",
        ),
    )
    return result


def _shade_crisis(axis: Axes, dates: pd.Series | pd.DatetimeIndex) -> None:
    """Highlight March-May 2023 without assigning a causal interpretation."""

    parsed = pd.to_datetime(dates)
    if len(parsed) == 0:
        return
    minimum = parsed.min()
    maximum = parsed.max()
    if maximum < CRISIS_SHADE_START or minimum > CRISIS_SHADE_END:
        return
    axis.axvspan(
        CRISIS_SHADE_START,
        CRISIS_SHADE_END,
        color="#E07A5F",
        alpha=0.12,
        linewidth=0,
        label="Mar–May 2023",
    )


def _shade_index_crisis(axis: Axes, dates: pd.DatetimeIndex) -> None:
    """Highlight the crisis interval on an image plot with integer x positions."""

    if len(dates) == 0:
        return
    mask = (dates >= CRISIS_SHADE_START) & (dates <= CRISIS_SHADE_END)
    if mask.any():
        positions = np.flatnonzero(mask)
        axis.axvspan(
            max(-0.5, positions[0] - 0.5),
            positions[-1] + 0.5,
            color="#E07A5F",
            alpha=0.12,
            linewidth=0,
        )


def _format_image_dates(axis: Axes, dates: pd.DatetimeIndex) -> None:
    """Use a small, readable set of dates on rolling heatmaps."""

    if len(dates) == 0:
        return
    count = min(7, len(dates))
    positions = np.linspace(0, len(dates) - 1, count, dtype=int)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [dates[position].strftime("%Y-%m-%d") for position in positions],
        rotation=30,
        ha="right",
    )


def plot_loading_heatmaps(loadings: pd.DataFrame) -> None:
    """Plot full-sample-aligned structural loadings through time."""

    data = loadings[
        loadings["alignment_method"] == "ex_post_full_sample_alignment"
    ]
    maximum = float(np.nanmax(np.abs(data["structural_loading"])))
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    figure, axes = plt.subplots(
        len(WINDOWS),
        len(FACTORS),
        figsize=(18, 10),
        squeeze=False,
        sharey=True,
    )
    image = None
    for row, window_size in enumerate(WINDOWS):
        for column, factor in enumerate(FACTORS):
            axis = axes[row, column]
            subset = data[
                (data["window_sessions"] == window_size)
                & (data["factor"] == factor)
            ]
            pivot = subset.pivot(
                index="stock",
                columns="window_end",
                values="structural_loading",
            ).reindex(STOCKS)
            dates = pd.DatetimeIndex(pd.to_datetime(pivot.columns))
            image = axis.imshow(
                pivot.to_numpy(dtype=float),
                cmap="RdBu_r",
                norm=norm,
                aspect="auto",
                interpolation="nearest",
            )
            _shade_index_crisis(axis, dates)
            _format_image_dates(axis, dates)
            axis.set_title(f"{window_size}-session: {factor}")
            if column == 0:
                axis.set_yticks(range(len(STOCKS)))
                axis.set_yticklabels(STOCKS)
            else:
                axis.set_yticks(range(len(STOCKS)))
                axis.set_yticklabels([])
            axis.tick_params(axis="both", length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
    colorbar_axis = figure.add_axes([0.925, 0.16, 0.015, 0.68])
    figure.colorbar(image, cax=colorbar_axis, label="Structural loading")
    figure.suptitle(
        "Rolling L1-rotation structural loadings",
        x=0.06,
        ha="left",
        y=0.98,
        fontsize=14,
    )
    figure.text(
        0.06,
        0.945,
        "Ex-post full-sample alignment is descriptive only; the shaded interval is March–May 2023.",
        color="#666666",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.05,
        right=0.90,
        bottom=0.10,
        top=0.86,
        wspace=0.14,
        hspace=0.28,
    )
    save_figure(figure, OUT_DIR / "14_rolling_l1_loading_heatmaps.png", tight_bbox=True)


def plot_support_heatmaps(support: pd.DataFrame) -> None:
    """Plot full-sample-aligned active support through time."""

    data = support[
        support["alignment_method"] == "ex_post_full_sample_alignment"
    ]
    figure, axes = plt.subplots(
        len(WINDOWS),
        len(FACTORS),
        figsize=(18, 10),
        squeeze=False,
        sharey=True,
    )
    image = None
    for row, window_size in enumerate(WINDOWS):
        for column, factor in enumerate(FACTORS):
            axis = axes[row, column]
            subset = data[
                (data["window_sessions"] == window_size)
                & (data["factor"] == factor)
            ]
            pivot = subset.pivot(
                index="stock",
                columns="window_end",
                values="is_active",
            ).reindex(STOCKS)
            dates = pd.DatetimeIndex(pd.to_datetime(pivot.columns))
            image = axis.imshow(
                pivot.astype(float).to_numpy(),
                cmap="Blues",
                vmin=0.0,
                vmax=1.0,
                aspect="auto",
                interpolation="nearest",
            )
            _shade_index_crisis(axis, dates)
            _format_image_dates(axis, dates)
            axis.set_title(f"{window_size}-session: {factor}")
            axis.set_yticks(range(len(STOCKS)))
            axis.set_yticklabels(STOCKS if column == 0 else [])
            axis.tick_params(axis="both", length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
    colorbar_axis = figure.add_axes([0.925, 0.16, 0.015, 0.68])
    figure.colorbar(image, cax=colorbar_axis, label="Active support (1 = active)")
    figure.suptitle(
        "Rolling L1-rotation support membership",
        x=0.06,
        ha="left",
        y=0.98,
        fontsize=14,
    )
    figure.text(
        0.06,
        0.945,
        "Support uses |structural loading| ≥ h_n; labels are aligned ex post for readability.",
        color="#666666",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.05,
        right=0.90,
        bottom=0.10,
        top=0.86,
        wspace=0.14,
        hspace=0.28,
    )
    save_figure(
        figure,
        OUT_DIR / "14_rolling_support_membership_heatmaps.png",
        tight_bbox=True,
    )


def plot_stability(stability: pd.DataFrame) -> None:
    """Plot past-only cosine and support-Jaccard diagnostics."""

    data = stability[stability["alignment_method"] == "past_only"]
    figure, axes = plt.subplots(
        len(WINDOWS),
        2,
        figsize=(15, 8),
        squeeze=False,
        sharex="col",
        sharey="col",
    )
    metrics = [
        ("cosine_similarity_anchor", "Cosine to past-only anchor", 0.80),
        ("support_jaccard_anchor", "Support Jaccard to past-only anchor", 0.50),
    ]
    for row, window_size in enumerate(WINDOWS):
        subset = data[data["window_sessions"] == window_size]
        for column, (metric, ylabel, threshold) in enumerate(metrics):
            axis = axes[row, column]
            for factor in FACTORS:
                factor_data = subset[subset["factor"] == factor].sort_values(
                    "window_end"
                )
                axis.plot(
                    pd.to_datetime(factor_data["window_end"]),
                    factor_data[metric],
                    color=COLORS[factor],
                    linewidth=1.4,
                    label=factor,
                )
            axis.axhline(threshold, color="#777777", linestyle="--", linewidth=1.0)
            _shade_crisis(axis, pd.to_datetime(subset["window_end"]))
            axis.set_ylim(-0.02, 1.05)
            axis.set_title(f"{window_size}-session window")
            axis.set_ylabel(ylabel)
            style_axis(axis, format_dates=True)
            if row == len(WINDOWS) - 1:
                axis.set_xlabel("Window end")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    figure.suptitle(
        "Past-only factor stability diagnostics",
        x=0.06,
        ha="left",
        y=1.04,
        fontsize=14,
    )
    figure.text(
        0.06,
        1.005,
        "Dashed lines are descriptive instability thresholds, not structural-break tests.",
        color="#666666",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(figure, OUT_DIR / "14_rolling_factor_stability.png", tight_bbox=True)


def plot_small_loading_counts(stability: pd.DataFrame) -> None:
    """Plot small-loading counts against the local-factor critical count."""

    data = stability[stability["alignment_method"] == "past_only"]
    figure, axes = plt.subplots(len(WINDOWS), 1, figsize=(12, 7), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, window_size in zip(axes, WINDOWS):
        subset = data[data["window_sessions"] == window_size]
        for factor in FACTORS:
            factor_data = subset[subset["factor"] == factor].sort_values("window_end")
            axis.plot(
                pd.to_datetime(factor_data["window_end"]),
                factor_data["n_small_loadings"],
                color=COLORS[factor],
                linewidth=1.4,
                label=factor,
            )
        gamma = float(subset["gamma_n"].iloc[0])
        axis.axhline(gamma, color="#777777", linestyle="--", linewidth=1.0, label="gamma_n")
        _shade_crisis(axis, pd.to_datetime(subset["window_end"]))
        axis.set_ylim(0, len(STOCKS) + 0.2)
        axis.set_ylabel("Small loadings")
        axis.set_title(f"{window_size}-session window")
        style_axis(axis, format_dates=True)
    axes[-1].set_xlabel("Window end")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False)
    figure.suptitle(
        "Local-factor diagnostic through time",
        x=0.08,
        ha="left",
        y=1.05,
        fontsize=14,
    )
    figure.text(
        0.08,
        1.015,
        "A factor is local under the reference rule only when its small-loading count exceeds gamma_n.",
        color="#666666",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(figure, OUT_DIR / "14_small_loading_counts.png", tight_bbox=True)


def plot_variance_diagnostics(diagnostics: pd.DataFrame) -> None:
    """Plot explained variance and rolling SPY/XLF variance removal."""

    data = diagnostics[diagnostics["alignment_method"] == "past_only"]
    metrics = [
        ("pc1_explained_pct", "PC1 explained variance (%)"),
        ("cumulative_explained_pct_first3", "First-three explained variance (%)"),
        ("benchmark_variance_removed_pct", "Mean SPY/XLF variance removed (%)"),
    ]
    figure, axes = plt.subplots(len(WINDOWS), len(metrics), figsize=(17, 8), sharex="col")
    axes = np.atleast_2d(axes)
    for row, window_size in enumerate(WINDOWS):
        subset = data[data["window_sessions"] == window_size].sort_values("window_end")
        for column, (metric, ylabel) in enumerate(metrics):
            axis = axes[row, column]
            axis.plot(
                pd.to_datetime(subset["window_end"]),
                subset[metric],
                color="#2F6690" if metric != "benchmark_variance_removed_pct" else "#C26A2E",
                linewidth=1.5,
            )
            _shade_crisis(axis, pd.to_datetime(subset["window_end"]))
            axis.set_title(f"{window_size}-session window")
            axis.set_ylabel(ylabel)
            style_axis(axis, format_dates=True)
            if row == len(WINDOWS) - 1:
                axis.set_xlabel("Window end")
    figure.suptitle(
        "Rolling variance and benchmark diagnostics",
        x=0.06,
        ha="left",
        y=1.02,
        fontsize=14,
    )
    figure.text(
        0.06,
        0.985,
        "All quantities are re-estimated within the trailing window; shaded interval is March–May 2023.",
        color="#666666",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(figure, OUT_DIR / "14_variance_benchmark_diagnostics.png", tight_bbox=True)


def _grouped_bar(
    axis: Axes,
    frame: pd.DataFrame,
    value: str,
    ylabel: str,
    title: str,
) -> None:
    """Draw a compact factor-by-regime grouped bar chart."""

    regimes = list(frame["regime"].drop_duplicates())
    positions = np.arange(len(regimes))
    width = 0.24
    for index, factor in enumerate(FACTORS):
        values = []
        for regime in regimes:
            subset = frame[(frame["regime"] == regime) & (frame["factor"] == factor)]
            values.append(float(subset[value].iloc[0]) if len(subset) else np.nan)
        axis.bar(
            positions + (index - 1) * width,
            values,
            width=width,
            color=COLORS[factor],
            label=factor,
        )
    axis.set_xticks(positions, [regime.replace("_", "\n") for regime in regimes])
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    style_axis(axis)


def plot_regime_comparison(regime_summary: pd.DataFrame) -> None:
    """Plot the explicit pre-crisis, crisis, post-crisis, and recent comparison."""

    data = regime_summary[regime_summary["alignment_method"] == "past_only"]
    selected = data[data["window_sessions"] == 60]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    _grouped_bar(
        axes[0],
        selected,
        "mean_cosine_similarity_anchor",
        "Mean cosine similarity",
        "60-session regime comparison",
    )
    _grouped_bar(
        axes[1],
        selected,
        "mean_support_jaccard_anchor",
        "Mean support Jaccard",
        "60-session regime comparison",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    figure.suptitle(
        "Factor stability by market regime",
        x=0.06,
        ha="left",
        y=1.08,
        fontsize=14,
    )
    figure.text(
        0.06,
        1.03,
        "Regime is assigned by the majority of sessions in a window; mixed windows are retained and flagged.",
        color="#666666",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(figure, OUT_DIR / "14_regime_comparison.png", tight_bbox=True)


def plot_window_comparison(stability: pd.DataFrame) -> None:
    """Compare 60- and 120-session stability summaries."""

    data = stability[stability["alignment_method"] == "past_only"]
    summary = data.groupby(["window_sessions", "factor"], as_index=False).agg(
        mean_cosine=("cosine_similarity_anchor", "mean"),
        mean_jaccard=("support_jaccard_anchor", "mean"),
        local_rate=("is_local_factor", "mean"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True)
    metrics = [
        ("mean_cosine", "Mean past-only cosine", (0.0, 1.02)),
        ("mean_jaccard", "Mean past-only Jaccard", (0.0, 1.02)),
        ("local_rate", "Local-factor rate", (0.0, 1.02)),
    ]
    positions = np.arange(len(FACTORS))
    width = 0.34
    for axis, (metric, ylabel, limits) in zip(axes, metrics):
        for index, window_size in enumerate(WINDOWS):
            values = [
                float(
                    summary[
                        (summary["window_sessions"] == window_size)
                        & (summary["factor"] == factor)
                    ][metric].iloc[0]
                )
                for factor in FACTORS
            ]
            axis.bar(
                positions + (index - 0.5) * width,
                values,
                width=width,
                label=f"{window_size} sessions",
                color="#2F6690" if window_size == 60 else "#D99A2B",
            )
        axis.set_xticks(positions, FACTORS)
        axis.set_ylim(*limits)
        axis.set_ylabel(ylabel)
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    figure.suptitle(
        "60-session versus 120-session factor diagnostics",
        x=0.06,
        ha="left",
        y=1.08,
        fontsize=14,
    )
    figure.text(
        0.06,
        1.03,
        "The 60-session window is primary; 120 sessions is the persistence robustness check.",
        color="#666666",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(figure, OUT_DIR / "14_window_size_comparison.png", tight_bbox=True)


def write_outputs(
    loadings: pd.DataFrame,
    score_loadings: pd.DataFrame,
    stability: pd.DataFrame,
    support: pd.DataFrame,
    diagnostics: pd.DataFrame,
    regime_summary: pd.DataFrame,
    unstable_windows: pd.DataFrame,
    sensitivity: pd.DataFrame,
    score_correlations: pd.DataFrame,
) -> None:
    """Write all machine-readable stage-14 artifacts to the ignored report folder."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "14_rolling_l1_loadings.csv": loadings,
        "14_rolling_score_loadings.csv": score_loadings,
        "14_rolling_factor_stability.csv": stability,
        "14_rolling_support_membership.csv": support,
        "14_rolling_window_diagnostics.csv": diagnostics,
        "14_regime_summary.csv": regime_summary,
        "14_unstable_windows.csv": unstable_windows,
        "14_start_count_sensitivity.csv": sensitivity,
        "14_rolling_score_correlations.csv": score_correlations,
    }
    for filename, table in tables.items():
        table.to_csv(OUT_DIR / filename, index=False)


def make_plots(
    loadings: pd.DataFrame,
    support: pd.DataFrame,
    stability: pd.DataFrame,
    diagnostics: pd.DataFrame,
    regime_summary: pd.DataFrame,
) -> None:
    """Create all requested diagnostic figures."""

    plot_loading_heatmaps(loadings)
    plot_support_heatmaps(support)
    plot_stability(stability)
    plot_small_loading_counts(stability)
    plot_variance_diagnostics(diagnostics)
    plot_regime_comparison(regime_summary)
    plot_window_comparison(stability)


def run_analysis(
    panel: pd.DataFrame | None = None,
    *,
    window_sizes: Iterable[int] = WINDOWS,
    step_sessions: int = STEP_SESSIONS,
    primary_starts: int = PRIMARY_STARTS,
    sensitivity_starts: int = SENSITIVITY_STARTS,
    full_sample_starts: int = FULL_SAMPLE_STARTS,
    base_seed: int = RANDOM_STATE,
    full_sample_seed: int = FULL_SAMPLE_RANDOM_STATE,
    make_figures: bool = True,
    write_files: bool = True,
) -> dict[str, Any]:
    """Run the dynamic local-factor regime analysis and return all result tables."""

    if panel is None:
        required = STOCKS + list(BENCHMARKS)
        panel = load_panel(
            RETURN_MATRIX_CLEAN_FILE,
            required,
            context="Dynamic local-factor panel",
            drop_incomplete=True,
        )
    else:
        panel = panel.copy()
        validate_panel(panel, context="Dynamic local-factor panel", require_complete=True)
        missing = [column for column in STOCKS + list(BENCHMARKS) if column not in panel]
        if missing:
            raise ValueError(f"Dynamic local-factor panel is missing columns: {missing}")
        panel = panel.loc[:, STOCKS + list(BENCHMARKS)]

    window_sizes = tuple(int(size) for size in window_sizes)
    if set(window_sizes) != set(WINDOWS):
        raise ValueError(f"The production specification must use windows {WINDOWS}.")
    definitions = build_window_definitions(panel, window_sizes, step_sessions)
    definitions_by_id = {definition.window_id: definition for definition in definitions}
    if not definitions:
        raise ValueError("The panel does not contain enough sessions for stage 14.")

    full_reference = fit_full_sample_reference(
        panel,
        n_starts=full_sample_starts,
        random_seed=full_sample_seed,
    )
    fits, loadings, score_loadings, support, extra = run_rolling_fits(
        panel,
        definitions,
        primary_starts=primary_starts,
        full_sample_reference=full_reference,
        base_seed=base_seed,
    )
    stability = _annotate_persistence(extra["stability"])
    primary_candidates = set(
        stability.loc[
            (stability["alignment_method"] == "past_only")
            & (
                stability["is_quarterly_window"]
                | stability["instability_flag"]
            ),
            "window_id",
        ]
    )
    sensitivity = run_start_count_sensitivity(
        panel,
        definitions_by_id,
        fits,
        extra["contexts"],
        primary_candidates,
        sensitivity_starts=sensitivity_starts,
        base_seed=base_seed,
    )
    stability = _merge_sensitivity_annotations(stability, sensitivity)
    diagnostics = build_window_diagnostics(stability)
    diagnostics = diagnostics.sort_values(
        ["window_sessions", "window_end", "alignment_method"]
    ).reset_index(drop=True)
    regime_summary = build_regime_summary(stability)
    unstable_windows = build_unstable_windows(stability)
    for frame in (
        loadings,
        score_loadings,
        support,
        stability,
        regime_summary,
        unstable_windows,
    ):
        if not frame.empty:
            sort_columns = [
                column
                for column in (
                    "window_sessions",
                    "window_end",
                    "alignment_method",
                    "regime",
                    "factor",
                )
                if column in frame.columns
            ]
            frame.sort_values(sort_columns, inplace=True)
            frame.reset_index(drop=True, inplace=True)
    extra["diagnostics"] = diagnostics
    extra["stability"] = stability
    extra["regime_summary"] = regime_summary
    extra["unstable_windows"] = unstable_windows
    extra["sensitivity"] = sensitivity
    extra["loadings"] = loadings
    extra["score_loadings"] = score_loadings
    extra["support"] = support
    extra["fits"] = fits
    extra["full_sample_reference"] = full_reference

    if write_files:
        ensure_project_directories()
        write_outputs(
            loadings,
            score_loadings,
            stability,
            support,
            diagnostics,
            regime_summary,
            unstable_windows,
            sensitivity,
            extra["score_correlations"],
        )
    if make_figures:
        ensure_project_directories()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        make_plots(loadings, support, stability, diagnostics, regime_summary)
    return extra


def _print_summary(results: dict[str, Any]) -> None:
    """Print a concise real-data summary for reproducibility logs."""

    stability = results["stability"]
    diagnostics = results["diagnostics"]
    sensitivity = results["sensitivity"]
    primary = stability[stability["alignment_method"] == "past_only"]
    print("=== DYNAMIC L1 LOCAL FACTOR REGIMES ===")
    print(f"Windows: {WINDOWS}; session step: {STEP_SESSIONS}")
    print(f"Primary starts: {PRIMARY_STARTS}; sensitivity starts: {SENSITIVITY_STARTS}")
    print(f"Primary rows: {len(primary)} factor-window rows")
    print(f"Sensitivity windows: {sensitivity['window_id'].nunique() if not sensitivity.empty else 0}")
    summary = (
        primary.groupby(["window_sessions", "factor"], as_index=False)
        .agg(
            mean_cosine=("cosine_similarity_anchor", "mean"),
            p05_cosine=("cosine_similarity_anchor", lambda values: values.quantile(0.05)),
            mean_jaccard=("support_jaccard_anchor", "mean"),
            local_rate=("is_local_factor", "mean"),
            n_regime_candidates=("regime_candidate", "sum"),
        )
    )
    print(summary.round(4).to_string(index=False))
    print("\nRegime summary (60-session, past-only):")
    print(
        results["regime_summary"][
            (results["regime_summary"]["window_sessions"] == 60)
            & (results["regime_summary"]["alignment_method"] == "past_only")
        ].round(4).to_string(index=False)
    )
    print("\nWindow diagnostics:")
    print(diagnostics.head().round(4).to_string(index=False))
    print(f"\nOutputs saved to: {OUT_DIR}")


def main() -> None:
    """Run stage 14 on the configured real-data panel."""

    results = run_analysis()
    _print_summary(results)


if __name__ == "__main__":
    main()
