"""Reusable routines for dynamic factor."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config import BENCHMARKS, CORE_UNIVERSE
from utils.benchmark import residualize_against_benchmarks
from utils.data import validate_panel
from utils.l1_rotation import align_loading_columns, fit_l1_rotation
from utils.pca import fit_pca
from utils.rolling import rolling_starts, session_index, session_window_positions

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
                raise AssertionError(
                    "A rolling window did not contain the requested sessions."
                )
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
            anchor_local_flags = (~reference_active).sum(axis=0).astype(
                int
            ) > fit.gamma_n

            common = {
                **_base_window_metadata(definition),
                **_factor_metadata(fit),
                "alignment_method": alignment_method,
            }
            for factor_number, factor in enumerate(FACTORS):
                local_changed = bool(
                    local_flags[factor_number] != anchor_local_flags[factor_number]
                )
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
                            "is_active": bool(
                                current_active[stock_number, factor_number]
                            ),
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
                            "is_active": bool(
                                current_active[stock_number, factor_number]
                            ),
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
                    "minimum_cosine_similarity_anchor": float(
                        np.nanmin(anchor_cosines)
                    ),
                    "mean_support_jaccard_anchor": float(np.nanmean(anchor_jaccards)),
                    "minimum_support_jaccard_anchor": float(np.nanmin(anchor_jaccards)),
                    "n_local_factors": int(local_flags.sum()),
                    "n_instability_flags": int(
                        sum(
                            bool(reasons)
                            for reasons in [
                                _instability_reasons(
                                    float(previous_cosines[index]),
                                    float(anchor_cosines[index]),
                                    float(anchor_jaccards[index]),
                                    bool(
                                        local_flags[index] != anchor_local_flags[index]
                                    ),
                                    fit.rotation_condition_number,
                                )
                                for index in range(N_COMPONENTS)
                            ]
                        )
                    ),
                    "sensitivity_run": False,
                    "sensitivity_reason": "",
                    "sensitivity_conclusion_stable": np.nan,
                    "sensitivity_instability_flag": False,
                }
            )
            previous_aligned[mode_key] = alignment.structural_loadings.copy()

        anchors[definition.window_sessions].update(past_alignment.structural_loadings)

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
        sensitivity_values = aligned_sensitivity.structural_loadings.to_numpy(
            dtype=float
        )
        primary_small = (np.abs(primary_values) < threshold).sum(axis=0).astype(int)
        sensitivity_small = (
            (np.abs(sensitivity_values) < threshold).sum(axis=0).astype(int)
        )
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
            np.all(
                aligned_sensitivity.cosine_similarity >= COSINE_INSTABILITY_THRESHOLD
            )
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
    result = frame.merge(
        summary, on="window_id", how="left", suffixes=("", "_sensitivity")
    )
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

    group_columns = [
        "window_id",
        "window_sessions",
        "window_start",
        "window_end",
        "alignment_method",
    ]
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

    return stability.groupby(
        ["window_sessions", "alignment_method", "regime", "factor"],
        as_index=False,
    ).agg(
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
