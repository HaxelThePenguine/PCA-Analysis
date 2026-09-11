"""13-bis: session Kalman factors followed by the existing L1 rotation.

Intraday residuals are summed by trading session to keep the state-space
extension small. PCA supplies a loading basis in each trailing window,
factors follow an AR(1) Kalman model, and L1 rotates that same basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
from data_utils import validate_panel
from l1_rotation_utils import align_loading_columns, fit_l1_rotation, local_factor_test
from pca_utils import fit_pca
from plotting_utils import save_figure, style_axis
from rolling_utils import rolling_starts, session_index


OUT_DIR = REPORTS_DIR / "kalman_dynamic_local_factors"
STOCKS = list(CORE_UNIVERSE)
PRIMARY_K = 3
SENSITIVITY_K = (2, 4, 5)
WINDOW = 60
STEP = 20
TRAIN_FRACTION = 0.80
PRIMARY_STARTS = 80
SENSITIVITY_STARTS = 30
SENSITIVITY_EVERY = 5
SEED = 13213
SMALL_TOLERANCE = 1e-8
CRISIS_START = pd.Timestamp("2023-03-01")
CRISIS_END = pd.Timestamp("2023-05-31")


@dataclass(frozen=True)
class Window:
    """Trailing session window and train/holdout positions."""

    name: str
    start: pd.Timestamp
    train_end: pd.Timestamp
    end: pd.Timestamp
    train: np.ndarray
    test: np.ndarray


def load_session_residuals() -> pd.DataFrame:
    """Residualize the intraday panel, then aggregate residual returns by session."""

    columns = STOCKS + list(BENCHMARKS)
    panel = pd.read_parquet(RETURN_MATRIX_CLEAN_FILE).loc[:, columns].dropna()
    residuals = residualize_against_benchmarks(
        panel, stocks=STOCKS
    )["residual_returns"]
    sessions = residuals.groupby(residuals.index.normalize(), sort=True).sum()
    sessions.index.name = "session"
    validate_panel(sessions, context="Session residual panel", require_complete=True)
    return sessions


def make_windows(panel: pd.DataFrame) -> list[Window]:
    """Split trailing windows by whole sessions, never by individual rows."""

    sessions, codes = session_index(panel.index)
    train_sessions = int(round(WINDOW * TRAIN_FRACTION))
    windows = []
    for start in rolling_starts(len(sessions), WINDOW, STEP):
        end = start + WINDOW - 1
        train_end = start + train_sessions - 1
        train = np.flatnonzero((codes >= start) & (codes <= train_end))
        test = np.flatnonzero((codes > train_end) & (codes <= end))
        if len(train) and len(test):
            windows.append(
                Window(
                    name=f"{WINDOW}_{sessions[end].date().isoformat()}",
                    start=pd.Timestamp(sessions[start]),
                    train_end=pd.Timestamp(sessions[train_end]),
                    end=pd.Timestamp(sessions[end]),
                    train=train,
                    test=test,
                )
            )
    if not windows:
        raise ValueError("Not enough sessions for the dynamic windows.")
    return windows


def spd(value: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Symmetrize and regularize a small covariance matrix."""

    value = (value + value.T) / 2.0
    minimum = float(np.linalg.eigvalsh(value).min())
    return value if minimum >= floor else value + np.eye(len(value)) * (floor - minimum)


def kalman(
    y: np.ndarray,
    H: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    transition_first: bool,
) -> dict[str, np.ndarray]:
    """Linear Kalman filter for y_t = H f_t + eps_t."""

    t_count, n_variables = y.shape
    k = H.shape[1]
    pred = np.zeros((t_count, k))
    filt = np.zeros_like(pred)
    pred_cov = np.zeros((t_count, k, k))
    filt_cov = np.zeros_like(pred_cov)
    innovations = np.zeros((t_count, n_variables))
    innovation_cov = np.zeros((t_count, n_variables, n_variables))
    eye = np.eye(k)
    for t in range(t_count):
        if t == 0 and not transition_first:
            m, P = mean, covariance
        else:
            m, P = A @ mean, A @ covariance @ A.T + Q
        P = spd(P)
        v = y[t] - H @ m
        S = spd(H @ P @ H.T + R)
        gain = np.linalg.solve(S, H @ P).T
        mean = m + gain @ v
        update = eye - gain @ H
        covariance = spd(update @ P @ update.T + gain @ R @ gain.T)
        pred[t], pred_cov[t], filt[t], filt_cov[t] = m, P, mean, covariance
        innovations[t], innovation_cov[t] = v, S
    return {
        "pred": pred,
        "pred_cov": pred_cov,
        "filt": filt,
        "filt_cov": filt_cov,
        "innovations": innovations,
        "innovation_cov": innovation_cov,
    }


def smooth(result: dict[str, np.ndarray], A: np.ndarray) -> np.ndarray:
    """RTS smoother for the training state path."""

    state = result["filt"].copy()
    state_cov = result["filt_cov"].copy()
    for t in range(len(state) - 2, -1, -1):
        gain = np.linalg.solve(
            result["pred_cov"][t + 1], A @ result["filt_cov"][t]
        ).T
        state[t] += gain @ (state[t + 1] - result["pred"][t + 1])
        state_cov[t] = spd(
            result["filt_cov"][t]
            + gain @ (state_cov[t + 1] - result["pred_cov"][t + 1]) @ gain.T
        )
    result["smooth"] = state
    result["smooth_cov"] = state_cov
    return state


def state_parameters(
    x: np.ndarray, H: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate AR(1), process noise, diagonal idiosyncratic noise, and P0."""

    factors = x @ H / len(H)
    A = np.linalg.lstsq(factors[:-1], factors[1:], rcond=None)[0].T
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    if radius > 0.98:
        A *= 0.98 / radius
    Q = spd(np.cov(factors[1:] - factors[:-1] @ A.T, rowvar=False, ddof=1))
    errors = x - factors @ H.T
    R = np.diag(np.maximum(np.var(errors, axis=0, ddof=1), 1e-8))
    P0 = spd(np.cov(factors, rowvar=False, ddof=1))
    return A, Q, R, P0, factors


def project(x: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """Reconstruct x from a possibly oblique loading basis."""

    scores = x @ loadings @ np.linalg.inv(loadings.T @ loadings)
    return scores @ loadings.T


def subspace_error(base: np.ndarray, rotated: np.ndarray) -> float:
    """Maximum projector difference: the L1 rotation must not change the space."""

    base_projection = base @ np.linalg.pinv(base.T @ base) @ base.T
    rotated_projection = rotated @ np.linalg.pinv(rotated.T @ rotated) @ rotated.T
    return float(np.max(np.abs(base_projection - rotated_projection)))


def fit_window(
    panel: pd.DataFrame,
    window: Window,
    k: int,
    starts: int,
    seed: int,
) -> dict[str, object]:
    """Fit window PCA, Kalman factors, smoothing, and the L1 rotation."""

    train = panel.iloc[window.train]
    test = panel.iloc[window.test]
    pca = fit_pca(train, method="correlation")
    x_train = pca.analysis_data.to_numpy(dtype=float)
    x_test = test.subtract(pca.means, axis="columns").divide(
        pca.scales, axis="columns"
    ).to_numpy(dtype=float)
    H = np.sqrt(len(STOCKS)) * pca.eigenvectors[:, :k]
    rotation_result = fit_l1_rotation(
        pca, n_components=k, n_starts=starts, random_state=seed
    )
    rotation = rotation_result.rotation.to_numpy(dtype=float)
    rotated = H @ rotation
    A, Q, R, P0, factors = state_parameters(x_train, H)
    train_filter = kalman(
        x_train, H, A, Q, R, factors[0], P0, transition_first=False
    )
    smooth(train_filter, A)
    test_filter = kalman(
        x_test,
        H,
        A,
        Q,
        R,
        train_filter["filt"][-1],
        train_filter["filt_cov"][-1],
        transition_first=True,
    )
    return {
        "window": window,
        "pca": pca,
        "H": H,
        "rotation": rotation,
        "rotated": rotated,
        "x_train": x_train,
        "x_test": x_test,
        "train_filter": train_filter,
        "test_filter": test_filter,
        "static_train": project(x_train, rotated),
        "static_test": project(x_test, rotated),
        "dynamic_train": train_filter["smooth"] @ H.T,
        "dynamic_test": test_filter["filt"] @ H.T,
        "one_step_test": test_filter["pred"] @ H.T,
    }


def variance(observed: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float, float]:
    """Return reconstructed variance, residual variance, and explained percent."""

    residual = observed - reconstructed
    total = float(np.square(observed).mean())
    residual_var = float(np.square(residual).mean())
    reconstructed_var = float(np.square(reconstructed).mean())
    explained = 100.0 * (1.0 - residual_var / total) if total else np.nan
    return reconstructed_var, residual_var, explained


def innovations(result: dict[str, np.ndarray]) -> tuple[float, float, float]:
    """Return raw RMSE, standardized RMS, and maximum innovation conditioning."""

    raw = float(np.sqrt(np.square(result["innovations"]).mean()))
    z = []
    condition = []
    for v, S in zip(result["innovations"], result["innovation_cov"]):
        z.append(float(v @ np.linalg.solve(S, v)) / len(v))
        condition.append(float(np.linalg.cond(S)))
    return raw, float(np.sqrt(np.mean(z))), max(condition)


def align(
    previous: np.ndarray,
    current: np.ndarray,
    states: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align loadings, states, and the final state covariance."""

    aligned, order, cosine = align_loading_columns(previous, current)
    signs = np.array(
        [
            1.0 if previous[:, i] @ current[:, j] >= 0 else -1.0
            for i, j in enumerate(order)
        ]
    )
    states = states[:, order] * signs
    covariance = covariance[np.ix_(order, order)] * signs[:, None] * signs[None, :]
    return aligned, cosine, states, covariance


def jaccard(current: np.ndarray, previous: np.ndarray, threshold: float) -> np.ndarray:
    current = np.abs(current) >= threshold
    previous = np.abs(previous) >= threshold
    return np.array(
        [
            np.logical_and(current[:, i], previous[:, i]).sum()
            / max(np.logical_or(current[:, i], previous[:, i]).sum(), 1)
            for i in range(current.shape[1])
        ]
    )


def turnover(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    current = np.abs(current) / np.abs(current).sum(axis=0, keepdims=True)
    previous = np.abs(previous) / np.abs(previous).sum(axis=0, keepdims=True)
    return 0.5 * np.abs(current - previous).sum(axis=0)


def group_metrics(window: Window, loadings: np.ndarray) -> list[dict[str, object]]:
    """Measure data-driven concentration for the three requested bank groups."""

    groups = {
        "morgan_stanley_local": ["MS"],
        "large_bank": ["JPM", "BAC", "WFC", "C"],
        "regional_bank": ["USB", "TFC", "KEY", "RF", "FITB", "CFG", "HBAN"],
    }
    rows = []
    for factor in range(loadings.shape[1]):
        absolute = np.abs(loadings[:, factor])
        for name, members in groups.items():
            inside = [STOCKS.index(stock) for stock in members]
            outside = [i for i in range(len(STOCKS)) if i not in inside]
            rows.append(
                {
                    "window_id": window.name,
                    "window_end": window.end.date().isoformat(),
                    "factor": f"LF{factor + 1}",
                    "group": name,
                    "absolute_loading_share": float(absolute[inside].sum() / absolute.sum()),
                    "group_density_ratio": float(
                        absolute[inside].mean() / max(absolute[outside].mean(), 1e-12)
                    ),
                }
            )
    return rows


def candidate_summary(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (factor, group), values in groups.groupby(["factor", "group"]):
        minimum_share = {
            "morgan_stanley_local": 0.15,
            "large_bank": 0.35,
            "regional_bank": 0.55,
        }[group]
        share = values["absolute_loading_share"]
        density = values["group_density_ratio"]
        rate = float(((share >= minimum_share) & (density >= 1.5)).mean())
        rows.append(
            {
                "factor": factor,
                "group": group,
                "median_share": float(share.median()),
                "median_density_ratio": float(density.median()),
                "recurrence_rate": rate,
                "recurring_candidate": rate >= 0.60,
            }
        )
    return pd.DataFrame(rows)


def plot_loadings(loadings: pd.DataFrame) -> None:
    pivot = loadings.pivot_table(
        index="window_end", columns=["factor", "stock"], values="loading"
    ).sort_index()
    dates = pd.to_datetime(pivot.index)
    figure, axes = plt.subplots(1, PRIMARY_K, figsize=(13, 4), sharex=True)
    for number, axis in enumerate(np.atleast_1d(axes), start=1):
        factor = f"LF{number}"
        columns = [c for c in pivot.columns if c[0] == factor]
        top = pivot[columns].abs().median().nlargest(4).index
        for _, stock in top:
            axis.plot(dates, pivot[(factor, stock)], label=stock, linewidth=1.2)
        axis.set_title(factor)
        axis.set_ylabel("Loading")
        axis.legend(frameon=False, fontsize=8, ncol=2)
        style_axis(axis, format_dates=True)
    figure.suptitle("13-bis: dynamic L1 loadings", x=0.06, ha="left")
    figure.tight_layout()
    save_figure(figure, OUT_DIR / "13_bis_loading_evolution.png", tight_bbox=True)


def main() -> None:
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_session_residuals()
    windows = make_windows(panel)

    static_pca = fit_pca(panel, method="correlation")
    static_result = fit_l1_rotation(
        static_pca, n_components=PRIMARY_K, n_starts=PRIMARY_STARTS, random_state=SEED
    )
    static_loadings = static_result.rotated_loadings.to_numpy(dtype=float)
    previous = static_loadings.copy()
    loading_rows, metric_rows, group_rows, comparison_rows = [], [], [], []

    for number, window in enumerate(windows):
        result = fit_window(panel, window, PRIMARY_K, PRIMARY_STARTS, SEED + number)
        inverse = np.linalg.inv(result["rotation"])
        rotated_states = result["test_filter"]["filt"] @ inverse.T
        rotated_covariance = (
            inverse
            @ result["test_filter"]["filt_cov"][-1]
            @ inverse.T
        )
        aligned, cosine, states, state_covariance = align(
            previous, result["rotated"], rotated_states, rotated_covariance
        )
        diagnostic = local_factor_test(aligned)
        support = jaccard(aligned, previous, diagnostic.h_n)
        changes = turnover(aligned, previous)
        innovation_rmse, innovation_z, innovation_condition = innovations(
            result["test_filter"]
        )
        state_sd = np.sqrt(np.maximum(np.diag(state_covariance), 0.0))
        train_static = variance(result["x_train"], result["static_train"])
        test_static = variance(result["x_test"], result["static_test"])
        train_dynamic = variance(result["x_train"], result["dynamic_train"])
        test_dynamic = variance(result["x_test"], result["dynamic_test"])
        test_one_step = variance(result["x_test"], result["one_step_test"])
        space_error = subspace_error(result["H"], result["rotated"])

        for factor in range(PRIMARY_K):
            static_cosine = abs(
                static_loadings[:, factor] @ aligned[:, factor]
                / (
                    np.linalg.norm(static_loadings[:, factor])
                    * np.linalg.norm(aligned[:, factor])
                )
            )
            metric_rows.append(
                {
                    "window_id": window.name,
                    "window_end": window.end.date().isoformat(),
                    "factor": f"LF{factor + 1}",
                    "l1_norm": float(np.abs(aligned[:, factor]).sum()),
                    "near_zero_count": int((np.abs(aligned[:, factor]) < SMALL_TOLERANCE).sum()),
                    "small_count": int(diagnostic.small_counts.iloc[factor]),
                    "small_loading_threshold": float(diagnostic.h_n),
                    "cosine_previous": np.nan if number == 0 else float(cosine[factor]),
                    "cosine_to_static_reference": float(static_cosine),
                    "support_jaccard_previous": np.nan if number == 0 else float(support[factor]),
                    "factor_turnover": np.nan if number == 0 else float(changes[factor]),
                    "state_estimate": float(states[-1, factor]),
                    "state_sd": float(state_sd[factor]),
                    "state_ci95_low": float(states[-1, factor] - 1.96 * state_sd[factor]),
                    "state_ci95_high": float(states[-1, factor] + 1.96 * state_sd[factor]),
                    "innovation_rmse": innovation_rmse,
                    "innovation_z_rms": innovation_z,
                    "max_innovation_condition_number": innovation_condition,
                    "rotation_condition_number": float(np.linalg.cond(result["rotation"])),
                    "subspace_preservation_error": space_error,
                    "dynamic_train_residual_variance": train_dynamic[1],
                    "dynamic_train_explained_pct": train_dynamic[2],
                    "kalman_filtered_holdout_residual_variance": test_dynamic[1],
                    "sampling_noise_warning": bool(
                        number and (cosine[factor] < 0.80 or support[factor] < 0.50)
                    ),
                }
            )
        for factor in range(PRIMARY_K):
            for stock, value in zip(STOCKS, aligned[:, factor]):
                loading_rows.append(
                    {
                        "window_id": window.name,
                        "window_end": window.end.date().isoformat(),
                        "factor": f"LF{factor + 1}",
                        "stock": stock,
                        "loading": float(value),
                    }
                )
        group_rows.extend(group_metrics(window, aligned))
        comparison_rows.append(
            {
                "window_id": window.name,
                "window_end": window.end.date().isoformat(),
                "static_train_residual_variance": train_static[1],
                "static_train_explained_pct": train_static[2],
                "static_holdout_residual_variance": test_static[1],
                "static_holdout_explained_pct": test_static[2],
                "dynamic_train_residual_variance": train_dynamic[1],
                "dynamic_train_explained_pct": train_dynamic[2],
                "kalman_filtered_holdout_residual_variance": test_dynamic[1],
                "kalman_filtered_holdout_explained_pct": test_dynamic[2],
                "kalman_one_step_holdout_residual_variance": test_one_step[1],
                "kalman_one_step_holdout_explained_pct": test_one_step[2],
                "subspace_preservation_error": space_error,
            }
        )
        previous = aligned

    loadings = pd.DataFrame(loading_rows)
    metrics = pd.DataFrame(metric_rows)
    groups = pd.DataFrame(group_rows)
    comparisons = pd.DataFrame(comparison_rows)
    candidates = candidate_summary(groups)

    sensitivity_rows = []
    for k in SENSITIVITY_K:
        for number, window in enumerate(windows[::SENSITIVITY_EVERY]):
            result = fit_window(
                panel, window, k, SENSITIVITY_STARTS, SEED + 1000 * k + number
            )
            diagnostic = local_factor_test(result["rotated"])
            static_holdout = variance(result["x_test"], result["static_test"])
            dynamic_holdout = variance(result["x_test"], result["dynamic_test"])
            one_step = variance(result["x_test"], result["one_step_test"])
            sensitivity_rows.append(
                {
                    "k": k,
                    "window_id": window.name,
                    "has_local_factors": diagnostic.has_local_factors,
                    "small_count_max": int(diagnostic.small_counts.max()),
                    "total_l1_norm": float(np.abs(result["rotated"]).sum()),
                    "rotation_condition_number": float(np.linalg.cond(result["rotation"])),
                    "static_holdout_residual_variance": static_holdout[1],
                    "kalman_filtered_holdout_residual_variance": dynamic_holdout[1],
                    "kalman_one_step_holdout_residual_variance": one_step[1],
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)

    regional = groups[groups["group"] == "regional_bank"].copy()
    regional["crisis"] = pd.to_datetime(regional["window_end"]).between(
        CRISIS_START, CRISIS_END
    )
    stress_rows = []
    for factor, values in regional.groupby("factor"):
        crisis = values[values["crisis"]]
        other = values[~values["crisis"]]
        crisis_share = crisis["absolute_loading_share"].median()
        other_share = other["absolute_loading_share"].median()
        stress_rows.append(
            {
                "factor": factor,
                "crisis_windows": len(crisis),
                "other_windows": len(other),
                "regional_share_crisis_median": crisis_share,
                "regional_share_other_median": other_share,
                "delta_crisis_minus_other": crisis_share - other_share,
                "stronger_during_crisis": bool(
                    len(crisis) and len(other) and crisis_share > other_share
                ),
            }
        )
    stress = pd.DataFrame(stress_rows)

    stable = metrics.dropna(subset=["cosine_previous"])
    full_x = static_pca.analysis_data.to_numpy(dtype=float)
    static_full = variance(full_x, project(full_x, static_loadings))
    summary = pd.DataFrame(
        [
            {
                "method": "Static PCA + L1 (full sample)",
                "n_windows": 1,
                "reconstruction_pct": static_full[2],
                "holdout_residual_variance_median": np.nan,
                "one_step_residual_variance_median": np.nan,
                "cosine_median": 1.0,
                "support_jaccard_median": 1.0,
                "turnover_median": 0.0,
                "max_subspace_preservation_error": 0.0,
            },
            {
                "method": "Static PCA + L1 (rolling holdout)",
                "n_windows": len(comparisons),
                "reconstruction_pct": comparisons["static_train_explained_pct"].mean(),
                "holdout_residual_variance_median": comparisons["static_holdout_residual_variance"].median(),
                "one_step_residual_variance_median": np.nan,
                "cosine_median": stable["cosine_previous"].median(),
                "support_jaccard_median": stable["support_jaccard_previous"].median(),
                "turnover_median": stable["factor_turnover"].median(),
                "max_subspace_preservation_error": comparisons["subspace_preservation_error"].max(),
            },
            {
                "method": "Kalman dynamic factors + L1",
                "n_windows": len(comparisons),
                "reconstruction_pct": comparisons["dynamic_train_explained_pct"].mean(),
                "holdout_residual_variance_median": comparisons["kalman_filtered_holdout_residual_variance"].median(),
                "one_step_residual_variance_median": comparisons["kalman_one_step_holdout_residual_variance"].median(),
                "cosine_median": stable["cosine_previous"].median(),
                "support_jaccard_median": stable["support_jaccard_previous"].median(),
                "turnover_median": stable["factor_turnover"].median(),
                "max_subspace_preservation_error": comparisons["subspace_preservation_error"].max(),
            },
        ]
    )

    static_reference = pd.DataFrame(
        [
            {"factor": f"LF{i + 1}", "stock": stock, "loading": static_loadings[row, i]}
            for i in range(PRIMARY_K)
            for row, stock in enumerate(STOCKS)
        ]
    )
    outputs = {
        "13_bis_dynamic_loadings.csv": loadings,
        "13_bis_dynamic_factor_metrics.csv": metrics,
        "13_bis_static_vs_kalman_windows.csv": comparisons,
        "13_bis_comparison_summary.csv": summary,
        "13_bis_k_sensitivity.csv": sensitivity,
        "13_bis_structural_group_metrics.csv": groups,
        "13_bis_structural_candidates.csv": candidates,
        "13_bis_regional_stress_comparison.csv": stress,
        "13_bis_static_reference_loadings.csv": static_reference,
    }
    for filename, table in outputs.items():
        table.to_csv(OUT_DIR / filename, index=False)
    plot_loadings(loadings)

    dynamic = summary.iloc[2]
    recurring = candidates[candidates["recurring_candidate"]]
    answer = (
        f"Kalman dynamic factors: median filtered holdout residual variance "
        f"{dynamic['holdout_residual_variance_median']:.6g}, one-step "
        f"{dynamic['one_step_residual_variance_median']:.6g}; loading cosine "
        f"{dynamic['cosine_median']:.3f}, support Jaccard "
        f"{dynamic['support_jaccard_median']:.3f}. Recurring candidates: "
        f"{len(recurring)}; group labels remain data-dependent."
    )
    (OUT_DIR / "13_bis_summary.txt").write_text(answer + "\n", encoding="utf-8")
    print("=== 13-BIS KALMAN DYNAMIC FACTORS + L1 ===")
    print(f"Session residual panel: {panel.shape}; windows: {len(windows)}")
    print(summary.round(5).to_string(index=False))
    print("\nRecurring candidates:")
    print(recurring.to_string(index=False) if len(recurring) else "none above rule")
    print("\nRegional stress:")
    print(stress.round(4).to_string(index=False))
    print(f"\n{answer}\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
