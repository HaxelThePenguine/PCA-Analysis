"""Intraday-volatility normalization and PCA robustness check."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CORE_UNIVERSE, REPORTS_DIR, RETURN_CORE_FILE, ensure_project_directories


OUT_DIR = REPORTS_DIR / "pca_intraday"
BLUE = "#2F6B9A"
GRID = "#D9DEE5"


def run_pca(matrix):
    values, vectors = np.linalg.eigh(matrix.to_numpy())
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    # The sign is arbitrary; orient the largest coefficient positively.
    for component in range(vectors.shape[1]):
        pivot = np.argmax(np.abs(vectors[:, component]))
        if vectors[pivot, component] < 0:
            vectors[:, component] *= -1

    explained = values / values.sum()
    labels = [f"PC{i}" for i in range(1, len(values) + 1)]
    summary = pd.DataFrame(
        {
            "eigenvalue": values,
            "explained_pct": explained * 100,
            "cumulative_pct": explained.cumsum() * 100,
        },
        index=labels,
    )
    loadings = pd.DataFrame(
        vectors[:, :3],
        index=matrix.columns,
        columns=["PC1", "PC2", "PC3"],
    )
    return values, vectors, explained, summary, loadings


def save_profile_plot(profile):
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(profile.index, profile["mean_volatility"], color=BLUE, linewidth=1.6)
    axis.set_title("Volatilità intraday media del pannello CORE")
    axis.set_xlabel("Minuti dall'apertura")
    axis.set_ylabel("Deviazione standard del rendimento")
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_intraday_volatility_profile.png", dpi=180)
    plt.close(fig)


def main():
    ensure_project_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    returns = pd.read_parquet(RETURN_CORE_FILE)
    missing = [symbol for symbol in CORE_UNIVERSE if symbol not in returns.columns]
    if missing:
        raise ValueError(f"CORE incompleto; simboli mancanti: {missing}")
    returns = returns.loc[:, list(CORE_UNIVERSE)]

    if returns.isna().any().any():
        raise ValueError("Il pannello CORE contiene NaN.")

    minute_from_open = pd.Series(
        returns.index.hour * 60 + returns.index.minute - (9 * 60 + 30),
        index=returns.index,
        name="minute_from_open",
    )
    volatility_profile = returns.groupby(minute_from_open).std(ddof=1)
    observations = minute_from_open.value_counts().sort_index()
    profile = pd.DataFrame(
        {
            "mean_volatility": volatility_profile.mean(axis=1),
            "min_volatility": volatility_profile.min(axis=1),
            "max_volatility": volatility_profile.max(axis=1),
            "observations": observations,
        }
    )
    profile.index.name = "minute_from_open"

    # Each row is divided by the volatility of that ticker at that minute.
    scale = volatility_profile.loc[minute_from_open.to_numpy()].copy()
    scale.index = returns.index
    if scale.isna().any().any() or (scale <= 0).any().any():
        raise ValueError("Profilo di volatilità non valido.")

    normalized_returns = returns.divide(scale)
    if not np.isfinite(normalized_returns.to_numpy()).all():
        raise ValueError("I rendimenti normalizzati contengono valori non finiti.")

    normalized_profile = normalized_returns.groupby(minute_from_open).std(ddof=1)
    normalization_error = np.max(np.abs(normalized_profile.to_numpy() - 1))

    centered = normalized_returns.subtract(
        normalized_returns.mean(),
        axis="columns",
    )
    covariance = (centered.T @ centered) / (len(centered) - 1)
    covariance_diff = (covariance - centered.cov()).abs().to_numpy().max()
    values, vectors, explained, summary, loadings = run_pca(covariance)

    scores = centered.to_numpy() @ vectors
    score_error = np.max(
        np.abs(values - pd.DataFrame(scores).var(ddof=1).to_numpy())
    )

    tables = {
        "08_intraday_volatility_profile.csv": profile,
        "08_normalized_return_stats.csv": pd.DataFrame(
            {
                "mean": normalized_returns.mean(),
                "std": normalized_returns.std(ddof=1),
            }
        ),
        "08_normalized_covariance_matrix.csv": covariance,
        "08_normalized_pca_summary.csv": summary,
        "08_normalized_pca_loadings_pc1_pc3.csv": loadings,
    }
    for filename, table in tables.items():
        table.to_csv(OUT_DIR / filename)
    save_profile_plot(profile)

    print("=== INTRADAY NORMALIZATION ===")
    print(f"Panel: {returns.shape} | NaN: 0")
    print(f"Profilo: minuto {profile.index.min()} -> {profile.index.max()}")
    print(f"Osservazioni per minuto: {observations.min()} -> {observations.max()}")
    print(f"Controllo volatilita normalizzata: {normalization_error:.3e}")
    print(f"Covariance check: {covariance_diff:.3e}")
    print(f"Score/eigenvalue check: {score_error:.3e}")
    print(f"Somma autovalori: {values.sum():.8f}")
    print(f"Prime 3 componenti: {explained[:3].sum() * 100:.4f}% spiegato")

    print("\n=== PROFILO INTRADAY, INIZIO/FINE ===")
    print(pd.concat([profile.head(3), profile.tail(3)]).round(6).to_string())
    print("\n=== PCA NORMALIZZATA ===")
    print(summary.round(4).to_string())
    print("\nCoefficienti PC1-PC3:")
    print(loadings.round(4).to_string())
    print(f"\nOutput salvati in: {OUT_DIR}")


if __name__ == "__main__":
    main()
