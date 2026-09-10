import numpy as np
import pandas as pd

from config import RETURN_MATRIX_FILE

r = pd.read_parquet(RETURN_MATRIX_FILE)

print("\n=== RETURN SUMMARY ===\n")

summary = pd.DataFrame({
    "mean": r.mean(),
    "std": r.std(),
    "min": r.min(),
    "max": r.max(),
    "p001": r.quantile(0.001),
    "p999": r.quantile(0.999),
})

print(summary)


print("\n=== LARGEST ABSOLUTE RETURNS ===\n")

stacked = r.stack()

largest = (
    stacked
    .abs()
    .sort_values(ascending=False)
    .head(50)
)

for (timestamp, symbol), abs_return in largest.items():

    ret = r.loc[timestamp, symbol]

    print(
        timestamp,
        symbol,
        f"return={ret:.6f}",
        f"pct={100 * (np.exp(ret) - 1):.3f}%"
    )
