from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import levene


INPUT_PATH = Path("data/processed/estimation_sample.csv")
OUTPUT_PATH = Path("outputs/tables/variance_regime_diagnostics.csv")

MARKET_VARIABLES = [
    "d_five_year_yield",
    "d_ten_year_yield",
    "r_sp500",
    "r_global_equity",
    "r_investment_grade_bonds",
    "r_high_yield_bonds",
    "r_inflation_linked_bonds",
    "r_brent_oil",
    "r_gold",
    "r_dollar_index",
    "r_vix",
]


def main():
    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["date"],
    )

    results = []

    for variable in MARKET_VARIABLES:
        high = (
            data.loc[data["regime"] == "high", variable]
            .dropna()
            .to_numpy()
        )

        low = (
            data.loc[data["regime"] == "low", variable]
            .dropna()
            .to_numpy()
        )

        variance_high = np.var(high, ddof=0)
        variance_low = np.var(low, ddof=0)

        variance_difference = (
            variance_high - variance_low
        )

        variance_ratio = (
            variance_high / variance_low
            if variance_low > 0
            else np.nan
        )

        if len(high) >= 2 and len(low) >= 2:
            _, levene_pvalue = levene(
                high,
                low,
                center="median",
            )
        else:
            levene_pvalue = np.nan

        results.append(
            {
                "variable": variable,
                "high_observations": len(high),
                "low_observations": len(low),
                "variance_low": variance_low,
                "variance_high": variance_high,
                "variance_difference": variance_difference,
                "variance_ratio": variance_ratio,
                "levene_pvalue": levene_pvalue,
            }
        )

    diagnostics = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(OUTPUT_PATH, index=False)

    print("Variance regime diagnostics:\n")

    display = diagnostics.copy()

    numeric_columns = [
        "variance_low",
        "variance_high",
        "variance_difference",
        "variance_ratio",
        "levene_pvalue",
    ]

    display[numeric_columns] = (
        display[numeric_columns].round(4)
    )

    print(display.to_string(index=False))

    increased = (
        diagnostics["variance_difference"] > 0
    ).sum()

    print(
        f"\nVariables with higher variance on "
        f"high-news days: {increased}/"
        f"{len(diagnostics)}"
    )

    anchor = diagnostics[
        diagnostics["variable"]
        == "d_five_year_yield"
    ].iloc[0]

    print("\nReference-variable check:")
    print(
        f"Five-year yield variance ratio: "
        f"{anchor['variance_ratio']:.3f}"
    )
    print(
        f"Five-year yield variance difference: "
        f"{anchor['variance_difference']:.6f}"
    )

    if anchor["variance_difference"] <= 0:
        print(
            "WARNING: The reference-variable variance "
            "did not increase on high-news days."
        )
    else:
        print(
            "The reference variable satisfies the required "
            "variance-direction condition."
        )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()