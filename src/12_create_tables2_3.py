from pathlib import Path

import numpy as np
import pandas as pd


ESTIMATES_PATH = Path(
    "outputs/tables/war_risk_estimates.csv"
)

SAMPLE_PATH = Path(
    "data/processed/estimation_sample.csv"
)

FULL_MARKET_PATH = Path(
    "data/processed/market_changes.csv"
)

TABLE_2_PATH = Path(
    "outputs/tables/table2_estimated_market_effects.csv"
)

TABLE_3_PATH = Path(
    "outputs/tables/table3_variance_explained.csv"
)

ANCHOR = "d_five_year_yield"
ANCHOR_SHOCK = 0.10

VARIABLE_LABELS = {
    "d_five_year_yield": "5-Year Treasury Yield",
    "d_ten_year_yield": "10-Year Treasury Yield",
    "r_sp500": "S&P 500",
    "r_global_equity": "Global Equity",
    "r_investment_grade_bonds": "Investment-Grade Bonds",
    "r_high_yield_bonds": "High-Yield Bonds",
    "r_inflation_linked_bonds": "Inflation-Linked Bonds",
    "r_brent_oil": "Brent Oil",
    "r_gold": "Gold",
    "r_dollar_index": "U.S. Dollar Index",
    "r_vix": "VIX",
    "r_ig_credit_proxy": "IG Credit Excess-Return Proxy",
    "r_hy_credit_proxy": "HY Credit Excess-Return Proxy",
}


def population_variance(series: pd.Series) -> float:
    values = series.dropna().to_numpy(dtype=float)

    if len(values) == 0:
        return np.nan

    return np.var(values, ddof=0)


def create_table_2(
    estimates: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in estimates.iterrows():
        variable = row["variable"]

        effect_1 = (
            row["estimator_1"]
            * ANCHOR_SHOCK
        )

        effect_2 = (
            row["estimator_2"]
            * ANCHOR_SHOCK
        )

        combined_effect = (
            row["combined_estimator"]
            * ANCHOR_SHOCK
        )

        confidence_low = row["confidence_low"]
        confidence_high = row["confidence_high"]

        if variable == "d_ten_year_yield":
            unit = "Basis-point change"

            effect_1 *= 100
            effect_2 *= 100
            combined_effect *= 100
            confidence_low *= 100
            confidence_high *= 100

        else:
            unit = "Percent change"

        rows.append(
            {
                "Financial Variable": (
                    VARIABLE_LABELS[variable]
                ),
                "Units": unit,
                "Estimator 1": effect_1,
                "Estimator 2": effect_2,
                "Combined IV": combined_effect,
                "95% CI Low": confidence_low,
                "95% CI High": confidence_high,
                "Significant at 5%": row[
                    "statistically_significant"
                ],
            }
        )

    table_2 = pd.DataFrame(rows)

    numeric_columns = [
        "Estimator 1",
        "Estimator 2",
        "Combined IV",
        "95% CI Low",
        "95% CI High",
    ]

    table_2[numeric_columns] = (
        table_2[numeric_columns].round(3)
    )

    return table_2


def create_table_3(
    estimates: pd.DataFrame,
    sample: pd.DataFrame,
    full_market: pd.DataFrame,
) -> pd.DataFrame:
    high = sample[
        sample["regime"] == "high"
    ]

    low = sample[
        sample["regime"] == "low"
    ]

    variance_anchor_high = population_variance(
        high[ANCHOR]
    )

    variance_anchor_low = population_variance(
        low[ANCHOR]
    )

    delta_variance_anchor = (
        variance_anchor_high
        - variance_anchor_low
    )

    rows = [
        {
            "Financial Variable": (
                VARIABLE_LABELS[ANCHOR]
            ),
            "Variance on Low Days": (
                variance_anchor_low
            ),
            "Variance on High Days": (
                variance_anchor_high
            ),
            "Predicted Change in Variance": np.nan,
            "Explained on High Days (%)": np.nan,
            "Explained in Full Sample (%)": np.nan,
        }
    ]

    for _, estimate in estimates.iterrows():
        variable = estimate["variable"]

        variance_high = population_variance(
            high[variable]
        )

        variance_low = population_variance(
            low[variable]
        )

        variance_full = population_variance(
            full_market[variable]
        )

        beta = estimate["combined_estimator"]

        predicted_variance_change = (
            beta**2
            * delta_variance_anchor
        )

        explained_high = (
            100
            * predicted_variance_change
            / variance_high
            if variance_high > 0
            else np.nan
        )

        number_high = high[variable].notna().sum()
        number_full = (
            full_market[variable]
            .notna()
            .sum()
        )

        explained_full = (
            100
            * number_high
            * predicted_variance_change
            / (number_full * variance_full)
            if variance_full > 0
            else np.nan
        )

        rows.append(
            {
                "Financial Variable": (
                    VARIABLE_LABELS[variable]
                ),
                "Variance on Low Days": variance_low,
                "Variance on High Days": variance_high,
                "Predicted Change in Variance": (
                    predicted_variance_change
                ),
                "Explained on High Days (%)": (
                    explained_high
                ),
                "Explained in Full Sample (%)": (
                    explained_full
                ),
            }
        )

    table_3 = pd.DataFrame(rows)

    variance_columns = [
        "Variance on Low Days",
        "Variance on High Days",
        "Predicted Change in Variance",
    ]

    percentage_columns = [
        "Explained on High Days (%)",
        "Explained in Full Sample (%)",
    ]

    table_3[variance_columns] = (
        table_3[variance_columns].round(5)
    )

    table_3[percentage_columns] = (
        table_3[percentage_columns].round(2)
    )

    return table_3


def main():
    estimates = pd.read_csv(ESTIMATES_PATH)

    sample = pd.read_csv(SAMPLE_PATH)

    full_market = pd.read_csv(
        FULL_MARKET_PATH
    )

    table_2 = create_table_2(estimates)

    table_3 = create_table_3(
        estimates,
        sample,
        full_market,
    )

    TABLE_2_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_2.to_csv(
        TABLE_2_PATH,
        index=False,
    )

    table_3.to_csv(
        TABLE_3_PATH,
        index=False,
    )

    print(
        "Table 2: Estimated Impact of "
        "an Increase in Iran War Risk\n"
    )

    print(table_2.to_string(index=False))

    print(
        "\nTable 3: Variances of "
        "Financial Variables\n"
    )

    print(table_3.to_string(index=False))

    print(f"\nTable 2 saved to: {TABLE_2_PATH}")
    print(f"Table 3 saved to: {TABLE_3_PATH}")


if __name__ == "__main__":
    main()