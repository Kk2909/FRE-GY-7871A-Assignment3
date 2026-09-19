from pathlib import Path

import pandas as pd
import statsmodels.api as sm


RISK_PATH = Path(
    "data/processed/daily_iran_war_risk.csv"
)

MARKET_PATH = Path(
    "data/processed/market_changes.csv"
)

OUTPUT_PATH = Path(
    "outputs/tables/direct_nlp_regressions.csv"
)

MARKET_VARIABLES = {
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
}


def standardize(series: pd.Series) -> pd.Series:
    standard_deviation = series.std(ddof=0)

    if standard_deviation == 0:
        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        series - series.mean()
    ) / standard_deviation


def estimate_regression(
    dependent: pd.Series,
    independent: pd.Series,
) -> dict:
    regression_data = pd.concat(
        [dependent, independent],
        axis=1,
    ).dropna()

    y = regression_data.iloc[:, 0]

    x = sm.add_constant(
        regression_data.iloc[:, 1]
    )

    model = sm.OLS(
        y,
        x,
    ).fit(cov_type="HC3")

    variable_name = regression_data.columns[1]

    return {
        "observations": int(model.nobs),
        "coefficient": model.params[variable_name],
        "standard_error": model.bse[variable_name],
        "t_statistic": model.tvalues[variable_name],
        "p_value": model.pvalues[variable_name],
        "r_squared": model.rsquared,
    }


def main():
    risk = pd.read_csv(
        RISK_PATH,
        parse_dates=["market_date"],
    )

    market = pd.read_csv(
        MARKET_PATH,
        parse_dates=["date"],
    )

    data = market.merge(
        risk,
        left_on="date",
        right_on="market_date",
        how="inner",
    )

    # Positive values represent stronger escalation language.
    data["signed_risk_z"] = standardize(
        data["weighted_direction"]
    )

    # Higher values represent more intense or uncertain news.
    data["risk_intensity_z"] = standardize(
        data["war_risk_index"]
    )

    results = []

    for variable, label in MARKET_VARIABLES.items():
        directional = estimate_regression(
            dependent=data[variable],
            independent=data["signed_risk_z"],
        )

        volatility = estimate_regression(
            dependent=data[variable].abs(),
            independent=data["risk_intensity_z"],
        )

        results.append(
            {
                "variable": variable,
                "label": label,
                "observations": directional[
                    "observations"
                ],
                "direction_coefficient": directional[
                    "coefficient"
                ],
                "direction_standard_error": directional[
                    "standard_error"
                ],
                "direction_t_statistic": directional[
                    "t_statistic"
                ],
                "direction_p_value": directional[
                    "p_value"
                ],
                "direction_r_squared": directional[
                    "r_squared"
                ],
                "intensity_coefficient": volatility[
                    "coefficient"
                ],
                "intensity_standard_error": volatility[
                    "standard_error"
                ],
                "intensity_t_statistic": volatility[
                    "t_statistic"
                ],
                "intensity_p_value": volatility[
                    "p_value"
                ],
                "intensity_r_squared": volatility[
                    "r_squared"
                ],
            }
        )

    results = pd.DataFrame(results)

    results["direction_significant"] = (
        results["direction_p_value"] < 0.05
    )

    results["intensity_significant"] = (
        results["intensity_p_value"] < 0.05
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    directional_display = results[
        [
            "label",
            "direction_coefficient",
            "direction_standard_error",
            "direction_p_value",
            "direction_r_squared",
            "direction_significant",
        ]
    ].copy()

    volatility_display = results[
        [
            "label",
            "intensity_coefficient",
            "intensity_standard_error",
            "intensity_p_value",
            "intensity_r_squared",
            "intensity_significant",
        ]
    ].copy()

    directional_numeric = (
        directional_display
        .select_dtypes(include="number")
        .columns
    )

    directional_display[directional_numeric] = (
        directional_display[
            directional_numeric
        ].round(4)
    )

    volatility_numeric = (
        volatility_display
        .select_dtypes(include="number")
        .columns
    )

    volatility_display[volatility_numeric] = (
        volatility_display[
            volatility_numeric
        ].round(4)
    )

    print(
        "Directional NLP regression\n"
        "Response to a one-standard-deviation increase "
        "in escalation language:\n"
    )

    print(
        directional_display.to_string(index=False)
    )

    print(
        "\nNLP intensity regression\n"
        "Change in absolute market movement associated "
        "with a one-standard-deviation increase "
        "in news intensity:\n"
    )

    print(
        volatility_display.to_string(index=False)
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()