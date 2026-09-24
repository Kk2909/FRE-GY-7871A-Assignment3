from pathlib import Path

import pandas as pd
import statsmodels.api as sm


RISK_PATH = Path("data/processed/daily_iran_war_risk.csv")
MARKET_PATH = Path("data/processed/market_changes.csv")

SUMMARY_PATH = Path(
    "outputs/tables/three_regime_market_summary.csv"
)
REGRESSION_PATH = Path(
    "outputs/tables/three_regime_regressions.csv"
)

VARIABLES = {
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


def classify_regime(
    row: pd.Series,
    threshold: float,
) -> str:
    if row["war_risk_index"] < threshold:
        return "No or low war news"

    if row["weighted_direction"] >= 0:
        return "Bad war news"

    return "Good war news"


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

    directional_threshold = data[
        "war_risk_index"
    ].quantile(0.75)

    data["directional_regime"] = data.apply(
        classify_regime,
        axis=1,
        threshold=directional_threshold,
    )

    summary_rows = []

    regime_order = [
        "Bad war news",
        "Good war news",
        "No or low war news",
    ]

    for variable, label in VARIABLES.items():
        for regime in regime_order:
            values = data.loc[
                data["directional_regime"] == regime,
                variable,
            ].dropna()

            summary_rows.append(
                {
                    "variable": variable,
                    "label": label,
                    "regime": regime,
                    "observations": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "variance": values.var(ddof=0),
                    "standard_deviation": values.std(ddof=0),
                }
            )

    summary = pd.DataFrame(summary_rows)

    regression_rows = []

    data["bad_news_dummy"] = (
        data["directional_regime"] == "Bad war news"
    ).astype(int)

    data["good_news_dummy"] = (
        data["directional_regime"] == "Good war news"
    ).astype(int)

    for variable, label in VARIABLES.items():
        regression_data = data[
            [
                variable,
                "bad_news_dummy",
                "good_news_dummy",
            ]
        ].dropna()

        explanatory_variables = sm.add_constant(
            regression_data[
                ["bad_news_dummy", "good_news_dummy"]
            ]
        )

        model = sm.OLS(
            regression_data[variable],
            explanatory_variables,
        ).fit(cov_type="HC3")

        for term in [
            "bad_news_dummy",
            "good_news_dummy",
        ]:
            regression_rows.append(
                {
                    "variable": variable,
                    "label": label,
                    "term": term,
                    "coefficient": model.params[term],
                    "standard_error": model.bse[term],
                    "p_value": model.pvalues[term],
                    "confidence_low": model.conf_int().loc[
                        term, 0
                    ],
                    "confidence_high": model.conf_int().loc[
                        term, 1
                    ],
                    "r_squared": model.rsquared,
                    "observations": int(model.nobs),
                }
            )

    regressions = pd.DataFrame(regression_rows)

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    regressions.to_csv(
        REGRESSION_PATH,
        index=False,
    )

    print(
        "Directional threshold "
        f"(75th percentile): {directional_threshold:.4f}"
    )

    print("\nRegime counts:")
    print(
        data["directional_regime"]
        .value_counts()
        .reindex(regime_order, fill_value=0)
        .to_string()
    )

    print("\nThree-regime market summary:")
    display_summary = summary.copy()

    numeric_columns = [
        "mean",
        "median",
        "variance",
        "standard_deviation",
    ]

    display_summary[numeric_columns] = (
        display_summary[numeric_columns].round(4)
    )

    print(display_summary.to_string(index=False))

    print("\nThree-regime regressions:")
    display_regressions = regressions.copy()

    regression_numeric_columns = [
        "coefficient",
        "standard_error",
        "p_value",
        "confidence_low",
        "confidence_high",
        "r_squared",
    ]

    display_regressions[
        regression_numeric_columns
    ] = display_regressions[
        regression_numeric_columns
    ].round(4)

    print(display_regressions.to_string(index=False))

    print(f"\nSummary saved to: {SUMMARY_PATH}")
    print(f"Regressions saved to: {REGRESSION_PATH}")


if __name__ == "__main__":
    main()