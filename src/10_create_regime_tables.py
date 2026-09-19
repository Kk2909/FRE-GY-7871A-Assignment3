from pathlib import Path

import numpy as np
import pandas as pd


DAILY_PATH = Path(
    "data/processed/daily_iran_war_risk.csv"
)

SAMPLE_PATH = Path(
    "data/processed/estimation_sample.csv"
)

REGIME_TABLE_PATH = Path(
    "outputs/tables/table1_high_and_low_news_regimes.csv"
)

APPENDIX_PATH = Path(
    "outputs/tables/appendix_all_daily_war_risk.csv"
)


def classify_direction(
    value: float,
    neutral_band: float,
) -> str:
    if value > neutral_band:
        return "Escalation"

    if value < -neutral_band:
        return "De-escalation"

    return "Mixed or unclear"


def main():
    daily = pd.read_csv(
        DAILY_PATH,
        parse_dates=["market_date"],
    )

    sample = pd.read_csv(
        SAMPLE_PATH,
        parse_dates=["date"],
    )

    neutral_band = (
        0.20
        * daily["weighted_direction"].std(ddof=0)
    )

    daily["News Direction"] = daily[
        "weighted_direction"
    ].apply(
        lambda value: classify_direction(
            value,
            neutral_band,
        )
    )

    high_dates = set(
        sample.loc[
            sample["regime"] == "high",
            "date",
        ]
    )

    low_dates = set(
        sample.loc[
            sample["regime"] == "low",
            "date",
        ]
    )

    daily["Regime"] = "Other"

    daily.loc[
        daily["market_date"].isin(high_dates),
        "Regime",
    ] = "High variance"

    daily.loc[
        daily["market_date"].isin(low_dates),
        "Regime",
    ] = "Matched low variance"

    appendix = daily[
        [
            "market_date",
            "Regime",
            "war_risk_index",
            "News Direction",
            "article_count",
            "mean_relevance",
            "top10_intensity",
            "uncertainty_rate",
            "top_article",
            "top_source",
        ]
    ].copy()

    appendix = appendix.rename(
        columns={
            "market_date": "Date",
            "war_risk_index": "NLP Risk Score",
            "article_count": "Article Count",
            "mean_relevance": "Mean Relevance",
            "top10_intensity": "Top-10 Intensity",
            "uncertainty_rate": "Uncertainty Rate",
            "top_article": "Primary Event",
            "top_source": "Primary Source",
        }
    )

    numeric_columns = [
        "NLP Risk Score",
        "Mean Relevance",
        "Top-10 Intensity",
        "Uncertainty Rate",
    ]

    appendix[numeric_columns] = (
        appendix[numeric_columns].round(4)
    )

    appendix = appendix.sort_values(
        "Date"
    ).reset_index(drop=True)

    regime_table = appendix[
        appendix["Regime"].isin(
            [
                "High variance",
                "Matched low variance",
            ]
        )
    ].copy()

    regime_order = {
        "High variance": 0,
        "Matched low variance": 1,
    }

    regime_table["regime_order"] = (
        regime_table["Regime"].map(regime_order)
    )

    regime_table = (
        regime_table.sort_values(
            ["regime_order", "Date"]
        )
        .drop(columns=["regime_order"])
        .reset_index(drop=True)
    )

    REGIME_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    regime_table.to_csv(
        REGIME_TABLE_PATH,
        index=False,
    )

    appendix.to_csv(
        APPENDIX_PATH,
        index=False,
    )

    print(
        "Table 1: High- and Low-News "
        "Heteroskedasticity Regimes\n"
    )

    print(
        regime_table[
            [
                "Date",
                "Regime",
                "NLP Risk Score",
                "News Direction",
                "Article Count",
                "Primary Event",
                "Primary Source",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nMain regime table rows: "
        f"{len(regime_table)}"
    )

    print(
        f"Full daily appendix rows: "
        f"{len(appendix)}"
    )

    print(
        f"\nRegime table saved to: "
        f"{REGIME_TABLE_PATH}"
    )

    print(
        f"Daily appendix saved to: "
        f"{APPENDIX_PATH}"
    )


if __name__ == "__main__":
    main()