from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


RISK_PATH = Path("data/processed/daily_iran_war_risk.csv")
MARKET_PATH = Path("data/processed/market_changes.csv")

SAMPLE_PATH = Path("data/processed/estimation_sample.csv")
MATCHES_PATH = Path("outputs/tables/matched_news_days.csv")


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

    data = (
        data.sort_values("date")
        .reset_index(drop=True)
    )

    data["trading_position"] = np.arange(len(data))

    high_days = data[
        data["high_news_day"].astype(bool)
    ].copy()

    low_candidates = data[
        data["low_news_candidate"].astype(bool)
        & ~data["high_news_day"].astype(bool)
    ].copy()

    if len(low_candidates) < len(high_days):
        raise ValueError(
            "There are not enough low-news candidates."
        )

    high_positions = (
        high_days["trading_position"]
        .to_numpy()
        .reshape(-1, 1)
    )

    low_positions = (
        low_candidates["trading_position"]
        .to_numpy()
        .reshape(1, -1)
    )

    # The main matching criterion is proximity in trading days.
    # Lower risk-index values break ties between nearby candidates.
    distance_cost = np.abs(
        high_positions - low_positions
    ).astype(float)

    risk_tiebreaker = (
        low_candidates["war_risk_index"]
        .to_numpy()
        .reshape(1, -1)
    )

    cost_matrix = (
        distance_cost
        + 0.001 * risk_tiebreaker
    )

    high_rows, low_columns = linear_sum_assignment(
        cost_matrix
    )

    matched_high = high_days.iloc[
        high_rows
    ].reset_index(drop=True)

    matched_low = low_candidates.iloc[
        low_columns
    ].reset_index(drop=True)

    matches = pd.DataFrame(
        {
            "high_date": matched_high["date"],
            "high_risk_index": matched_high["war_risk_index"],
            "high_article": matched_high["top_article"],
            "low_date": matched_low["date"],
            "low_risk_index": matched_low["war_risk_index"],
            "trading_day_distance": np.abs(
                matched_high["trading_position"].to_numpy()
                - matched_low["trading_position"].to_numpy()
            ),
        }
    )

    matches = matches.sort_values(
        "high_date"
    ).reset_index(drop=True)

    data["regime"] = "other"

    data.loc[
        data["date"].isin(matches["high_date"]),
        "regime",
    ] = "high"

    data.loc[
        data["date"].isin(matches["low_date"]),
        "regime",
    ] = "low"

    estimation_sample = data[
        data["regime"].isin(["high", "low"])
    ].copy()

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)

    estimation_sample.to_csv(
        SAMPLE_PATH,
        index=False,
    )

    matches.to_csv(
        MATCHES_PATH,
        index=False,
    )

    print(f"Merged market dates: {len(data):,}")
    print(f"Matched high-news days: {len(matched_high):,}")
    print(f"Matched low-news days: {len(matched_low):,}")

    print(
        "Median matching distance: "
        f"{matches['trading_day_distance'].median():.0f} "
        "trading days"
    )

    print(
        "Maximum matching distance: "
        f"{matches['trading_day_distance'].max():.0f} "
        "trading days"
    )

    print("\nMatched regimes:")
    print(
        matches[
            [
                "high_date",
                "high_risk_index",
                "low_date",
                "low_risk_index",
                "trading_day_distance",
            ]
        ].to_string(index=False)
    )

    print(f"\nEstimation sample saved to: {SAMPLE_PATH}")
    print(f"Matched dates saved to: {MATCHES_PATH}")


if __name__ == "__main__":
    main()