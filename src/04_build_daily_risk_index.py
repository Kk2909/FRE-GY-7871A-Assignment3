from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


INPUT_PATH = Path("data/processed/iran_news_scored.csv")
OUTPUT_PATH = Path("data/processed/daily_iran_war_risk.csv")
TOP_DAYS_PATH = Path("outputs/tables/high_war_news_days.csv")


def assign_market_date(timestamp: pd.Timestamp):
    if pd.isna(timestamp):
        return pd.NaT

    local_date = timestamp.tz_localize(None).normalize()

    # News released after the U.S. market close is assigned
    # to the next business day.
    if timestamp.weekday() >= 5 or timestamp.hour >= 16:
        local_date = local_date + BDay(1)

    return local_date


def z_score(series: pd.Series) -> pd.Series:
    standard_deviation = series.std(ddof=0)

    if standard_deviation == 0:
        return pd.Series(0.0, index=series.index)

    return (series - series.mean()) / standard_deviation


def summarize_day(group: pd.DataFrame) -> pd.Series:
    strongest = group.nlargest(
        min(10, len(group)),
        "war_news_intensity",
    )

    total_words = group["word_count"].sum()

    uncertainty_rate = (
        100 * group["uncertainty_terms"].sum() / total_words
        if total_words > 0
        else 0
    )

    weighted_direction = np.average(
        group["war_risk_direction"],
        weights=group["war_news_intensity"] + 1e-8,
    )

    representative_article = group.loc[
        group["war_news_intensity"].idxmax()
    ]

    return pd.Series(
        {
            "article_count": len(group),
            "mean_relevance": group["relevance_score"].mean(),
            "mean_intensity": group["war_news_intensity"].mean(),
            "top10_intensity": strongest["war_news_intensity"].mean(),
            "uncertainty_rate": uncertainty_rate,
            "escalation_share": (
                group["dominant_signal"]
                .eq("escalation")
                .mean()
            ),
            "weighted_direction": weighted_direction,
            "top_article": representative_article["title_clean"],
            "top_source": representative_article["source"],
        }
    )


def main():
    news = pd.read_csv(INPUT_PATH)

    published_utc = pd.to_datetime(
        news["published_utc"],
        errors="coerce",
        utc=True,
    )

    news["published_et"] = (
        published_utc.dt.tz_convert("America/New_York")
    )

    news["market_date"] = news["published_et"].apply(
        assign_market_date
    )

    news = news.dropna(subset=["market_date"]).copy()

    daily = (
        news.groupby("market_date")
        .apply(summarize_day, include_groups=False)
        .reset_index()
        .sort_values("market_date")
        .reset_index(drop=True)
    )

    daily["intensity_z"] = z_score(
        daily["top10_intensity"]
    )

    daily["uncertainty_z"] = z_score(
        daily["uncertainty_rate"]
    )

    # Semantic event intensity receives the larger weight.
    # Uncertainty language provides a secondary confirmation.
    daily["war_risk_index"] = (
        0.80 * daily["intensity_z"]
        + 0.20 * daily["uncertainty_z"]
    )

    high_threshold = daily["war_risk_index"].quantile(0.90)
    low_threshold = daily["war_risk_index"].quantile(0.50)

    daily["high_news_day"] = (
        daily["war_risk_index"] >= high_threshold
    )

    daily["low_news_candidate"] = (
        daily["war_risk_index"] <= low_threshold
    )

    daily["risk_rank"] = (
        daily["war_risk_index"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOP_DAYS_PATH.parent.mkdir(parents=True, exist_ok=True)

    daily.to_csv(OUTPUT_PATH, index=False)

    high_days = (
        daily[daily["high_news_day"]]
        .sort_values("war_risk_index", ascending=False)
    )

    high_days.to_csv(TOP_DAYS_PATH, index=False)

    print(f"Trading/news dates: {len(daily):,}")
    print(f"High-news threshold: {high_threshold:.4f}")
    print(f"High-news days: {daily['high_news_day'].sum():,}")
    print(
        f"Low-news candidates: "
        f"{daily['low_news_candidate'].sum():,}"
    )

    print("\nTop 20 Iran War Risk days:")

    display_columns = [
        "risk_rank",
        "market_date",
        "war_risk_index",
        "article_count",
        "top_article",
        "top_source",
    ]

    print(
        high_days[display_columns]
        .head(20)
        .to_string(index=False)
    )

    print(f"\nDaily index saved to: {OUTPUT_PATH}")
    print(f"High-news table saved to: {TOP_DAYS_PATH}")


if __name__ == "__main__":
    main()