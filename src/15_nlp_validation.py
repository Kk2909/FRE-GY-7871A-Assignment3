from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/iran_news_scored.csv"
)

OUTPUT_PATH = Path(
    "outputs/tables/nlp_validation_sample.csv"
)

SAMPLE_START = pd.Timestamp("2026-02-28")
RANDOM_SEED = 42


def main():
    news = pd.read_csv(INPUT_PATH)

    news["published_utc"] = pd.to_datetime(
        news["published_utc"],
        errors="coerce",
        utc=True,
    )

    news = news[
        news["published_utc"].dt.tz_localize(None)
        >= SAMPLE_START
    ].copy()

    news = news.dropna(
        subset=[
            "title_clean",
            "war_news_intensity",
            "war_risk_direction",
        ]
    ).copy()

    news = news.drop_duplicates(
        subset=["title_clean", "source"]
    ).reset_index(drop=True)

    # Group 1: the 20 highest-intensity articles.
    highest_intensity = (
        news.nlargest(20, "war_news_intensity")
        .copy()
    )

    highest_intensity["sample_type"] = (
        "Highest intensity"
    )

    remaining = news.drop(
        index=highest_intensity.index
    ).copy()

    # Group 2: 15 articles closest to the 75th-percentile
    # intensity threshold.
    intensity_threshold = news[
        "war_news_intensity"
    ].quantile(0.75)

    remaining["distance_from_threshold"] = (
        remaining["war_news_intensity"]
        - intensity_threshold
    ).abs()

    near_threshold = (
        remaining.nsmallest(
            15,
            "distance_from_threshold",
        )
        .copy()
    )

    near_threshold["sample_type"] = (
        "Near intensity threshold"
    )

    remaining = remaining.drop(
        index=near_threshold.index
    ).copy()

    # Group 3: 15 randomly selected articles.
    random_sample = remaining.sample(
        n=min(15, len(remaining)),
        random_state=RANDOM_SEED,
    ).copy()

    random_sample["sample_type"] = (
        "Random sample"
    )

    validation = pd.concat(
        [
            highest_intensity,
            near_threshold,
            random_sample,
        ],
        ignore_index=True,
    )

    validation["predicted_signal"] = validation[
        "dominant_signal"
    ]

    validation["manual_label"] = ""
    validation["classification_correct"] = ""
    validation["novel_phrasing"] = ""
    validation["notes"] = ""

    output_columns = [
        "sample_type",
        "published_utc",
        "title_clean",
        "source",
        "predicted_signal",
        "war_news_intensity",
        "war_risk_direction",
        "relevance_score",
        "manual_label",
        "classification_correct",
        "novel_phrasing",
        "notes",
    ]

    validation = validation[output_columns]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Validation articles selected: "
        f"{len(validation):,}"
    )

    print("\nSample composition:")
    print(
        validation["sample_type"]
        .value_counts()
        .to_string()
    )

    print(
        "\nManual review columns are blank and must be "
        "completed before reporting validation accuracy."
    )

    print(
        "\nUse these manual labels: "
        "escalation, de-escalation, neutral, or unclear."
    )

    print(f"\nValidation sample saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()