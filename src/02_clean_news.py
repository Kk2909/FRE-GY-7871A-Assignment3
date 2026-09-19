from html import unescape
from pathlib import Path
import re

import pandas as pd
from bs4 import BeautifulSoup


INPUT_PATH = Path("data/raw/iran_news_2026.csv")
OUTPUT_PATH = Path("data/processed/iran_news_clean.csv")


def clean_html(value: str) -> str:
    if pd.isna(value):
        return ""

    text = BeautifulSoup(str(value), "html.parser").get_text(" ")
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_source_from_title(title: str, source: str) -> str:
    title = clean_html(title)
    source = clean_html(source)

    suffix = f" - {source}"

    if source and title.endswith(suffix):
        title = title[: -len(suffix)]

    return title.strip()


def main():
    news = pd.read_csv(INPUT_PATH)

    print(f"Raw articles: {len(news):,}")

    news["published_utc"] = pd.to_datetime(
        news["published"],
        errors="coerce",
        utc=True,
    )

    news["published_et"] = (
        news["published_utc"]
        .dt.tz_convert("America/New_York")
    )

    news["news_date"] = news["published_et"].dt.date

    news["source"] = news["source"].fillna("").apply(clean_html)
    news["title_clean"] = news.apply(
        lambda row: remove_source_from_title(
            row["title"],
            row["source"],
        ),
        axis=1,
    )

    news["summary_clean"] = (
        news["summary"]
        .fillna("")
        .apply(clean_html)
    )

    news["text"] = (
        news["title_clean"]
        + ". "
        + news["summary_clean"]
    ).str.strip()

    news["word_count"] = (
        news["text"]
        .str.split()
        .str.len()
    )

    news = news.dropna(
        subset=["published_utc"]
    ).copy()

    news = news[
        news["title_clean"].str.len() > 0
    ].copy()

    before_duplicates = len(news)

    news["title_key"] = (
        news["title_clean"]
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    news = news.drop_duplicates(
        subset=["title_key"],
        keep="first",
    ).copy()

    duplicates_removed = before_duplicates - len(news)

    news = news.sort_values(
        "published_utc"
    ).reset_index(drop=True)

    columns = [
        "published_utc",
        "published_et",
        "news_date",
        "source",
        "title_clean",
        "summary_clean",
        "text",
        "word_count",
        "url",
        "query_category",
    ]

    news = news[columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(OUTPUT_PATH, index=False)

    print(f"Clean articles: {len(news):,}")
    print(f"Additional duplicates removed: {duplicates_removed:,}")
    print(f"Missing summaries: {(news['summary_clean'] == '').sum():,}")
    print(f"Median word count: {news['word_count'].median():.0f}")
    print(f"First date: {news['news_date'].min()}")
    print(f"Last date: {news['news_date'].max()}")

    print("\nArticles by query category:")
    print(news["query_category"].value_counts())

    print("\nTop 10 news sources:")
    print(news["source"].value_counts().head(10))

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()