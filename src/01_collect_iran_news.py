from datetime import timedelta
from pathlib import Path
from urllib.parse import quote_plus
import time

import feedparser
import pandas as pd
import requests


START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-09-19")

OUTPUT_PATH = Path("data/raw/iran_news_2026.csv")

SEARCH_QUERIES = {
    "military": (
        'Iran (war OR attack OR strike OR missile OR military OR conflict)'
    ),
    "nuclear": (
        'Iran (nuclear OR enrichment OR IAEA OR sanctions)'
    ),
    "diplomatic": (
        'Iran (ceasefire OR negotiations OR diplomacy OR talks)'
    ),
}


def fetch_rss(query: str) -> list[dict]:
    encoded_query = quote_plus(query)

    url = (
        "https://news.google.com/rss/search"
        f"?q={encoded_query}"
        "&hl=en-US"
        "&gl=US"
        "&ceid=US:en"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
        )
    }

    for attempt in range(1, 4):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()

            feed = feedparser.parse(response.content)

            if feed.bozo and not feed.entries:
                raise ValueError("RSS feed could not be parsed.")

            articles = []

            for entry in feed.entries:
                source = ""

                if hasattr(entry, "source"):
                    source = entry.source.get("title", "")

                articles.append(
                    {
                        "title": entry.get("title", ""),
                        "url": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": source,
                        "summary": entry.get("summary", ""),
                    }
                )

            return articles

        except (requests.RequestException, ValueError) as error:
            wait_seconds = 10 * attempt
            print(
                f"Attempt {attempt} failed: {error}. "
                f"Waiting {wait_seconds} seconds."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Unable to collect query: {query}")


def main():
    all_articles = []
    window_start = START_DATE
    window_number = 0

    while window_start < END_DATE:
        window_end = min(
            window_start + timedelta(days=7),
            END_DATE,
        )
        window_number += 1

        print(
            f"\nWindow {window_number:02d}: "
            f"{window_start.date()} to "
            f"{(window_end - timedelta(days=1)).date()}"
        )

        for category, base_query in SEARCH_QUERIES.items():
            dated_query = (
                f"{base_query} "
                f"after:{window_start.date()} "
                f"before:{window_end.date()}"
            )

            articles = fetch_rss(dated_query)

            for article in articles:
                article["query_category"] = category
                article["window_start"] = window_start.date()
                article["window_end"] = window_end.date()

            all_articles.extend(articles)

            print(f"  {category}: {len(articles):,} articles")
            time.sleep(2)

        window_start = window_end

    news = pd.DataFrame(all_articles)

    if news.empty:
        print("No articles were collected.")
        return

    original_count = len(news)

    news["published"] = pd.to_datetime(
        news["published"],
        errors="coerce",
        utc=True,
    )

    news = news.dropna(subset=["published"]).copy()

    news = news[
        (news["published"] >= START_DATE.tz_localize("UTC"))
        & (news["published"] < END_DATE.tz_localize("UTC"))
    ].copy()

    news["clean_title"] = (
        news["title"]
        .str.replace(r"\s+-\s+[^-]+$", "", regex=True)
        .str.strip()
        .str.lower()
    )

    news = news.drop_duplicates(
        subset=["clean_title"],
        keep="first",
    )

    news = news.sort_values("published").reset_index(drop=True)
    news = news.drop(columns=["clean_title"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(OUTPUT_PATH, index=False)

    print("\nCollection complete")
    print(f"Records before cleaning: {original_count:,}")
    print(f"Unique dated articles: {len(news):,}")
    print(f"Duplicates or invalid records removed: {original_count - len(news):,}")
    print(f"Date range: {news['published'].min()} to {news['published'].max()}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()