from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


INPUT_PATH = Path("data/processed/iran_news_clean.csv")
OUTPUT_PATH = Path("data/processed/iran_news_scored.csv")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RELEVANCE_PROMPTS = [
    "Military conflict, war, attacks, or security threats involving Iran.",
    "Iranian nuclear tensions, sanctions, and geopolitical confrontation.",
    "International security risks involving Iran, Israel, or the United States.",
]

ESCALATION_PROMPTS = [
    "War risk is increasing because of military attacks or missile strikes.",
    "Iran threatens retaliation and the conflict may expand.",
    "Nuclear tensions, military threats, and geopolitical confrontation are escalating.",
]

DEESCALATION_PROMPTS = [
    "War risk is decreasing because of a ceasefire or peace agreement.",
    "Diplomatic negotiations are reducing tensions with Iran.",
    "The countries are seeking compromise and avoiding military conflict.",
]

ESCALATION_TERMS = [
    "attack",
    "airstrike",
    "bomb",
    "conflict",
    "drone",
    "escalation",
    "invasion",
    "military action",
    "missile",
    "retaliation",
    "strike",
    "threat",
    "troops",
    "war",
]

DEESCALATION_TERMS = [
    "ceasefire",
    "compromise",
    "de-escalation",
    "diplomacy",
    "diplomatic",
    "negotiation",
    "peace",
    "talks",
    "truce",
]

UNCERTAINTY_TERMS = [
    "could",
    "may",
    "might",
    "possible",
    "potential",
    "risk",
    "uncertain",
    "uncertainty",
    "unlikely",
]


def count_terms(text: str, terms: list[str]) -> int:
    text = str(text).lower()
    total = 0

    for term in terms:
        pattern = rf"\b{re.escape(term)}\b"
        total += len(re.findall(pattern, text))

    return total


def maximum_similarity(
    article_embeddings: np.ndarray,
    prompt_embeddings: np.ndarray,
) -> np.ndarray:
    similarities = article_embeddings @ prompt_embeddings.T
    return similarities.max(axis=1)


def main():
    news = pd.read_csv(INPUT_PATH)

    texts = (
        news["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Articles to score: {len(news):,}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    article_embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    relevance_embeddings = model.encode(
        RELEVANCE_PROMPTS,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    escalation_embeddings = model.encode(
        ESCALATION_PROMPTS,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    deescalation_embeddings = model.encode(
        DEESCALATION_PROMPTS,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    news["relevance_score"] = maximum_similarity(
        article_embeddings,
        relevance_embeddings,
    )

    news["escalation_score"] = maximum_similarity(
        article_embeddings,
        escalation_embeddings,
    )

    news["deescalation_score"] = maximum_similarity(
        article_embeddings,
        deescalation_embeddings,
    )

    news["escalation_terms"] = news["text"].apply(
        lambda text: count_terms(text, ESCALATION_TERMS)
    )

    news["deescalation_terms"] = news["text"].apply(
        lambda text: count_terms(text, DEESCALATION_TERMS)
    )

    news["uncertainty_terms"] = news["text"].apply(
        lambda text: count_terms(text, UNCERTAINTY_TERMS)
    )

    news["event_strength"] = news[
        ["escalation_score", "deescalation_score"]
    ].max(axis=1)

    news["war_news_intensity"] = (
        news["relevance_score"]
        * news["event_strength"]
    )

    news["war_risk_direction"] = (
        news["relevance_score"]
        * (
            news["escalation_score"]
            - news["deescalation_score"]
        )
    )

    news["dominant_signal"] = np.where(
        news["escalation_score"]
        >= news["deescalation_score"],
        "escalation",
        "deescalation",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(OUTPUT_PATH, index=False)

    score_columns = [
        "relevance_score",
        "escalation_score",
        "deescalation_score",
        "war_news_intensity",
        "war_risk_direction",
    ]

    print("\nScore summary:")
    print(
        news[score_columns]
        .describe()
        .round(4)
        .to_string()
    )

    print("\nDominant signal:")
    print(news["dominant_signal"].value_counts())

    print("\nTop 10 articles by war-news intensity:")

    top_articles = news.nlargest(
        10,
        "war_news_intensity",
    )[
        [
            "news_date",
            "source",
            "title_clean",
            "war_news_intensity",
            "dominant_signal",
        ]
    ]

    print(top_articles.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()