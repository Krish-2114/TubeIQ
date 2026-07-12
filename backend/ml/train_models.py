import pandas as pd
import numpy as np
import joblib
import os
import json
import re
from sklearn.feature_extraction.text import (
    TfidfVectorizer
)
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

NICHES = [
    "gaming", "entertainment", "cooking",
    "finance", "fitness", "education", "tech",
    "streaming"
]

FEATURE_COLUMNS = [
    "duration_seconds", "duration_encoded",
    "day_of_week", "hour_of_day",
    "title_length", "word_count",
    "has_number", "has_dollar", "has_rupee",
    "has_question", "has_exclamation",
    "has_vs", "has_challenge", "has_hours",
    "has_last", "has_win", "has_free",
    "has_extreme", "has_best", "has_worst",
    "has_every", "has_first", "has_only",
    "has_never", "has_days", "has_review",
    "has_how", "has_why", "has_what"
]

os.makedirs("backend/saved_models", exist_ok=True)


def clean_title(title, channel_title):
    if not title:
        return ''

    channel_lower = str(channel_title or '').lower()
    channel_words = set(re.findall(r'\b\w+\b', channel_lower))
    channel_compact = re.sub(r'[^\w]', '', channel_lower)

    tokens = str(title).split()
    cleaned = []

    for token in tokens:
        bare = token.strip('.,!?;:"\'()[]')

        if not bare:
            continue

        token_lower = bare.lower()

        # Hashtags like #shorts
        if token_lower.startswith('#'):
            continue

        # @mentions
        if bare.startswith('@'):
            continue

        word_clean = re.sub(r'[^\w]', '', token_lower)

        if not word_clean:
            continue

        # Direct match with channel name or its parts
        if word_clean in channel_words:
            continue

        # CamelCase handles matching channel name pattern
        is_camel_handle = bool(
            re.search(r'[a-z][A-Z]', bare) or
            re.search(r'^[A-Z][a-z]+[A-Z]', bare)
        )
        if is_camel_handle and channel_compact:
            if (
                word_clean == channel_compact or
                (len(channel_compact) >= 4 and
                 channel_compact in word_clean) or
                (len(word_clean) >= 4 and
                 word_clean in channel_compact)
            ):
                continue

        cleaned.append(bare)

    return ' '.join(cleaned)


def train_tfidf_for_niche(df, niche):
    print(f"\n  Training TF-IDF for {niche}...")

    df = df.copy()
    df['clean_title'] = df.apply(
        lambda row: clean_title(
            row.get('title', ''),
            row.get('channel_title', '')
        ),
        axis=1
    )
    titles = df['clean_title'].fillna('').tolist()

    if len(titles) < 10:
        print(f"  Skipping — insufficient titles")
        return None

    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2
    )

    tfidf.fit(titles)

    joblib.dump(
        tfidf,
        f"backend/saved_models/"
        f"tfidf_{niche}.pkl"
    )

    # Proven-winners reference set for title comparison
    high_viral_mask = df['performance_label'].isin(
        ['High', 'Viral']
    )
    reference_titles = df.loc[
        high_viral_mask, 'title'
    ].fillna('').tolist()
    if reference_titles:
        reference_vecs = tfidf.transform(reference_titles)
        joblib.dump(
            {
                'titles': reference_titles,
                'vectors': reference_vecs,
            },
            f"backend/saved_models/"
            f"reference_set_{niche}.pkl"
        )
        print(
            f"  Reference set: "
            f"{len(reference_titles)} High/Viral titles"
        )
    else:
        print("  Reference set: skipped — no High/Viral titles")

    print(f"  Vocabulary size: "
          f"{len(tfidf.vocabulary_)}")

    return tfidf


def train_kmeans_for_niche(df, niche, n_clusters=6):
    print(f"\n  Training K-Means for {niche}...")

    df = df.copy()
    df['clean_title'] = df.apply(
        lambda row: clean_title(
            row.get('title', ''),
            row.get('channel_title', '')
        ),
        axis=1
    )
    titles = df['clean_title'].fillna('').tolist()

    if len(titles) < n_clusters * 3:
        print(f"  Skipping — insufficient data")
        return None

    tfidf = joblib.load(
        f"backend/saved_models/"
        f"tfidf_{niche}.pkl"
    )

    X = tfidf.transform(titles)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)

    # Get top keywords per cluster
    feature_names = tfidf.get_feature_names_out()
    cluster_keywords = {}

    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-5:][::-1]
        keywords = [
            feature_names[idx]
            for idx in top_indices
        ]
        cluster_keywords[i] = keywords

    print(f"  Clusters:")
    for cluster_id, keywords in \
            cluster_keywords.items():
        joined = ", ".join(keywords)
        # Windows console encoding can throw UnicodeEncodeError (e.g. emojis).
        safe_joined = joined.encode("cp1252", errors="replace").decode(
            "cp1252"
        )
        print(
            f"    Cluster {cluster_id}: "
            f"{safe_joined}"
        )

    joblib.dump(
        kmeans,
        f"backend/saved_models/"
        f"kmeans_{niche}.pkl"
    )

    # Save cluster keywords
    with open(
        f"backend/saved_models/"
        f"clusters_{niche}.json", 'w'
    ) as f:
        json.dump(cluster_keywords, f, indent=2)

    return kmeans, cluster_keywords


def train_all_models():
    print("="*50)
    print("TRAINING ALL TUBEIQ MODELS")
    print("="*50)

    trained = []

    for niche in NICHES:
        path = f"backend/data/{niche}_features.csv"

        if not os.path.exists(path):
            print(f"\nSkipping {niche} — "
                  f"CSV not found")
            continue

        print(f"\n{'='*40}")
        print(f"NICHE: {niche.upper()}")
        print(f"{'='*40}")

        df = pd.read_csv(path)
        print(f"Loaded {len(df)} videos")

        # Train TF-IDF (+ reference set)
        tfidf = train_tfidf_for_niche(df, niche)

        # Train K-Means
        if tfidf:
            train_kmeans_for_niche(df, niche)
            trained.append(niche)

    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"\nTrained niches: {', '.join(trained) or 'none'}")

    print(f"\nAll models saved to "
          f"backend/saved_models/")

    # Count saved files
    model_files = os.listdir("backend/saved_models")
    print(f"Total model files: {len(model_files)}")


if __name__ == "__main__":
    train_all_models()
