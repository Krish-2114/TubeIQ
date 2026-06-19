import pandas as pd
import numpy as np
import joblib
import os
import json
import re
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import (
    TfidfVectorizer
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split
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


def train_xgboost_for_niche(df, niche):
    print(f"\n  Training XGBoost for {niche}...")

    df = df.dropna(subset=FEATURE_COLUMNS +
                   ['performance_label'])

    if len(df) < 50:
        print(f"  Skipping — only {len(df)} rows")
        return None, None

    X = df[FEATURE_COLUMNS].fillna(0)
    y_raw = df['performance_label']

    le = LabelEncoder()
    le.fit(['Low', 'Medium', 'High', 'Viral'])
    y = le.transform(y_raw)

    # Time-based split — last 20% as test
    split_idx = int(len(df) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y[:split_idx]
    y_test  = y[split_idx:]

    if len(X_test) < 10:
        X_train, X_test, y_train, y_test = \
            train_test_split(
                X, y, test_size=0.2,
                random_state=42,
                stratify=y
            )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(
        y_test, preds, average='weighted'
    )

    print(f"  Accuracy: {accuracy:.3f} | "
          f"F1: {f1:.3f}")
    print(f"  Test size: {len(X_test)} videos")

    # Feature importance
    importance = dict(zip(
        FEATURE_COLUMNS,
        model.feature_importances_
    ))
    top_features = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    print(f"  Top features: "
          f"{[f[0] for f in top_features]}")

    # Save model and encoder
    joblib.dump(
        model,
        f"backend/saved_models/"
        f"xgboost_{niche}.pkl"
    )
    joblib.dump(
        le,
        f"backend/saved_models/"
        f"label_encoder_{niche}.pkl"
    )

    return model, {
        "accuracy": round(accuracy, 3),
        "f1": round(f1, 3),
        "test_size": len(X_test),
        "train_size": len(X_train),
        "top_features": [f[0] for f in top_features]
    }


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
        print(f"    Cluster {cluster_id}: "
              f"{', '.join(keywords)}")

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

    all_results = {}

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

        # Train XGBoost
        model, xgb_results = \
            train_xgboost_for_niche(df, niche)

        # Train TF-IDF
        tfidf = train_tfidf_for_niche(df, niche)

        # Train K-Means
        if tfidf:
            kmeans, clusters = \
                train_kmeans_for_niche(df, niche)

        if xgb_results:
            all_results[niche] = xgb_results

    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"\nXGBoost Results Summary:")
    for niche, results in all_results.items():
        print(f"  {niche:<15} "
              f"Accuracy: {results['accuracy']:.3f} "
              f"F1: {results['f1']:.3f}")

    # Save training results
    with open(
        "backend/saved_models/training_results.json",
        'w'
    ) as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll models saved to "
          f"backend/saved_models/")

    # Count saved files
    model_files = os.listdir("backend/saved_models")
    print(f"Total model files: {len(model_files)}")


if __name__ == "__main__":
    train_all_models()
