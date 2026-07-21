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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from itertools import combinations
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

BLEND_FEATURE_NAMES = [
    "tfidf_diff",
    "has_number_diff",
    "has_question_diff",
    "title_length_diff",
]

MIN_VIEW_RATIO = 2.0
MAX_AGE_GAP_DAYS = 120
MIN_BLEND_PAIRS = 200

# LOCO blend−duration accuracies (age-gap fixed) for sanity flags
LOCO_BLEND_ACC = {
    "gaming": 0.534,
    "entertainment": 0.542,
    "cooking": 0.609,
    "finance": 0.634,
    "fitness": 0.578,
    "education": 0.547,
    "tech": 0.586,
    "streaming": 0.604,
}

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
    reference_rows = df.loc[
        high_viral_mask, ['title', 'video_id']
    ].fillna('')
    reference_titles = reference_rows['title'].tolist()
    reference_video_ids = reference_rows['video_id'].tolist()
    if reference_titles:
        reference_vecs = tfidf.transform(reference_titles)
        joblib.dump(
            {
                'titles': reference_titles,
                'video_ids': reference_video_ids,
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


def _inline_title_feats(title: str) -> dict:
    """Lightweight title features — no DB import from feature_engineering."""
    title = title or ""
    return {
        "title_length": len(title),
        "has_number": int(bool(re.search(r"\d", title))),
        "has_question": int("?" in title),
    }


def _avg_sim_excluding(
    per_ref_sims: np.ndarray,
    ref_video_ids: list,
    exclude_video_ids: set,
) -> float:
    keep = [
        i
        for i, vid in enumerate(ref_video_ids)
        if vid not in exclude_video_ids
    ]
    if not keep:
        return 0.0
    return float(np.mean(per_ref_sims[keep]))


def train_blend_for_niche(df, niche, tfidf=None):
    """
    Fit production blend model on ALL channels in the niche.
    Features: leakage-safe tfidf_diff + structural diffs (no duration).
    """
    print(f"\n  Training blend model for {niche}...")

    needed = [
        "title", "view_velocity", "channel_id",
        "age_days", "performance_label", "video_id",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"  Skipping blend — missing columns: {missing}")
        return None

    work = df.dropna(subset=needed).copy()
    work = work[work["view_velocity"] > 0]
    work["title"] = work["title"].fillna("").astype(str)

    if tfidf is None:
        path = f"backend/saved_models/tfidf_{niche}.pkl"
        if not os.path.exists(path):
            print("  Skipping blend — TF-IDF model not found")
            return None
        tfidf = joblib.load(path)

    ref_path = f"backend/saved_models/reference_set_{niche}.pkl"
    if not os.path.exists(ref_path):
        print("  Skipping blend — reference set not found")
        return None
    ref = joblib.load(ref_path)
    ref_titles = ref.get("titles") or []
    ref_vectors = ref.get("vectors")
    ref_video_ids = ref.get("video_ids") or []
    if not ref_titles or ref_vectors is None or ref_vectors.shape[0] == 0:
        print("  Skipping blend — empty reference set")
        return None
    if len(ref_video_ids) != len(ref_titles):
        print(
            f"  ERROR: reference set video_ids length "
            f"({len(ref_video_ids)}) != titles length "
            f"({len(ref_titles)}) — re-run TF-IDF training "
            f"to regenerate reference_set_{niche}.pkl"
        )
        return None

    pairs = []
    for _, channel_df in work.groupby("channel_id"):
        records = channel_df.to_dict("records")
        for row_a, row_b in combinations(records, 2):
            age_gap = abs(
                float(row_a["age_days"]) - float(row_b["age_days"])
            )
            if age_gap > MAX_AGE_GAP_DAYS:
                continue
            vel_a = float(row_a["view_velocity"])
            vel_b = float(row_b["view_velocity"])
            if vel_a == vel_b:
                continue
            lo, hi = min(vel_a, vel_b), max(vel_a, vel_b)
            if lo <= 0 or hi / lo < MIN_VIEW_RATIO:
                continue
            pairs.append((row_a, row_b))

    if len(pairs) < MIN_BLEND_PAIRS:
        print(
            f"  Skipping blend — only {len(pairs)} valid pairs "
            f"(need >= {MIN_BLEND_PAIRS})"
        )
        return None

    print(f"  Blend pairs: {len(pairs):,}")

    all_titles = work["title"].tolist()
    all_video_ids = work["video_id"].tolist()
    video_vectors = tfidf.transform(all_titles)
    sim_matrix = cosine_similarity(video_vectors, ref_vectors)
    video_ref_sims = {
        all_video_ids[i]: sim_matrix[i]
        for i in range(len(all_video_ids))
    }

    X = []
    y = []
    for row_a, row_b in pairs:
        exclude = {row_a["video_id"], row_b["video_id"]}
        sim_a = _avg_sim_excluding(
            video_ref_sims[row_a["video_id"]],
            ref_video_ids,
            exclude,
        )
        sim_b = _avg_sim_excluding(
            video_ref_sims[row_b["video_id"]],
            ref_video_ids,
            exclude,
        )
        feats_a = _inline_title_feats(row_a["title"])
        feats_b = _inline_title_feats(row_b["title"])
        X.append([
            sim_b - sim_a,
            feats_b["has_number"] - feats_a["has_number"],
            feats_b["has_question"] - feats_a["has_question"],
            feats_b["title_length"] - feats_a["title_length"],
        ])
        y.append(
            1 if float(row_b["view_velocity"]) > float(row_a["view_velocity"])
            else 0
        )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    clf.fit(X_scaled, y)

    scores = clf.decision_function(X_scaled)
    preds = (scores > 0).astype(int)
    ties = scores == 0
    train_correct = int(np.sum((preds == y) & ~ties))
    train_acc = train_correct / len(y)

    coefs = {
        name: float(coef)
        for name, coef in zip(BLEND_FEATURE_NAMES, clf.coef_[0])
    }

    joblib.dump(
        {"scaler": scaler, "model": clf},
        f"backend/saved_models/blend_{niche}.pkl",
    )

    print(f"  Training accuracy: {train_acc * 100:.1f}%")
    print(f"  Coefficients:")
    for name, coef in coefs.items():
        print(f"    {name}: {coef:+.4f}")

    loco = LOCO_BLEND_ACC.get(niche)
    if loco is not None:
        delta_pp = (train_acc - loco) * 100
        if train_acc >= 0.95:
            print(
                f"  FLAG: training accuracy {train_acc * 100:.1f}% is "
                f"near 100% (possible overfitting vs LOCO "
                f"{loco * 100:.1f}%)"
            )
        elif train_acc <= 0.52:
            print(
                f"  FLAG: training accuracy {train_acc * 100:.1f}% is "
                f"near chance — something may be broken "
                f"(LOCO was {loco * 100:.1f}%)"
            )
        elif abs(delta_pp) > 20:
            print(
                f"  FLAG: training accuracy differs from LOCO by "
                f"{delta_pp:+.1f}pp "
                f"(train {train_acc * 100:.1f}% vs LOCO {loco * 100:.1f}%)"
            )
        else:
            print(
                f"  Sanity vs LOCO: train {train_acc * 100:.1f}% vs "
                f"LOCO {loco * 100:.1f}% ({delta_pp:+.1f}pp)"
            )

    return {
        "niche": niche,
        "num_pairs": len(pairs),
        "coefficients": coefs,
        "train_accuracy": round(train_acc, 4),
        "loco_accuracy": loco,
    }


def train_all_models():
    print("="*50)
    print("TRAINING ALL TUBEIQ MODELS")
    print("="*50)

    trained = []
    blend_results = []

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

            # Production blend on all channels
            blend = train_blend_for_niche(df, niche, tfidf=tfidf)
            if blend:
                blend_results.append(blend)

    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"\nTrained niches: {', '.join(trained) or 'none'}")

    if blend_results:
        print(f"\n{'='*72}")
        print("BLEND MODEL SUMMARY")
        print(f"{'='*72}")
        print(
            f"{'Niche':<15} {'Pairs':>8} {'Train%':>8} "
            f"{'tfidf':>9} {'number':>9} {'question':>9} {'length':>9}"
        )
        print(
            f"{'-' * 15} {'-' * 8} {'-' * 8} "
            f"{'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9}"
        )
        for row in blend_results:
            c = row["coefficients"]
            print(
                f"{row['niche']:<15} {row['num_pairs']:>8,} "
                f"{row['train_accuracy'] * 100:>7.1f}% "
                f"{c['tfidf_diff']:>+9.4f} "
                f"{c['has_number_diff']:>+9.4f} "
                f"{c['has_question_diff']:>+9.4f} "
                f"{c['title_length_diff']:>+9.4f}"
            )

    print(f"\nAll models saved to "
          f"backend/saved_models/")

    model_files = os.listdir("backend/saved_models")
    print(f"Total model files: {len(model_files)}")
    blend_files = sorted(
        f for f in model_files
        if f.startswith("blend_") and f.endswith(".pkl")
    )
    print(f"Blend models: {len(blend_files)}")
    for f in blend_files:
        print(f"  {f}")


if __name__ == "__main__":
    train_all_models()
