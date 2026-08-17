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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
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

# Production blend features — selected by select_rf_features.py (top-25 by
# mean RF importance). age_days excluded (label leak).
# Order is the model input order; do not reshuffle without retraining.
BLEND_FEATURE_NAMES = [
    "duration_seconds_diff",
    "duration_encoded_diff",
    "tfidf_diff",
    "uppercase_ratio_diff",
    "uppercase_count_diff",
    "hour_of_day_diff",
    "all_caps_word_count_diff",
    "title_length_diff",
    "digit_count_diff",
    "char_density_diff",
    "space_count_diff",
    "word_count_diff",
    "word_length_mean_diff",
    "unique_word_ratio_diff",
    "punctuation_count_diff",
    "day_of_week_diff",
    "exclamation_count_diff",
    "has_exclamation_diff",
    "word_length_max_diff",
    "has_year_diff",
    "emoji_count_diff",
    "hashtag_count_diff",
    "long_word_count_diff",
    "has_number_diff",
    "has_quotes_diff",
]

# Side columns (everything except tfidf) used to build pair diffs
BLEND_SIDE_COLS = [
    name[: -len("_diff")]
    for name in BLEND_FEATURE_NAMES
    if name != "tfidf_diff"
]

# Tuned RF; held-out numbers refreshed after top-25 selection
BLEND_RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 14,
    "min_samples_leaf": 4,
    "max_features": "log2",
    "random_state": 42,
    "n_jobs": -1,
}

MIN_VIEW_RATIO = 2.0
MAX_AGE_GAP_DAYS = 120
MIN_BLEND_PAIRS = 200
BLEND_PAIR_SEED = 42

# Held-out accuracy from blend_eval.py (random 80/20 over pairs).
# Sanity flags only — production trains on every channel.
EVAL_BLEND_ACC = {
    "gaming": 0.822,
    "entertainment": 0.757,
    "cooking": 0.825,
    "finance": 0.745,
    "fitness": 0.785,
    "education": 0.743,
    "tech": 0.725,
    "streaming": 0.864,
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
        high_viral_mask,
        [
            'title',
            'video_id',
            'channel_title',
            'view_count',
            'performance_label',
        ],
    ].copy()
    reference_rows['title'] = reference_rows['title'].fillna('')
    reference_rows['video_id'] = reference_rows['video_id'].fillna('')
    reference_rows['channel_title'] = (
        reference_rows['channel_title'].fillna('')
    )
    reference_titles = reference_rows['title'].tolist()
    reference_video_ids = reference_rows['video_id'].tolist()
    if reference_titles:
        reference_vecs = tfidf.transform(reference_titles)
        joblib.dump(
            {
                'titles': reference_titles,
                'video_ids': reference_video_ids,
                'vectors': reference_vecs,
                'channel_titles': (
                    reference_rows['channel_title'].tolist()
                ),
                'view_counts': (
                    reference_rows['view_count'].tolist()
                ),
                'performance_labels': (
                    reference_rows['performance_label'].tolist()
                ),
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


def compute_and_save_gaps(df, niche, tfidf, kmeans, cluster_keywords):
    """Precompute content-gap rankings and write gaps_{niche}.csv."""
    print(f"\n  Computing gaps for {niche}...")

    work = df.copy()
    cleaned_titles = work.apply(
        lambda row: clean_title(
            row.get("title", ""),
            row.get("channel_title", ""),
        ),
        axis=1,
    )
    vectors = tfidf.transform(cleaned_titles.fillna("").tolist())
    cluster_labels = kmeans.predict(vectors)
    work["cluster_id"] = cluster_labels

    cluster_avgs = (
        work.groupby("cluster_id")["view_count"]
        .mean()
        .to_dict()
    )
    overall_avg = float(np.mean(list(cluster_avgs.values())))

    gap_rows = []
    for cluster_id, avg_views in cluster_avgs.items():
        if avg_views >= overall_avg:
            continue

        opportunity_score = round(
            float((overall_avg - avg_views) / max(overall_avg, 1)),
            4,
        )
        keywords = (
            cluster_keywords.get(cluster_id)
            or cluster_keywords.get(str(cluster_id), [])
        )
        gap_rows.append({
            "cluster_id": int(cluster_id),
            "keywords": ";".join(keywords),
            "avg_views": round(float(avg_views), 2),
            "opportunity_score": opportunity_score,
        })

    gap_rows.sort(
        key=lambda item: item["opportunity_score"],
        reverse=True,
    )

    out_path = f"backend/saved_models/gaps_{niche}.csv"
    pd.DataFrame(
        gap_rows,
        columns=[
            "cluster_id",
            "keywords",
            "avg_views",
            "opportunity_score",
        ],
    ).to_csv(out_path, index=False)

    print(f"  Gaps saved: {len(gap_rows)} clusters → {out_path}")
    return gap_rows


def _duration_encoded(seconds) -> int:
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return -1
    if seconds <= 0:
        return -1
    if seconds < 120:
        return 0  # short
    if seconds < 600:
        return 1  # medium
    if seconds < 1800:
        return 2  # long
    return 3  # very_long


def extract_blend_side_features(
    title: str,
    duration_seconds=None,
    day_of_week=None,
    hour_of_day=None,
) -> dict:
    """
    Features for one side of a title comparison.
    Matches columns used to train blend_{niche}.pkl (selected top-25).
    """
    title = title or ""
    words_raw = re.findall(r"\b[\w']+\b", title)
    words_lower = [w.lower() for w in words_raw]
    word_lengths = [len(w) for w in words_lower]
    letters = [c for c in title if c.isalpha()]
    emoji_count = len(
        re.findall(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", title)
    )
    non_space = sum(not c.isspace() for c in title)

    if duration_seconds is None:
        duration_seconds = 600.0
    if day_of_week is None:
        day_of_week = 2
    if hour_of_day is None:
        hour_of_day = 14

    return {
        "duration_seconds": float(duration_seconds),
        "duration_encoded": float(_duration_encoded(duration_seconds)),
        "hour_of_day": float(int(hour_of_day)),
        "day_of_week": float(int(day_of_week)),
        "title_length": float(len(title)),
        "word_count": float(len(title.split()) if title else 0),
        "has_number": float(int(bool(re.search(r"\d", title)))),
        "has_exclamation": float(int("!" in title)),
        "uppercase_count": float(sum(c.isupper() for c in title)),
        "uppercase_ratio": (
            sum(c.isupper() for c in letters) / len(letters)
            if letters
            else 0.0
        ),
        "all_caps_word_count": float(
            sum(1 for w in words_raw if len(w) >= 2 and w.isupper())
        ),
        "digit_count": float(sum(c.isdigit() for c in title)),
        "char_density": float(non_space / len(title)) if title else 0.0,
        "space_count": float(title.count(" ")),
        "word_length_mean": (
            float(np.mean(word_lengths)) if word_lengths else 0.0
        ),
        "word_length_max": float(max(word_lengths)) if word_lengths else 0.0,
        "long_word_count": float(sum(L >= 8 for L in word_lengths)),
        "unique_word_ratio": (
            len(set(words_lower)) / len(words_lower) if words_lower else 0.0
        ),
        "punctuation_count": float(
            sum(c in "!?.,:;—-()[]{}\"'" for c in title)
        ),
        "exclamation_count": float(title.count("!")),
        "has_year": float(bool(re.search(r"\b20[2-3]\d\b", title))),
        "emoji_count": float(emoji_count),
        "hashtag_count": float(title.count("#")),
        "has_quotes": float(int('"' in title or "'" in title)),
    }


def blend_pair_feature_vector(
    sim_a: float,
    sim_b: float,
    feats_a: dict,
    feats_b: dict,
) -> list[float]:
    """B − A diffs in BLEND_FEATURE_NAMES order."""
    vec = []
    for name in BLEND_FEATURE_NAMES:
        if name == "tfidf_diff":
            vec.append(float(sim_b) - float(sim_a))
        else:
            col = name[: -len("_diff")]
            vec.append(float(feats_b[col]) - float(feats_a[col]))
    return vec


def _inline_title_feats(title: str) -> dict:
    """Backward-compatible alias — prefer extract_blend_side_features."""
    return extract_blend_side_features(title)


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
    RandomForest on selected top-25 diffs (no age_days).
    """
    print(f"\n  Training blend model for {niche}...")

    needed = [
        "title", "view_velocity", "channel_id",
        "age_days", "performance_label", "video_id",
        "duration_seconds", "duration_encoded",
        "day_of_week", "hour_of_day",
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

    rng = np.random.default_rng(BLEND_PAIR_SEED)
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
            if rng.random() < 0.5:
                pairs.append((row_b, row_a))
            else:
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

    # Precompute side features once per video
    side_by_vid = {}
    for _, row in work.iterrows():
        side_by_vid[row["video_id"]] = extract_blend_side_features(
            row["title"],
            duration_seconds=row["duration_seconds"],
            day_of_week=row["day_of_week"],
            hour_of_day=row["hour_of_day"],
        )

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
        X.append(
            blend_pair_feature_vector(
                sim_a,
                sim_b,
                side_by_vid[row_a["video_id"]],
                side_by_vid[row_b["video_id"]],
            )
        )
        y.append(
            1 if float(row_b["view_velocity"]) > float(row_a["view_velocity"])
            else 0
        )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    X_aug = np.vstack([X, -X])
    y_aug = np.concatenate([y, 1 - y])

    clf = RandomForestClassifier(**BLEND_RF_PARAMS)
    clf.fit(X_aug, y_aug)

    probs = clf.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    train_acc = float((preds == y).mean())

    importances = {
        name: float(imp)
        for name, imp in zip(BLEND_FEATURE_NAMES, clf.feature_importances_)
    }
    top_imp = sorted(importances.items(), key=lambda kv: -kv[1])[:8]

    joblib.dump(
        {
            "model": clf,
            "feature_names": BLEND_FEATURE_NAMES,
            "model_type": "random_forest",
            "rf_params": BLEND_RF_PARAMS,
        },
        f"backend/saved_models/blend_{niche}.pkl",
    )

    print(f"  Model: RandomForest ({len(BLEND_FEATURE_NAMES)} features)")
    print(f"  Training accuracy: {train_acc * 100:.1f}%")
    print("  Top feature importances:")
    for name, imp in top_imp:
        print(f"    {name}: {imp:.4f}")

    held_out = EVAL_BLEND_ACC.get(niche)
    if held_out is not None:
        delta_pp = (train_acc - held_out) * 100
        if train_acc >= 0.95:
            print(
                f"  FLAG: training accuracy {train_acc * 100:.1f}% is "
                f"near 100% (possible overfitting vs held-out "
                f"{held_out * 100:.1f}%)"
            )
        elif train_acc <= 0.52:
            print(
                f"  FLAG: training accuracy {train_acc * 100:.1f}% is "
                f"near chance — something may be broken "
                f"(held-out was {held_out * 100:.1f}%)"
            )
        elif abs(delta_pp) > 25:
            print(
                f"  FLAG: training accuracy differs from held-out by "
                f"{delta_pp:+.1f}pp "
                f"(train {train_acc * 100:.1f}% vs "
                f"held-out {held_out * 100:.1f}%)"
            )
        else:
            print(
                f"  Sanity vs held-out: train {train_acc * 100:.1f}% vs "
                f"{held_out * 100:.1f}% ({delta_pp:+.1f}pp)"
            )

    return {
        "niche": niche,
        "num_pairs": len(pairs),
        "importances": importances,
        "train_accuracy": round(train_acc, 4),
        "held_out_accuracy": held_out,
        "n_features": len(BLEND_FEATURE_NAMES),
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
            kmeans_result = train_kmeans_for_niche(df, niche)
            if kmeans_result:
                kmeans, cluster_keywords = kmeans_result
                compute_and_save_gaps(
                    df, niche, tfidf, kmeans, cluster_keywords
                )
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
        print("BLEND MODEL SUMMARY (RandomForest, title_meta)")
        print(f"{'='*72}")
        print(
            f"{'Niche':<15} {'Pairs':>8} {'Feats':>6} "
            f"{'Train%':>8} {'Held%':>8}"
        )
        print(
            f"{'-' * 15} {'-' * 8} {'-' * 6} "
            f"{'-' * 8} {'-' * 8}"
        )
        for row in blend_results:
            held = row.get("held_out_accuracy")
            held_txt = (
                f"{held * 100:7.1f}%" if held is not None else f"{'-':>8}"
            )
            print(
                f"{row['niche']:<15} {row['num_pairs']:>8,} "
                f"{row['n_features']:>6} "
                f"{row['train_accuracy'] * 100:>7.1f}% "
                f"{held_txt}"
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
