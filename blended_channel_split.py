import random
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.ml.train_models import NICHES, clean_title  # noqa: E402

MIN_VELOCITY_RATIO = 2.0
MAX_AGE_GAP_DAYS = 120
SEED = 42
MIN_TRAIN_PAIRS = 200
TEST_FRACTION = 0.2  # ~20% of channels held out for testing


def struct_features(title):
    title = title or ""
    return {
        "title_length": len(title),
        "has_number": int(bool(re.search(r"\d", title))),
        "has_question": int("?" in title),
    }


def load_niche(niche):
    path = Path("backend/data") / f"{niche}_features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["title", "view_velocity", "channel_id", "age_days"])
    df = df[df["view_velocity"] > 0].reset_index(drop=True)
    return df


def gen_pairs(cdf):
    pairs = []
    for a, b in combinations(cdf.to_dict("records"), 2):
        va, vb = a["view_velocity"], b["view_velocity"]
        if va == vb:
            continue
        lo, hi = min(va, vb), max(va, vb)
        if lo <= 0 or hi / lo < MIN_VELOCITY_RATIO:
            continue
        if abs(a["age_days"] - b["age_days"]) > MAX_AGE_GAP_DAYS:
            continue
        pairs.append((a, b))
    return pairs


def gen_all_pairs(df):
    pairs = []
    for _, cdf in df.groupby("channel_id"):
        pairs.extend(gen_pairs(cdf))
    return pairs


def fit_tfidf(train_df):
    train_df = train_df.copy()
    train_df["clean_title"] = train_df.apply(
        lambda r: clean_title(r.get("title", ""), r.get("channel_title", "")), axis=1
    )
    titles = train_df["clean_title"].fillna("").tolist()
    if len(titles) < 10:
        return None
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english", min_df=2)
    tfidf.fit(titles)
    return tfidf


def build_ref(train_df, tfidf):
    high_df = train_df[train_df["performance_label"].isin(["High", "Viral"])]
    high_df = high_df[high_df["title"].fillna("").str.strip() != ""]
    if high_df.empty:
        return np.array([])
    return tfidf.transform(high_df["title"].fillna("").tolist())


def sim_lookup(df_subset, tfidf, ref_vecs):
    if ref_vecs.shape[0] == 0:
        return {}
    titles = df_subset["title"].fillna("").tolist()
    vecs = tfidf.transform(titles)
    sim_matrix = cosine_similarity(vecs, ref_vecs)
    means = sim_matrix.mean(axis=1)
    return {vid: means[i] for i, vid in enumerate(df_subset["video_id"].tolist())}


def pair_x(a, b, sims):
    sim_a, sim_b = sims.get(a["video_id"], 0.0), sims.get(b["video_id"], 0.0)
    fa, fb = struct_features(a["title"]), struct_features(b["title"])
    return [
        sim_b - sim_a,
        fb["has_number"] - fa["has_number"],
        fb["has_question"] - fa["has_question"],
        fb["title_length"] - fa["title_length"],
    ]


def run_channel_split_blended(df, seed):
    channel_ids = sorted(df["channel_id"].unique().tolist())
    if len(channel_ids) < 2:
        return None

    random.seed(seed)
    shuffled = channel_ids.copy()
    random.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * TEST_FRACTION))
    test_channels = shuffled[:n_test]
    train_channels = shuffled[n_test:]

    train_df = df[df["channel_id"].isin(train_channels)]
    test_df = df[df["channel_id"].isin(test_channels)]

    tfidf = fit_tfidf(train_df)
    if tfidf is None:
        return None
    ref_vecs = build_ref(train_df, tfidf)
    if ref_vecs.shape[0] == 0:
        return None

    train_sims = sim_lookup(train_df, tfidf, ref_vecs)
    train_pairs = gen_all_pairs(train_df)
    if len(train_pairs) < MIN_TRAIN_PAIRS:
        return None

    X_train, y_train = [], []
    for a, b in train_pairs:
        X_train.append(pair_x(a, b, train_sims))
        y_train.append(int(b["view_velocity"] > a["view_velocity"]))
    X_train, y_train = np.array(X_train), np.array(y_train)
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)

    test_sims = sim_lookup(test_df, tfidf, ref_vecs)
    test_pairs = gen_all_pairs(test_df)
    if not test_pairs:
        return None

    correct = 0
    for a, b in test_pairs:
        x = pair_x(a, b, test_sims)
        y_true = int(b["view_velocity"] > a["view_velocity"])
        pred = int(clf.predict(scaler.transform([x]))[0])
        correct += int(pred == y_true)

    return {
        "acc": correct / len(test_pairs) * 100,
        "n_pairs": len(test_pairs),
        "train_channels": train_channels,
        "test_channels": test_channels,
        "n_train_videos": len(train_df),
        "n_test_videos": len(test_df),
        "tfidf": tfidf,
        "ref_vecs": ref_vecs,
        "scaler": scaler,
        "clf": clf,
    }


def main():
    print("SINGLE CHANNEL-LEVEL SPLIT, BLENDED MODEL")
    print("Train the model on ~80% of channels. Hold out the remaining ~20% of channels")
    print("ENTIRELY. Generate test pairs only from those held-out channels' own videos.\n")

    print(f"{'Niche':<15}{'Train ch':>10}{'Test ch':>9}{'TrainVids':>11}{'TestVids':>10}{'Pairs':>10}{'Accuracy':>11}")
    print("-" * 76)

    total_correct = 0
    total_pairs = 0
    for niche in NICHES:
        df = load_niche(niche)
        if df is None or df.empty:
            print(f"{niche:<15}  no data")
            continue
        result = run_channel_split_blended(df, SEED)
        if result is None:
            print(f"{niche:<15}  insufficient data for this split")
            continue
        print(
            f"{niche:<15}{len(result['train_channels']):>10}{len(result['test_channels']):>9}"
            f"{result['n_train_videos']:>11,}{result['n_test_videos']:>10,}"
            f"{result['n_pairs']:>10,}{result['acc']:>10.2f}%"
        )
        total_correct += round(result["acc"] / 100 * result["n_pairs"])
        total_pairs += result["n_pairs"]

        import joblib
        import os
        os.makedirs("backend/saved_models", exist_ok=True)
        joblib.dump(
            {
                "tfidf": result["tfidf"],
                "ref_vecs": result["ref_vecs"],
                "scaler": result["scaler"],
                "model": result["clf"],
                "trained_on_channels": result["train_channels"],
                "held_out_channels": result["test_channels"],
            },
            f"backend/saved_models/blend_holdout80_{niche}.pkl",
        )

    print("-" * 76)
    if total_pairs:
        overall = total_correct / total_pairs * 100
        print(f"{'OVERALL':<15}{'':>10}{'':>9}{'':>11}{'':>10}{total_pairs:>10,}{overall:>10.2f}%")


if __name__ == "__main__":
    main()
