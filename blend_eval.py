"""
Evaluation for the production TubeIQ blend model.

The task is pairwise: given two videos from the same channel, predict which one
earned more views per day. Every valid pair in a niche is pooled and split 80/20
at random. Reports pairwise accuracy and ROC-AUC.

Features match the shipped model in backend/ml/train_models.py exactly, and the
same leakage guard is reused, so these numbers describe the model that
backend/main.py actually serves from blend_{niche}.pkl:
  tfidf_diff, has_number_diff, has_question_diff, title_length_diff

Pair rules: same channel, >= 2x view_velocity ratio, age_days gap <= 120.
Pair orientation (which video is A vs B) is randomized with a fixed seed so the
label is balanced ~50/50, which is what makes 50% the correct accuracy baseline.

CAVEAT — pairs are split randomly rather than by channel, so a channel can
appear in both train and test and the model can lean on channel-specific style.
Read these numbers as "how well does it rank a creator's titles against their
own back catalogue," not "how well does it generalize to a brand-new channel."

Run from the tubeiq project root:
    python blend_eval.py
"""

from __future__ import annotations

import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backend.ml.train_models import (
    MAX_AGE_GAP_DAYS,
    MIN_BLEND_PAIRS,
    MIN_VIEW_RATIO,
    NICHES,
    _avg_sim_excluding,
    _inline_title_feats,
    clean_title,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "backend" / "data"
RANDOM_BASELINE = 0.5
TEST_FRACTION = 0.2
SEED = 42


def load_and_filter(niche: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{niche}_features.csv"
    if not path.exists():
        print(f"  Skipping {niche} — CSV not found at {path}")
        return None

    needed = [
        "title",
        "view_velocity",
        "channel_id",
        "age_days",
        "performance_label",
    ]
    df = pd.read_csv(path)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"  Skipping {niche} — missing columns: {missing}")
        return None

    df = df.dropna(subset=needed)
    df = df[df["view_velocity"] > 0].copy()
    df["title"] = df["title"].fillna("").astype(str)
    if "channel_title" not in df.columns:
        df["channel_title"] = ""
    if "video_id" not in df.columns:
        df["video_id"] = df.index.astype(str)
    # video_id keys the similarity lookup, so duplicates would collide
    df = df.drop_duplicates(subset=["video_id"]).reset_index(drop=True)
    return df


def fit_tfidf(df: pd.DataFrame) -> TfidfVectorizer | None:
    cleaned = df.apply(
        lambda row: clean_title(
            row.get("title", ""),
            row.get("channel_title", ""),
        ),
        axis=1,
    ).fillna("").tolist()
    if len(cleaned) < 10:
        return None
    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    tfidf.fit(cleaned)
    return tfidf


def build_sim_index(df: pd.DataFrame, tfidf: TfidfVectorizer):
    """Per-video similarity row against the High/Viral reference set."""
    high_df = df[df["performance_label"].isin(["High", "Viral"])]
    high_df = high_df[high_df["title"].str.strip() != ""]
    if high_df.empty:
        return None, None

    ref_vectors = tfidf.transform(high_df["title"].tolist())
    ref_video_ids = high_df["video_id"].tolist()

    all_vectors = tfidf.transform(df["title"].tolist())
    sim_matrix = cosine_similarity(all_vectors, ref_vectors)
    per_video = {
        vid: sim_matrix[i]
        for i, vid in enumerate(df["video_id"].tolist())
    }
    return per_video, ref_video_ids


def build_all_pairs(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> list[tuple]:
    """Same-channel pairs: >=2x velocity ratio AND age_days gap <= 120."""
    pairs = []
    for _, channel_df in df.groupby("channel_id"):
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
            # combinations() always yields the earlier CSV row first, and row
            # order correlates with velocity — that alone drives the positive
            # rate to ~36% and makes 50% the wrong accuracy baseline. Coin-flip
            # the orientation so the label carries no positional information.
            if rng.random() < 0.5:
                pairs.append((row_b, row_a))
            else:
                pairs.append((row_a, row_b))
    return pairs


def pair_feature_row(
    row_a: dict,
    row_b: dict,
    per_video: dict,
    ref_video_ids: list,
) -> tuple[list[float], int]:
    # Both videos are dropped from the reference set before averaging, so a
    # High/Viral video cannot match itself and leak its own label.
    exclude = {row_a["video_id"], row_b["video_id"]}
    sim_a = _avg_sim_excluding(
        per_video[row_a["video_id"]], ref_video_ids, exclude
    )
    sim_b = _avg_sim_excluding(
        per_video[row_b["video_id"]], ref_video_ids, exclude
    )
    feats_a = _inline_title_feats(row_a["title"])
    feats_b = _inline_title_feats(row_b["title"])
    x = [
        sim_b - sim_a,
        feats_b["has_number"] - feats_a["has_number"],
        feats_b["has_question"] - feats_a["has_question"],
        feats_b["title_length"] - feats_a["title_length"],
    ]
    label = (
        1
        if float(row_b["view_velocity"]) > float(row_a["view_velocity"])
        else 0
    )
    return x, label


def evaluate_niche(niche: str) -> dict | None:
    print(f"\n{'=' * 72}")
    print(f"NICHE: {niche.upper()}")
    print(f"{'=' * 72}")

    df = load_and_filter(niche)
    if df is None or df.empty:
        print("  Skipping — no usable rows")
        return None

    n_channels = df["channel_id"].nunique()
    print(f"  Videos: {len(df):,} | Channels: {n_channels}")

    tfidf = fit_tfidf(df)
    if tfidf is None:
        print("  Skipping — <10 titles for TF-IDF")
        return None

    per_video, ref_video_ids = build_sim_index(df, tfidf)
    if per_video is None:
        print("  Skipping — no High/Viral reference titles")
        return None
    print(f"  Reference titles (High/Viral): {len(ref_video_ids):,}")

    pairs = build_all_pairs(df, np.random.default_rng(SEED))
    if len(pairs) < MIN_BLEND_PAIRS:
        print(
            f"  Skipping — only {len(pairs)} valid pairs "
            f"(need >= {MIN_BLEND_PAIRS})"
        )
        return None

    X, y = [], []
    for row_a, row_b in pairs:
        feats, label = pair_feature_row(
            row_a, row_b, per_video, ref_video_ids
        )
        X.append(feats)
        y.append(label)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if len(np.unique(y)) < 2:
        print("  Skipping — only one class across all pairs")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_FRACTION,
        random_state=SEED,
        stratify=y,
    )
    print(
        f"  Pairs: {len(pairs):,} "
        f"(train={len(y_train):,} test={len(y_test):,}) | "
        f"positive rate={y.mean() * 100:.1f}%"
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
        solver="lbfgs",
    )
    model.fit(X_train_s, y_train)

    # decision_function == 0 is a tie → counted incorrect, so the model gets no
    # free credit for abstaining
    scores = model.decision_function(X_test_s)
    preds = (scores > 0).astype(int)
    ties = scores == 0
    correct = int(np.sum((preds == y_test) & ~ties))
    acc = correct / len(y_test)

    probs = model.predict_proba(X_test_s)[:, 1]
    auc = (
        float(roc_auc_score(y_test, probs))
        if len(np.unique(y_test)) > 1
        else None
    )

    # Orientation is randomized, so this should sit at ~50%. It is printed as a
    # guard: if it drifts, the accuracy figure needs a different baseline.
    pos_rate = float(y_test.mean())
    majority_acc = max(pos_rate, 1.0 - pos_rate)

    print(f"    accuracy: {acc * 100:.1f}% on {len(y_test):,} test pairs")
    print(
        f"    majority-class baseline: {majority_acc * 100:.1f}% "
        f"({(acc - majority_acc) * 100:+.1f}pp)"
    )
    if auc is not None:
        print(f"    ROC-AUC:  {auc:.3f}")

    return {
        "niche": niche,
        "n_channels": n_channels,
        "n_pairs": len(pairs),
        "n_test": len(y_test),
        "correct": correct,
        "acc": acc,
        "majority_acc": majority_acc,
        "auc": auc,
        "y_test": y_test,
        "probs": probs,
    }


def print_summary(summaries: list[dict]):
    print(f"\n{'=' * 92}")
    print("BLEND MODEL EVAL — SUMMARY (random 80/20 over pooled pairs)")
    print(f"{'=' * 92}")
    print(
        f"{'Niche':<15} {'Ch':>4} {'Pairs':>9} {'Test':>8} "
        f"{'Acc %':>8} {'Base %':>8} {'Lift':>9} {'ROC-AUC':>9}"
    )
    print(
        f"{'-' * 15} {'-' * 4} {'-' * 9} {'-' * 8} "
        f"{'-' * 8} {'-' * 8} {'-' * 9} {'-' * 9}"
    )

    total_correct = 0
    total_test = 0
    for row in summaries:
        total_correct += row["correct"]
        total_test += row["n_test"]

        auc_txt = f"{row['auc']:9.3f}" if row["auc"] is not None else f"{'-':>9}"
        vs_maj = (row["acc"] - row["majority_acc"]) * 100

        print(
            f"{row['niche']:<15} {row['n_channels']:>4} "
            f"{row['n_pairs']:>9,} {row['n_test']:>8,} "
            f"{row['acc'] * 100:7.1f}% {row['majority_acc'] * 100:7.1f}% "
            f"{vs_maj:+8.1f}pp {auc_txt}"
        )

    print(
        f"{'-' * 15} {'-' * 4} {'-' * 9} {'-' * 8} "
        f"{'-' * 8} {'-' * 8} {'-' * 9} {'-' * 9}"
    )

    overall_acc = total_correct / total_test if total_test else None

    # Pool every niche's labels/probs first, then score once — an average of
    # per-niche AUCs would weight a 1,000-pair niche like an 11,000-pair one.
    pooled_y = np.concatenate([r["y_test"] for r in summaries])
    pooled_p = np.concatenate([r["probs"] for r in summaries])
    pooled_auc = (
        float(roc_auc_score(pooled_y, pooled_p))
        if len(np.unique(pooled_y)) > 1
        else None
    )

    pooled_pos_rate = float(pooled_y.mean())
    pooled_majority = max(pooled_pos_rate, 1.0 - pooled_pos_rate)

    auc_txt = f"{pooled_auc:9.3f}" if pooled_auc is not None else f"{'-':>9}"
    print(
        f"{'OVERALL':<15} {'':>4} {'':>9} {total_test:>8,} "
        f"{overall_acc * 100:7.1f}% {pooled_majority * 100:7.1f}% "
        f"{(overall_acc - pooled_majority) * 100:+8.1f}pp {auc_txt}"
    )

    print(
        f"\nBaseline: {RANDOM_BASELINE * 100:.0f}% accuracy / 0.500 AUC "
        f"(positive rate is {pooled_pos_rate * 100:.1f}%, so the label is "
        f"balanced and\n50% is the correct comparator)"
    )
    print(
        "\nNOTE: pairs are split randomly, so a channel can appear in both "
        "train and test.\nThese numbers describe ranking a creator's titles "
        "against their own history,\nnot generalization to an unseen channel."
    )


def main():
    os.chdir(PROJECT_ROOT)
    print("=" * 92)
    print("TUBEIQ BLEND MODEL EVAL (production 4-feature model)")
    print("=" * 92)
    print(
        "Features: tfidf_diff, has_number_diff, has_question_diff, "
        "title_length_diff"
    )
    print(
        f"Pair rules: same channel, "
        f">= {MIN_VIEW_RATIO:.0f}x view_velocity, "
        f"age_days gap <= {MAX_AGE_GAP_DAYS}"
    )
    print(
        f"Split: random {int((1 - TEST_FRACTION) * 100)}/"
        f"{int(TEST_FRACTION * 100)} over pooled pairs, "
        f"stratified, seed={SEED}"
    )

    summaries: list[dict] = []
    for niche in NICHES:
        result = evaluate_niche(niche)
        if result is not None:
            summaries.append(result)

    if not summaries:
        print("\nNo niches evaluated.")
        sys.exit(1)

    print_summary(summaries)
    print("\nDone.")


if __name__ == "__main__":
    main()
