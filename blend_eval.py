"""
Evaluation for the production TubeIQ blend model (selected top-25 features).

Simple random 80/20 split over title pairs. Same RandomForest params and
feature extractor as train_models.py.

Run from the tubeiq project root:
    python blend_eval.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from backend.ml.train_models import (
    BLEND_FEATURE_NAMES,
    BLEND_RF_PARAMS,
    EVAL_BLEND_ACC,
    MIN_BLEND_PAIRS,
    NICHES,
    _avg_sim_excluding,
    blend_pair_feature_vector,
    extract_blend_side_features,
)
from feature_model_sweep import (
    build_pairs,
    build_sim_index,
    fit_tfidf,
    load_and_filter,
    SEED,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RANDOM_BASELINE = 0.5
TEST_FRACTION = 0.2


def build_xy(pairs, per_video, ref_ids):
    X, y = [], []
    for row_a, row_b in pairs:
        exclude = {row_a["video_id"], row_b["video_id"]}
        sim_a = _avg_sim_excluding(
            per_video[row_a["video_id"]], ref_ids, exclude
        )
        sim_b = _avg_sim_excluding(
            per_video[row_b["video_id"]], ref_ids, exclude
        )
        feats_a = extract_blend_side_features(
            row_a["title"],
            duration_seconds=row_a.get("duration_seconds"),
            day_of_week=row_a.get("day_of_week"),
            hour_of_day=row_a.get("hour_of_day"),
        )
        feats_b = extract_blend_side_features(
            row_b["title"],
            duration_seconds=row_b.get("duration_seconds"),
            day_of_week=row_b.get("day_of_week"),
            hour_of_day=row_b.get("hour_of_day"),
        )
        X.append(
            blend_pair_feature_vector(sim_a, sim_b, feats_a, feats_b)
        )
        y.append(
            1
            if float(row_b["view_velocity"]) > float(row_a["view_velocity"])
            else 0
        )
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def evaluate_niche(niche: str) -> dict | None:
    print(f"\n{'=' * 72}")
    print(f"NICHE: {niche.upper()}")
    print(f"{'=' * 72}")

    df = load_and_filter(niche)
    if df is None or df.empty:
        print("  Skipping — no usable rows")
        return None

    n_channels = int(df["channel_id"].nunique())
    print(f"  Videos: {len(df):,} | Channels: {n_channels}")

    tfidf = fit_tfidf(df)
    if tfidf is None:
        print("  Skipping — <10 titles")
        return None
    per_video, ref_ids = build_sim_index(df, tfidf)
    if per_video is None:
        print("  Skipping — no High/Viral reference")
        return None

    pairs = build_pairs(df, np.random.default_rng(SEED))
    if len(pairs) < MIN_BLEND_PAIRS:
        print(f"  Skipping — only {len(pairs)} pairs")
        return None

    X, y = build_xy(pairs, per_video, ref_ids)
    print(
        f"  Pairs: {len(pairs):,} | features: {X.shape[1]} | "
        f"positive rate={y.mean() * 100:.1f}%"
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=TEST_FRACTION,
        random_state=SEED,
        stratify=y,
    )

    # Match production: train on both pair orientations
    X_train = np.vstack([X_tr, -X_tr])
    y_train = np.concatenate([y_tr, 1 - y_tr])

    clf = RandomForestClassifier(**BLEND_RF_PARAMS)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_te)[:, 1]
    preds = (probs > 0.5).astype(int)
    acc = float((preds == y_te).mean())
    auc = float(roc_auc_score(y_te, probs))

    print(f"  accuracy: {acc * 100:.1f}% on {len(y_te):,} test pairs")
    print(f"  ROC-AUC:  {auc:.3f}")

    held = EVAL_BLEND_ACC.get(niche)
    if held is not None:
        print(
            f"  vs stored held-out: {held * 100:.1f}% "
            f"({(acc - held) * 100:+.1f}pp)"
        )

    return {
        "niche": niche,
        "n_channels": n_channels,
        "n_pairs": len(pairs),
        "n_test": len(y_te),
        "correct": int((preds == y_te).sum()),
        "acc": acc,
        "auc": auc,
        "y_test": y_te,
        "probs": probs,
    }


def print_summary(summaries: list[dict]):
    print(f"\n{'=' * 88}")
    print("BLEND EVAL — RandomForest + top-25 features (random 80/20)")
    print(f"{'=' * 88}")
    print(
        f"{'Niche':<15} {'Ch':>3} {'Pairs':>8} {'Test':>8} "
        f"{'Acc %':>8} {'ROC-AUC':>9}"
    )
    print(
        f"{'-' * 15} {'-' * 3} {'-' * 8} {'-' * 8} "
        f"{'-' * 8} {'-' * 9}"
    )

    total_correct = 0
    total_test = 0
    for row in summaries:
        total_correct += row["correct"]
        total_test += row["n_test"]
        print(
            f"{row['niche']:<15} {row['n_channels']:>3} "
            f"{row['n_pairs']:>8,} {row['n_test']:>8,} "
            f"{row['acc'] * 100:7.1f}% {row['auc']:9.3f}"
        )

    print(
        f"{'-' * 15} {'-' * 3} {'-' * 8} {'-' * 8} "
        f"{'-' * 8} {'-' * 9}"
    )

    y = np.concatenate([r["y_test"] for r in summaries])
    p = np.concatenate([r["probs"] for r in summaries])
    overall_acc = total_correct / total_test
    overall_auc = float(roc_auc_score(y, p))
    print(
        f"{'OVERALL':<15} {'':>3} {'':>8} {total_test:>8,} "
        f"{overall_acc * 100:7.1f}% {overall_auc:9.3f}"
    )

    print(
        f"\nBaseline: {RANDOM_BASELINE * 100:.0f}% accuracy / 0.500 AUC "
        f"(balanced pair orientation)"
    )
    print(f"Features: {len(BLEND_FEATURE_NAMES)} (importance-selected)")
    print(
        f"Range across niches: "
        f"{min(r['acc'] for r in summaries) * 100:.1f}%–"
        f"{max(r['acc'] for r in summaries) * 100:.1f}%"
    )


def main():
    os.chdir(PROJECT_ROOT)
    print("=" * 88)
    print("TUBEIQ BLEND MODEL EVAL (random 80/20 pair split)")
    print("=" * 88)
    print(f"Features: {len(BLEND_FEATURE_NAMES)}")
    print(
        f"Split: random {int((1 - TEST_FRACTION) * 100)}/"
        f"{int(TEST_FRACTION * 100)} over pairs, stratified, seed={SEED}"
    )

    summaries = []
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
