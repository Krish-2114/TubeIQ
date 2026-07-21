"""
Leave-one-channel-out (LOCO) evaluation of TF-IDF title comparison.

Fits TF-IDF + High/Viral reference set on all channels except one held-out
channel, then evaluates same-channel pairs drawn only from the held-out
channel. Rotates so every channel is unseen exactly once.

Run from the tubeiq project root:
    python unseen_channel_eval.py
"""

from __future__ import annotations

import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.ml.train_models import NICHES, clean_title

PROJECT_ROOT = Path(__file__).resolve().parent
MIN_VIEW_RATIO = 2.0
MAX_AGE_GAP_DAYS = 120
RANDOM_BASELINE = 0.5
DATA_DIR = PROJECT_ROOT / "backend" / "data"


def load_and_filter(niche: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{niche}_features.csv"
    if not path.exists():
        print(f"  Skipping {niche} — CSV not found at {path}")
        return None

    df = pd.read_csv(path)
    df = df.dropna(subset=["title", "view_velocity", "channel_id", "age_days"])
    df = df[df["view_velocity"] > 0].copy()
    df["title"] = df["title"].fillna("").astype(str)
    if "channel_title" not in df.columns:
        df["channel_title"] = ""
    return df


def generate_velocity_pairs(channel_df: pd.DataFrame) -> list[tuple]:
    """Same-channel pairs: >=2x view_velocity ratio AND age_days gap <= 120."""
    pairs = []
    records = channel_df.to_dict("records")

    for row_a, row_b in combinations(records, 2):
        age_gap = abs(float(row_a["age_days"]) - float(row_b["age_days"]))
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

    return pairs


def fit_tfidf(train_df: pd.DataFrame) -> TfidfVectorizer | None:
    cleaned = train_df.apply(
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


def build_reference_vectors(
    train_df: pd.DataFrame,
    tfidf: TfidfVectorizer,
):
    """High/Viral raw titles → TF-IDF vectors (matches train_models.py)."""
    high_df = train_df[
        train_df["performance_label"].isin(["High", "Viral"])
    ]
    titles = high_df["title"].fillna("").tolist()
    if not titles:
        return None
    return tfidf.transform(titles)


def score_pairs(
    pairs: list[tuple],
    tfidf: TfidfVectorizer,
    ref_vectors,
) -> tuple[int, int]:
    """
    Score pairs against the FULL reference set.
    Returns (correct, total). Ties count as incorrect.
    """
    if not pairs:
        return 0, 0

    titles_a = [p[0]["title"] for p in pairs]
    titles_b = [p[1]["title"] for p in pairs]
    vecs_a = tfidf.transform(titles_a)
    vecs_b = tfidf.transform(titles_b)

    sims_a = cosine_similarity(vecs_a, ref_vectors).mean(axis=1)
    sims_b = cosine_similarity(vecs_b, ref_vectors).mean(axis=1)

    correct = 0
    for i, (row_a, row_b) in enumerate(pairs):
        sim_a = float(sims_a[i])
        sim_b = float(sims_b[i])
        if sim_a == sim_b:
            continue  # tie → incorrect (not counted as correct)

        predicted_b_wins = sim_b > sim_a
        actual_b_wins = float(row_b["view_velocity"]) > float(
            row_a["view_velocity"]
        )
        if predicted_b_wins == actual_b_wins:
            correct += 1

    return correct, len(pairs)


def evaluate_niche(niche: str) -> dict | None:
    print(f"\n{'=' * 64}")
    print(f"NICHE: {niche.upper()}")
    print(f"{'=' * 64}")

    df = load_and_filter(niche)
    if df is None or df.empty:
        print("  Skipping — no usable rows after filter")
        return None

    channel_ids = sorted(df["channel_id"].dropna().unique().tolist())
    n_channels = len(channel_ids)
    print(f"  Videos: {len(df):,} | Channels: {n_channels}")

    if n_channels < 2:
        print(
            f"  Skipping — only {n_channels} unique channel(s); "
            "need >= 2 for leave-one-channel-out"
        )
        return None

    fold_rows: list[dict] = []
    total_correct = 0
    total_pairs = 0

    for held_out_id in channel_ids:
        train_df = df[df["channel_id"] != held_out_id]
        test_df = df[df["channel_id"] == held_out_id]

        channel_title = ""
        if not test_df.empty and "channel_title" in test_df.columns:
            channel_title = str(test_df["channel_title"].iloc[0])

        print(
            f"\n  Fold held-out channel_id={held_out_id}"
            f"{f' ({channel_title})' if channel_title else ''} "
            f"| train={len(train_df):,} test={len(test_df):,}",
            flush=True,
        )

        tfidf = fit_tfidf(train_df)
        if tfidf is None:
            print("    Skip fold — fewer than 10 train titles after cleaning")
            fold_rows.append({
                "held_out_id": held_out_id,
                "channel_title": channel_title,
                "num_pairs": 0,
                "accuracy": None,
                "correct": 0,
            })
            continue

        ref_vectors = build_reference_vectors(train_df, tfidf)
        if ref_vectors is None or ref_vectors.shape[0] == 0:
            print("    Skip fold — zero High/Viral titles in train set")
            fold_rows.append({
                "held_out_id": held_out_id,
                "channel_title": channel_title,
                "num_pairs": 0,
                "accuracy": None,
                "correct": 0,
            })
            continue

        pairs = generate_velocity_pairs(test_df)
        if not pairs:
            print("    No valid pairs in held-out channel")
            fold_rows.append({
                "held_out_id": held_out_id,
                "channel_title": channel_title,
                "num_pairs": 0,
                "accuracy": None,
                "correct": 0,
            })
            continue

        print(
            f"    Reference set: {ref_vectors.shape[0]:,} | "
            f"pairs: {len(pairs):,} ...",
            end=" ",
            flush=True,
        )
        correct, n_pairs = score_pairs(pairs, tfidf, ref_vectors)
        acc = correct / n_pairs if n_pairs else None
        print(
            f"accuracy {acc * 100:.1f}% ({correct}/{n_pairs})",
            flush=True,
        )

        fold_rows.append({
            "held_out_id": held_out_id,
            "channel_title": channel_title,
            "num_pairs": n_pairs,
            "accuracy": acc,
            "correct": correct,
        })
        total_correct += correct
        total_pairs += n_pairs

    niche_acc = (
        total_correct / total_pairs if total_pairs > 0 else None
    )
    print(f"\n  Niche total: {total_pairs:,} pairs | ", end="")
    if niche_acc is not None:
        print(
            f"accuracy {niche_acc * 100:.1f}% "
            f"(baseline {RANDOM_BASELINE * 100:.0f}%)"
        )
    else:
        print("no pairs evaluated")

    print("  Per-fold breakdown:")
    for fold in fold_rows:
        label = fold["channel_title"] or str(fold["held_out_id"])
        if fold["num_pairs"] == 0 or fold["accuracy"] is None:
            print(f"    {label}: 0 pairs")
        else:
            print(
                f"    {label}: {fold['num_pairs']:,} pairs | "
                f"{fold['accuracy'] * 100:.1f}%"
            )

    return {
        "niche": niche,
        "n_channels": n_channels,
        "num_pairs": total_pairs,
        "correct": total_correct,
        "accuracy": niche_acc,
        "folds": fold_rows,
    }


def print_summary(summaries: list[dict]):
    print(f"\n{'=' * 72}")
    print("UNSEEN-CHANNEL (LOCO) EVAL — SUMMARY")
    print(f"{'=' * 72}")
    print(
        f"{'Niche':<15} {'Channels':>8} {'Pairs':>10} "
        f"{'Accuracy %':>12} {'vs 50%':>10}"
    )
    print(f"{'-' * 15} {'-' * 8} {'-' * 10} {'-' * 12} {'-' * 10}")

    overall_correct = 0
    overall_pairs = 0

    for row in summaries:
        overall_correct += row["correct"]
        overall_pairs += row["num_pairs"]
        if row["num_pairs"] == 0 or row["accuracy"] is None:
            print(
                f"{row['niche']:<15} {row['n_channels']:>8} "
                f"{0:>10} {'—':>12} {'—':>10}"
            )
            continue
        acc_pct = row["accuracy"] * 100
        delta = acc_pct - RANDOM_BASELINE * 100
        sign = "+" if delta >= 0 else ""
        print(
            f"{row['niche']:<15} {row['n_channels']:>8} "
            f"{row['num_pairs']:>10,} "
            f"{acc_pct:>11.1f}% "
            f"{sign}{delta:>8.1f}pp"
        )

    print(f"{'-' * 15} {'-' * 8} {'-' * 10} {'-' * 12} {'-' * 10}")
    if overall_pairs > 0:
        overall_acc = overall_correct / overall_pairs * 100
        delta = overall_acc - RANDOM_BASELINE * 100
        sign = "+" if delta >= 0 else ""
        print(
            f"{'OVERALL':<15} {'':>8} "
            f"{overall_pairs:>10,} "
            f"{overall_acc:>11.1f}% "
            f"{sign}{delta:>8.1f}pp"
        )
        print(
            f"\nOverall: {overall_correct:,}/{overall_pairs:,} correct "
            f"= {overall_acc:.1f}%  |  random baseline = "
            f"{RANDOM_BASELINE * 100:.0f}%"
        )
    else:
        print(f"{'OVERALL':<15} {'':>8} {0:>10} {'—':>12} {'—':>10}")


def main():
    os.chdir(PROJECT_ROOT)

    print("=" * 72)
    print("TUBEIQ UNSEEN-CHANNEL EVAL (Leave-One-Channel-Out)")
    print("=" * 72)
    print(
        "Fit TF-IDF + High/Viral reference on all-but-one channel; "
        "evaluate pairs from the held-out channel only."
    )
    print(
        f"Pair rules: same held-out channel, "
        f">= {MIN_VIEW_RATIO:.0f}x view_velocity ratio, "
        f"age_days gap <= {MAX_AGE_GAP_DAYS}"
    )
    print(f"Random baseline: {RANDOM_BASELINE * 100:.0f}%")

    summaries: list[dict] = []
    for niche in NICHES:
        result = evaluate_niche(niche)
        if result is not None:
            summaries.append(result)

    if not summaries or all(s["num_pairs"] == 0 for s in summaries):
        print("\nNo pairs evaluated across any niche.")
        sys.exit(1)

    print_summary(summaries)
    print("\nDone.")


if __name__ == "__main__":
    main()
