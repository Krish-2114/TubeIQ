"""
Evaluate whether TF-IDF similarity to high-performing titles
predicts which of two same-channel videos performed better.

Run from the tubeiq project root:
    python title_comparison_eval.py

LEAKAGE NOTE:
  For each pair (A, B), the reference set of High/Viral titles used to
  score similarity excludes A and B themselves. The fitted TF-IDF
  vectorizer is loaded from saved_models (trained on the full niche corpus)
  and is never refit on comparison pairs.
"""

from __future__ import annotations

import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from backend.database import SessionLocal
from backend.main import load_niche_csv, load_tfidf
from backend.ml.train_models import NICHES
from backend.models import Channel, Video

PROJECT_ROOT = Path(__file__).resolve().parent
MAX_DAYS_APART = 120
MIN_VIEW_RATIO = 2.0
RANDOM_BASELINE = 0.5


def enrich_published_at(df: pd.DataFrame, niche: str) -> pd.DataFrame:
    """CSV exports omit published_at; join from DB by video_id."""
    if "video_id" not in df.columns:
        df["published_at"] = pd.NaT
        return df

    db = SessionLocal()
    try:
        rows = (
            db.query(Video.video_id, Video.published_at)
            .join(Channel, Video.channel_id == Channel.id)
            .filter(Channel.niche == niche)
            .all()
        )
        pub_map = {video_id: published_at for video_id, published_at in rows}
    finally:
        db.close()

    df = df.copy()
    df["published_at"] = pd.to_datetime(
        df["video_id"].map(pub_map),
        utc=True,
        errors="coerce",
    )
    return df


def load_niche_videos(niche: str) -> pd.DataFrame | None:
    path = PROJECT_ROOT / "backend" / "data" / f"{niche}_features.csv"
    if not path.exists():
        print(f"  Skipping {niche} — CSV not found")
        return None

    df = load_niche_csv(niche)
    df = enrich_published_at(df, niche)
    df = df.dropna(subset=["title", "view_count", "channel_id"])
    df = df[df["view_count"] > 0]
    return df


def build_reference_index(df: pd.DataFrame, tfidf):
    """
    Precompute TF-IDF vectors for all High/Viral titles in the niche.
    Per-pair scoring excludes the pair videos from this index (see below).
    """
    high_df = df[df["performance_label"].isin(["High", "Viral"])].copy()
    high_df = high_df.dropna(subset=["title"])
    high_df = high_df[high_df["title"].str.strip() != ""]

    if high_df.empty:
        return [], np.array([]), None

    ref_video_ids = high_df["video_id"].tolist()
    ref_titles = high_df["title"].fillna("").tolist()
    ref_vectors = tfidf.transform(ref_titles)
    return ref_video_ids, ref_vectors, high_df


def precompute_video_ref_similarities(
    df: pd.DataFrame,
    tfidf,
    ref_video_ids: list,
    ref_vectors,
) -> dict:
    """
    For each video, store cosine similarity to every reference title.
    Used to compute leakage-safe means by dropping pair members from the ref.
    """
    titles = df["title"].fillna("").tolist()
    video_ids = df["video_id"].tolist()
    video_vectors = tfidf.transform(titles)
    sim_matrix = cosine_similarity(video_vectors, ref_vectors)

    return {
        video_id: sim_matrix[i]
        for i, video_id in enumerate(video_ids)
    }


def avg_similarity_excluding(
    per_ref_sims: np.ndarray,
    ref_video_ids: list,
    exclude_video_ids: set,
) -> float:
    """
    Mean similarity to the reference set, excluding pair members (leakage guard).
    """
    keep = [
        i
        for i, vid in enumerate(ref_video_ids)
        if vid not in exclude_video_ids
    ]
    if not keep:
        return 0.0
    return float(np.mean(per_ref_sims[keep]))


def days_apart(dt_a, dt_b) -> float | None:
    if pd.isna(dt_a) or pd.isna(dt_b):
        return None
    return abs((dt_a - dt_b).total_seconds()) / 86400.0


def generate_channel_pairs(channel_df: pd.DataFrame) -> list[tuple]:
    """Valid pairs: same channel, within 120 days, 2x view ratio."""
    pairs = []
    records = channel_df.to_dict("records")

    for row_a, row_b in combinations(records, 2):
        gap = days_apart(row_a["published_at"], row_b["published_at"])
        if gap is None or gap > MAX_DAYS_APART:
            continue

        views_a = int(row_a["view_count"])
        views_b = int(row_b["view_count"])
        if views_a == views_b:
            continue

        lo, hi = min(views_a, views_b), max(views_a, views_b)
        if lo <= 0 or hi / lo < MIN_VIEW_RATIO:
            continue

        pairs.append((row_a, row_b))

    return pairs


def evaluate_niche(niche: str) -> tuple[list[dict], dict]:
    print(f"\n>>> Niche: {niche}")

    try:
        tfidf = load_tfidf(niche)
    except Exception as exc:
        print(f"  Skipping — could not load TF-IDF: {exc}")
        return [], {}

    df = load_niche_videos(niche)
    if df is None or df.empty:
        print("  Skipping — no video data")
        return [], {}

    ref_video_ids, ref_vectors, _ = build_reference_index(df, tfidf)
    if not ref_video_ids:
        print("  Skipping — no High/Viral reference titles")
        return [], {}

    print(
        f"  Loaded {len(df):,} videos | "
        f"reference set: {len(ref_video_ids):,} High/Viral titles"
    )

    print("  Precomputing TF-IDF similarities to reference set...", flush=True)
    video_ref_sims = precompute_video_ref_similarities(
        df, tfidf, ref_video_ids, ref_vectors
    )

    pair_rows: list[dict] = []
    channels = df.groupby("channel_id")

    for channel_id, channel_df in channels:
        pairs = generate_channel_pairs(channel_df)
        if not pairs:
            continue

        for row_a, row_b in pairs:
            exclude = {row_a["video_id"], row_b["video_id"]}
            sim_a = avg_similarity_excluding(
                video_ref_sims[row_a["video_id"]],
                ref_video_ids,
                exclude,
            )
            sim_b = avg_similarity_excluding(
                video_ref_sims[row_b["video_id"]],
                ref_video_ids,
                exclude,
            )

            views_a = int(row_a["view_count"])
            views_b = int(row_b["view_count"])
            actual_winner_is_b = views_b > views_a
            predicted_winner_is_b = sim_b > sim_a
            predicted_correct = actual_winner_is_b == predicted_winner_is_b

            if sim_a == sim_b:
                predicted_correct = False

            pair_rows.append(
                {
                    "niche": niche,
                    "channel_id": channel_id,
                    "title_a": row_a["title"],
                    "title_b": row_b["title"],
                    "similarity_a": round(sim_a, 4),
                    "similarity_b": round(sim_b, 4),
                    "views_a": views_a,
                    "views_b": views_b,
                    "predicted_correct": predicted_correct,
                }
            )

        print(
            f"  channel {channel_id}: "
            f"{len(pairs):,} pairs evaluated "
            f"(running total {len(pair_rows):,})",
            flush=True,
        )

    if not pair_rows:
        print("  No valid pairs found for this niche")
        return [], {
            "niche": niche,
            "num_pairs": 0,
            "accuracy_pct": None,
        }

    correct = sum(1 for r in pair_rows if r["predicted_correct"])
    accuracy = correct / len(pair_rows)
    print(
        f"  Done — {len(pair_rows):,} pairs | "
        f"accuracy {accuracy * 100:.1f}% "
        f"(baseline {RANDOM_BASELINE * 100:.0f}%)"
    )

    summary = {
        "niche": niche,
        "num_pairs": len(pair_rows),
        "accuracy_pct": round(accuracy * 100, 2),
    }
    return pair_rows, summary


def print_summary_table(summaries: list[dict], overall: dict):
    print(f"\n{'=' * 62}")
    print("TITLE COMPARISON EVAL — SUMMARY")
    print(f"{'=' * 62}")
    print(f"{'Niche':<15} {'Pairs':>8} {'Accuracy %':>12} {'vs 50%':>10}")
    print(f"{'-' * 15} {'-' * 8} {'-' * 12} {'-' * 10}")

    for row in summaries:
        if row["num_pairs"] == 0:
            print(f"{row['niche']:<15} {0:>8} {'—':>12} {'—':>10}")
            continue
        delta = row["accuracy_pct"] - RANDOM_BASELINE * 100
        sign = "+" if delta >= 0 else ""
        print(
            f"{row['niche']:<15} "
            f"{row['num_pairs']:>8,} "
            f"{row['accuracy_pct']:>11.1f}% "
            f"{sign}{delta:>8.1f}pp"
        )

    print(f"{'-' * 15} {'-' * 8} {'-' * 12} {'-' * 10}")
    if overall["num_pairs"] > 0:
        delta = overall["accuracy_pct"] - RANDOM_BASELINE * 100
        sign = "+" if delta >= 0 else ""
        print(
            f"{'OVERALL':<15} "
            f"{overall['num_pairs']:>8,} "
            f"{overall['accuracy_pct']:>11.1f}% "
            f"{sign}{delta:>8.1f}pp"
        )
    else:
        print(f"{'OVERALL':<15} {0:>8} {'—':>12} {'—':>10}")


def main():
    os.chdir(PROJECT_ROOT)

    print("=" * 62)
    print("TUBEIQ TITLE COMPARISON EVAL")
    print("=" * 62)
    print(
        "Task: predict higher-view video via avg TF-IDF similarity "
        "to High/Viral reference titles"
    )
    print(f"Pair rules: same channel, <= {MAX_DAYS_APART} days apart, "
          f">= {MIN_VIEW_RATIO:.0f}x view ratio")
    print(f"Random baseline: {RANDOM_BASELINE * 100:.0f}%")

    all_rows: list[dict] = []
    summaries: list[dict] = []

    for niche in NICHES:
        rows, summary = evaluate_niche(niche)
        all_rows.extend(rows)
        if summary:
            summaries.append(summary)

    if not all_rows:
        print("\nNo pairs evaluated across any niche.")
        sys.exit(1)

    results_df = pd.DataFrame(all_rows)
    csv_path = PROJECT_ROOT / "title_comparison_results.csv"
    results_df.to_csv(csv_path, index=False)

    total_pairs = len(all_rows)
    total_correct = int(results_df["predicted_correct"].sum())
    overall = {
        "niche": "overall",
        "num_pairs": total_pairs,
        "accuracy_pct": round(total_correct / total_pairs * 100, 2),
    }

    print_summary_table(summaries, overall)
    print(f"\nSaved pair-level results: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
