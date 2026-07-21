"""
LOCO evaluation of blended TF-IDF + structural title features.

Fits a logistic regression on pair-wise feature diffs:
  tfidf_diff, has_number_diff, has_question_diff, title_length_diff,
  [duration_seconds_diff]

Age-gap restricted (abs(age_days_a - age_days_b) <= 120) to block the
cross-era structural leakage confound. Includes a duration-control ablation:
full 5-feature blend vs original 4 features on the same pairs.

Run from the tubeiq project root:
    python blended_score_eval.py

NOTE: This file was recreated — it was referenced but not present in the repo.
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from backend.ml.train_models import NICHES, clean_title

PROJECT_ROOT = Path(__file__).resolve().parent
MIN_VIEW_RATIO = 2.0
MAX_AGE_GAP_DAYS = 120
RANDOM_BASELINE = 0.5
DATA_DIR = PROJECT_ROOT / "backend" / "data"

FEATURE_COLS_BASE = [
    "tfidf_diff",
    "has_number_diff",
    "has_question_diff",
    "title_length_diff",
]
FEATURE_COLS_FULL = FEATURE_COLS_BASE + ["duration_seconds_diff"]


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
        "has_number",
        "has_question",
        "title_length",
        "duration_seconds",
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
    return df


def generate_velocity_pairs(channel_df: pd.DataFrame) -> list[tuple]:
    """Same-channel pairs: >=2x velocity ratio AND age_days gap <= 120."""
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


def build_reference_vectors(train_df: pd.DataFrame, tfidf: TfidfVectorizer):
    high_df = train_df[
        train_df["performance_label"].isin(["High", "Viral"])
    ]
    titles = high_df["title"].fillna("").tolist()
    if not titles:
        return None
    return tfidf.transform(titles)


def mean_sim_to_ref(titles: list[str], tfidf, ref_vectors) -> np.ndarray:
    if not titles:
        return np.array([])
    vecs = tfidf.transform(titles)
    return cosine_similarity(vecs, ref_vectors).mean(axis=1)


def pair_feature_row(
    row_a: dict,
    row_b: dict,
    sim_a: float,
    sim_b: float,
    include_duration: bool,
) -> tuple[list[float], int]:
    feats = [
        float(sim_b) - float(sim_a),
        float(row_b["has_number"]) - float(row_a["has_number"]),
        float(row_b["has_question"]) - float(row_a["has_question"]),
        float(row_b["title_length"]) - float(row_a["title_length"]),
    ]
    if include_duration:
        feats.append(
            float(row_b["duration_seconds"]) - float(row_a["duration_seconds"])
        )
    label = (
        1
        if float(row_b["view_velocity"]) > float(row_a["view_velocity"])
        else 0
    )
    return feats, label


def build_pair_matrix(
    pairs: list[tuple],
    sim_lookup: dict,
    include_duration: bool,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for row_a, row_b in pairs:
        # Key by position identity — use video_id if present else title+published proxy
        key_a = row_a.get("video_id", id(row_a))
        key_b = row_b.get("video_id", id(row_b))
        # Fall back: sims keyed during precompute by video_id or index
        sim_a = sim_lookup[key_a]
        sim_b = sim_lookup[key_b]
        feats, label = pair_feature_row(
            row_a, row_b, sim_a, sim_b, include_duration
        )
        X.append(feats)
        y.append(label)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def precompute_sims(
    df: pd.DataFrame,
    tfidf: TfidfVectorizer,
    ref_vectors,
) -> dict:
    titles = df["title"].fillna("").tolist()
    sims = mean_sim_to_ref(titles, tfidf, ref_vectors)
    if "video_id" in df.columns:
        keys = df["video_id"].tolist()
    else:
        keys = list(range(len(df)))
        df = df.copy()
        df["_tmp_key"] = keys
    return {keys[i]: float(sims[i]) for i in range(len(keys))}


def pure_tfidf_score_pairs(
    pairs: list[tuple],
    sim_lookup: dict,
) -> tuple[int, int]:
    """Mean-sim comparison (no LR) — matches unseen_channel_eval scoring."""
    if not pairs:
        return 0, 0
    correct = 0
    for row_a, row_b in pairs:
        key_a = row_a.get("video_id", id(row_a))
        key_b = row_b.get("video_id", id(row_b))
        sim_a = sim_lookup[key_a]
        sim_b = sim_lookup[key_b]
        if sim_a == sim_b:
            continue
        predicted_b = sim_b > sim_a
        actual_b = float(row_b["view_velocity"]) > float(row_a["view_velocity"])
        if predicted_b == actual_b:
            correct += 1
    return correct, len(pairs)


def fit_and_score_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[int, int]:
    if len(X_train) < 10 or len(np.unique(y_train)) < 2:
        return 0, 0
    if len(X_test) == 0:
        return 0, 0

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(X_train_s, y_train)

    # decision_function == 0 is a tie → incorrect
    scores = model.decision_function(X_test_s)
    preds = (scores > 0).astype(int)
    ties = scores == 0
    correct = int(np.sum((preds == y_test) & ~ties))
    return correct, len(y_test)


def evaluate_niche(niche: str) -> dict | None:
    print(f"\n{'=' * 72}")
    print(f"NICHE: {niche.upper()}")
    print(f"{'=' * 72}")

    df = load_and_filter(niche)
    if df is None or df.empty:
        print("  Skipping — no usable rows")
        return None

    if "video_id" not in df.columns:
        df = df.copy()
        df["video_id"] = df.index.astype(str)

    channel_ids = sorted(df["channel_id"].dropna().unique().tolist())
    n_channels = len(channel_ids)
    print(f"  Videos: {len(df):,} | Channels: {n_channels}")
    if n_channels < 2:
        print("  Skipping — need >= 2 channels for LOCO")
        return None

    totals = {
        "pure_correct": 0,
        "pure_pairs": 0,
        "full_correct": 0,
        "full_pairs": 0,
        "ablate_correct": 0,
        "ablate_pairs": 0,
    }

    for held_out_id in channel_ids:
        train_df = df[df["channel_id"] != held_out_id].copy()
        test_df = df[df["channel_id"] == held_out_id].copy()
        title = ""
        if not test_df.empty:
            title = str(test_df["channel_title"].iloc[0])

        print(
            f"\n  Fold held-out={held_out_id}"
            f"{f' ({title})' if title else ''} "
            f"| train={len(train_df):,} test={len(test_df):,}",
            flush=True,
        )

        tfidf = fit_tfidf(train_df)
        if tfidf is None:
            print("    Skip — <10 train titles")
            continue
        ref = build_reference_vectors(train_df, tfidf)
        if ref is None or ref.shape[0] == 0:
            print("    Skip — empty reference set")
            continue

        # Pairs from every train channel + held-out channel
        train_pairs: list[tuple] = []
        for _, ch_df in train_df.groupby("channel_id"):
            train_pairs.extend(generate_velocity_pairs(ch_df))
        test_pairs = generate_velocity_pairs(test_df)

        if not test_pairs:
            print("    No valid age-gap pairs in held-out channel")
            continue

        sim_lookup = precompute_sims(df, tfidf, ref)

        # Pure TF-IDF on same test pairs
        p_corr, p_n = pure_tfidf_score_pairs(test_pairs, sim_lookup)
        totals["pure_correct"] += p_corr
        totals["pure_pairs"] += p_n

        if not train_pairs:
            print("    Skip LR — no train pairs")
            continue

        X_tr_full, y_tr = build_pair_matrix(
            train_pairs, sim_lookup, include_duration=True
        )
        X_te_full, y_te = build_pair_matrix(
            test_pairs, sim_lookup, include_duration=True
        )
        X_tr_abl, _ = build_pair_matrix(
            train_pairs, sim_lookup, include_duration=False
        )
        X_te_abl, _ = build_pair_matrix(
            test_pairs, sim_lookup, include_duration=False
        )

        f_corr, f_n = fit_and_score_lr(X_tr_full, y_tr, X_te_full, y_te)
        a_corr, a_n = fit_and_score_lr(X_tr_abl, y_tr, X_te_abl, y_te)

        totals["full_correct"] += f_corr
        totals["full_pairs"] += f_n
        totals["ablate_correct"] += a_corr
        totals["ablate_pairs"] += a_n

        def pct(c, n):
            return f"{c / n * 100:.1f}%" if n else "—"

        print(
            f"    test_pairs={p_n:,} | "
            f"pure_tfidf={pct(p_corr, p_n)} | "
            f"blend+dur={pct(f_corr, f_n)} | "
            f"blend-dur={pct(a_corr, a_n)}",
            flush=True,
        )

    def acc(c, n):
        return c / n if n else None

    result = {
        "niche": niche,
        "n_channels": n_channels,
        "pure_pairs": totals["pure_pairs"],
        "pure_acc": acc(totals["pure_correct"], totals["pure_pairs"]),
        "full_pairs": totals["full_pairs"],
        "full_acc": acc(totals["full_correct"], totals["full_pairs"]),
        "ablate_pairs": totals["ablate_pairs"],
        "ablate_acc": acc(totals["ablate_correct"], totals["ablate_pairs"]),
        "pure_correct": totals["pure_correct"],
        "full_correct": totals["full_correct"],
        "ablate_correct": totals["ablate_correct"],
    }

    print(f"\n  Niche totals:")
    for label, akey, pkey in [
        ("pure TF-IDF", "pure_acc", "pure_pairs"),
        ("blend + duration", "full_acc", "full_pairs"),
        ("blend − duration", "ablate_acc", "ablate_pairs"),
    ]:
        a = result[akey]
        n = result[pkey]
        if a is None:
            print(f"    {label}: no pairs")
        else:
            print(f"    {label}: {a * 100:.1f}% on {n:,} pairs")

    return result


def print_summary(summaries: list[dict]):
    print(f"\n{'=' * 88}")
    print("BLENDED SCORE EVAL — SUMMARY (age-gap <= 120)")
    print(f"{'=' * 88}")
    print(
        f"{'Niche':<15} {'Ch':>4} {'Pairs':>8} "
        f"{'Pure TF%':>9} {'+Dur %':>9} {'−Dur %':>9} "
        f"{'+Dur vs TF':>11}"
    )
    print(
        f"{'-' * 15} {'-' * 4} {'-' * 8} "
        f"{'-' * 9} {'-' * 9} {'-' * 9} {'-' * 11}"
    )

    pc = pp = fc = fp = ac = ap = 0
    for row in summaries:
        pc += row["pure_correct"]
        pp += row["pure_pairs"]
        fc += row["full_correct"]
        fp += row["full_pairs"]
        ac += row["ablate_correct"]
        ap += row["ablate_pairs"]

        def fmt(a):
            return f"{a * 100:8.1f}%" if a is not None else f"{'—':>9}"

        delta = ""
        if row["full_acc"] is not None and row["pure_acc"] is not None:
            d = (row["full_acc"] - row["pure_acc"]) * 100
            sign = "+" if d >= 0 else ""
            delta = f"{sign}{d:.1f}pp"
        print(
            f"{row['niche']:<15} {row['n_channels']:>4} "
            f"{row['pure_pairs']:>8,} "
            f"{fmt(row['pure_acc'])} "
            f"{fmt(row['full_acc'])} "
            f"{fmt(row['ablate_acc'])} "
            f"{delta:>11}"
        )

    print(
        f"{'-' * 15} {'-' * 4} {'-' * 8} "
        f"{'-' * 9} {'-' * 9} {'-' * 9} {'-' * 11}"
    )
    pure = pc / pp if pp else None
    full = fc / fp if fp else None
    ablate = ac / ap if ap else None

    def fmt(a):
        return f"{a * 100:8.1f}%" if a is not None else f"{'—':>9}"

    delta = ""
    if full is not None and pure is not None:
        d = (full - pure) * 100
        sign = "+" if d >= 0 else ""
        delta = f"{sign}{d:.1f}pp"
    print(
        f"{'OVERALL':<15} {'':>4} {pp:>8,} "
        f"{fmt(pure)} {fmt(full)} {fmt(ablate)} {delta:>11}"
    )

    print(f"\nRandom baseline: {RANDOM_BASELINE * 100:.0f}%")
    if full is not None:
        niches_clear = [
            s["niche"]
            for s in summaries
            if s["full_acc"] is not None and s["full_acc"] > RANDOM_BASELINE
        ]
        niches_fail = [
            s["niche"]
            for s in summaries
            if s["full_acc"] is not None and s["full_acc"] <= RANDOM_BASELINE
        ]
        print(
            f"Duration-controlled blend niches >50%: "
            f"{', '.join(niches_clear) or 'none'}"
        )
        print(
            f"Duration-controlled blend niches <=50%: "
            f"{', '.join(niches_fail) or 'none'}"
        )
        if pure is not None:
            print(
                f"Overall lift (blend+dur − pure TF-IDF): "
                f"{(full - pure) * 100:+.2f}pp"
            )
            if ablate is not None:
                print(
                    f"Duration ablation delta (+dur − −dur): "
                    f"{(full - ablate) * 100:+.2f}pp "
                    f"(positive ⇒ duration feature helps the blend)"
                )


def main():
    os.chdir(PROJECT_ROOT)
    print("=" * 88)
    print("TUBEIQ BLENDED SCORE EVAL (LOCO)")
    print("=" * 88)
    print(
        "Features: tfidf_diff + has_number/has_question/title_length diffs "
        "[+ duration_seconds_diff]"
    )
    print(
        f"Pair rules: same channel, "
        f">= {MIN_VIEW_RATIO:.0f}x view_velocity, "
        f"age_days gap <= {MAX_AGE_GAP_DAYS}"
    )
    print(
        "Ablation: full 5-feature blend vs 4-feature (no duration) "
        "on identical pairs"
    )

    summaries: list[dict] = []
    for niche in NICHES:
        result = evaluate_niche(niche)
        if result is not None:
            summaries.append(result)

    if not summaries or all(s["pure_pairs"] == 0 for s in summaries):
        print("\nNo pairs evaluated.")
        sys.exit(1)

    print_summary(summaries)
    print("\nDone.")


if __name__ == "__main__":
    main()
