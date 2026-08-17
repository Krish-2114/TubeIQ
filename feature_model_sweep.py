"""
Experiment harness: does the blend model improve with more features or a
different classifier?

Holds the evaluation protocol from blend_eval.py fixed (same pairs, same
80/20 split, same seed, balanced orientation) and varies two things:
  1. the pairwise feature set
  2. the classifier

Only pre-publication signals are eligible as features. Columns that describe
what happened AFTER upload — view_count, like_count, comment_count,
engagement_rate, performance_ratio, performance_label, channel_median_views,
view_velocity — are excluded, since view_velocity defines the label and the
rest are proxies for it.

Run from the tubeiq project root:
    python feature_model_sweep.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.ml.train_models import (
    MAX_AGE_GAP_DAYS,
    MIN_BLEND_PAIRS,
    MIN_VIEW_RATIO,
    NICHES,
    _avg_sim_excluding,
    clean_title,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "backend" / "data"
TEST_FRACTION = 0.2
SEED = 42

TITLE_FLAGS = [
    "has_number", "has_dollar", "has_rupee", "has_question",
    "has_exclamation", "has_vs", "has_challenge", "has_hours",
    "has_last", "has_win", "has_free", "has_extreme", "has_best",
    "has_worst", "has_every", "has_first", "has_only", "has_never",
    "has_days", "has_review", "has_how", "has_why", "has_what",
]
TITLE_SHAPE = ["title_length", "word_count"]
VIDEO_META = ["duration_seconds", "duration_encoded", "day_of_week", "hour_of_day"]

# Every diff column the harness can build; feature sets select from these.
ALL_DIFF_COLS = ["tfidf_diff"] + [
    f"{c}_diff" for c in TITLE_FLAGS + TITLE_SHAPE + VIDEO_META + ["age_days"]
]

FEATURE_SETS = {
    # what production ships today
    "base4": [
        "tfidf_diff", "has_number_diff", "has_question_diff",
        "title_length_diff",
    ],
    "title_all": ["tfidf_diff"] + [
        f"{c}_diff" for c in TITLE_FLAGS + TITLE_SHAPE
    ],
    "title_meta": ["tfidf_diff"] + [
        f"{c}_diff" for c in TITLE_FLAGS + TITLE_SHAPE + VIDEO_META
    ],
    "title_meta_age": ["tfidf_diff"] + [
        f"{c}_diff" for c in TITLE_FLAGS + TITLE_SHAPE + VIDEO_META
    ] + ["age_days_diff"],
}


def make_models() -> dict:
    return {
        "logreg": LogisticRegression(
            max_iter=1000, random_state=SEED, solver="lbfgs"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10,
            random_state=SEED, n_jobs=-1,
        ),
        "hist_gb": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=SEED
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="logloss", random_state=SEED, n_jobs=-1,
        ),
    }


NEEDS_SCALING = {"logreg"}


def load_and_filter(niche: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{niche}_features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    needed = [
        "title", "view_velocity", "channel_id", "age_days",
        "performance_label",
    ] + TITLE_FLAGS + TITLE_SHAPE + VIDEO_META
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
    return df.drop_duplicates(subset=["video_id"]).reset_index(drop=True)


def fit_tfidf(df: pd.DataFrame) -> TfidfVectorizer | None:
    cleaned = df.apply(
        lambda row: clean_title(
            row.get("title", ""), row.get("channel_title", "")
        ),
        axis=1,
    ).fillna("").tolist()
    if len(cleaned) < 10:
        return None
    tfidf = TfidfVectorizer(
        max_features=500, ngram_range=(1, 2),
        stop_words="english", min_df=2,
    )
    tfidf.fit(cleaned)
    return tfidf


def build_sim_index(df: pd.DataFrame, tfidf: TfidfVectorizer):
    high_df = df[df["performance_label"].isin(["High", "Viral"])]
    high_df = high_df[high_df["title"].str.strip() != ""]
    if high_df.empty:
        return None, None
    ref_vectors = tfidf.transform(high_df["title"].tolist())
    ref_video_ids = high_df["video_id"].tolist()
    sim_matrix = cosine_similarity(
        tfidf.transform(df["title"].tolist()), ref_vectors
    )
    per_video = {
        vid: sim_matrix[i] for i, vid in enumerate(df["video_id"].tolist())
    }
    return per_video, ref_video_ids


def build_pairs(df: pd.DataFrame, rng: np.random.Generator) -> list[tuple]:
    pairs = []
    for _, channel_df in df.groupby("channel_id"):
        records = channel_df.to_dict("records")
        for row_a, row_b in combinations(records, 2):
            if abs(
                float(row_a["age_days"]) - float(row_b["age_days"])
            ) > MAX_AGE_GAP_DAYS:
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
    return pairs


def split_indices(
    pairs: list[tuple],
    y: np.ndarray,
    protocol: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    pair  — random 80/20 over pairs. A given video appears in hundreds of pairs,
            so it lands in BOTH train and test. Any feature that identifies a
            video (duration, upload hour, age, title shape) then lets the model
            memorize that video's performance instead of learning about titles.
    video — split the videos 80/20 first and keep only pairs whose two videos
            are on the same side. No video is ever seen in both, so memorizing
            individual videos buys nothing.
    channel — split whole channels. Also blocks memorizing a channel's style,
            which is what you need before claiming the model works on a
            creator it has never seen.
    """
    if protocol == "pair":
        return train_test_split(
            np.arange(len(y)),
            test_size=TEST_FRACTION,
            random_state=SEED,
            stratify=y,
        )

    key = "video_id" if protocol == "video" else "channel_id"
    units = sorted({r[side][key] for r in pairs for side in (0, 1)})
    rng = np.random.default_rng(SEED)
    shuffled = np.array(units, dtype=object)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * TEST_FRACTION)))
    test_units = set(shuffled[:n_test].tolist())

    train_idx, test_idx = [], []
    for i, (row_a, row_b) in enumerate(pairs):
        a_test = row_a[key] in test_units
        b_test = row_b[key] in test_units
        if a_test and b_test:
            test_idx.append(i)
        elif not a_test and not b_test:
            train_idx.append(i)
        # pairs straddling the split are dropped
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def build_matrix(
    pairs: list[tuple],
    per_video: dict,
    ref_video_ids: list,
) -> tuple[pd.DataFrame, np.ndarray]:
    """One wide frame of every candidate diff; feature sets slice columns."""
    diff_cols = TITLE_FLAGS + TITLE_SHAPE + VIDEO_META + ["age_days"]
    rows = []
    labels = []
    for row_a, row_b in pairs:
        exclude = {row_a["video_id"], row_b["video_id"]}
        sim_a = _avg_sim_excluding(
            per_video[row_a["video_id"]], ref_video_ids, exclude
        )
        sim_b = _avg_sim_excluding(
            per_video[row_b["video_id"]], ref_video_ids, exclude
        )
        rec = {"tfidf_diff": sim_b - sim_a}
        for col in diff_cols:
            rec[f"{col}_diff"] = float(row_b[col]) - float(row_a[col])
        rows.append(rec)
        labels.append(
            1
            if float(row_b["view_velocity"]) > float(row_a["view_velocity"])
            else 0
        )
    return pd.DataFrame(rows), np.asarray(labels, dtype=int)


def score_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    augment: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (hard predictions, positive-class probabilities) for the fold."""
    if augment:
        # Every diff feature is antisymmetric: swapping A and B negates x and
        # flips y. Training on both directions tells the model that directly
        # instead of making it infer the symmetry from data.
        X_train = np.vstack([X_train, -X_train])
        y_train = np.concatenate([y_train, 1 - y_train])

    if model_name in NEEDS_SCALING:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = make_models()[model_name]
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    return preds, probs


def main():
    os.chdir(PROJECT_ROOT)
    print("=" * 88)
    print("TUBEIQ FEATURE / MODEL SWEEP")
    print("=" * 88)
    print(
        "Protocol fixed: same pairs, random 80/20, balanced orientation, "
        f"seed={SEED}"
    )
    print(f"Feature sets: {', '.join(FEATURE_SETS)}")
    print(f"Models: {', '.join(make_models())}")

    # (protocol, features, model, augment) -> pooled test labels / probs
    pooled: dict[tuple, dict] = {}

    for niche in NICHES:
        t0 = time.time()
        df = load_and_filter(niche)
        if df is None or df.empty:
            continue
        tfidf = fit_tfidf(df)
        if tfidf is None:
            continue
        per_video, ref_ids = build_sim_index(df, tfidf)
        if per_video is None:
            continue
        pairs = build_pairs(df, np.random.default_rng(SEED))
        if len(pairs) < MIN_BLEND_PAIRS:
            continue

        X_all, y_all = build_matrix(pairs, per_video, ref_ids)

        sizes = []
        for protocol in ("pair", "video", "channel"):
            idx_train, idx_test = split_indices(pairs, y_all, protocol)
            sizes.append(f"{protocol}={len(idx_train):,}/{len(idx_test):,}")
            if len(idx_test) < 50 or len(np.unique(y_all[idx_train])) < 2:
                continue

            for fs_name, cols in FEATURE_SETS.items():
                Xf = X_all[cols].to_numpy(dtype=float)
                for model_name in make_models():
                    for augment in (False, True):
                        preds, probs = score_config(
                            Xf[idx_train], y_all[idx_train],
                            Xf[idx_test], y_all[idx_test],
                            model_name, augment,
                        )
                        key = (protocol, fs_name, model_name, augment)
                        slot = pooled.setdefault(
                            key, {"y": [], "p": [], "correct": 0, "n": 0}
                        )
                        slot["y"].append(y_all[idx_test])
                        slot["p"].append(probs)
                        slot["correct"] += int(
                            (preds == y_all[idx_test]).sum()
                        )
                        slot["n"] += len(idx_test)

        print(
            f"  {niche:<15} pairs={len(pairs):>7,} "
            f"[{'  '.join(sizes)}] ({time.time() - t0:.0f}s)",
            flush=True,
        )

    if not pooled:
        print("No niches evaluated.")
        sys.exit(1)

    scored = {}
    for (protocol, fs_name, model_name, augment), slot in pooled.items():
        y = np.concatenate(slot["y"])
        p = np.concatenate(slot["p"])
        scored[(protocol, fs_name, model_name, augment)] = {
            "acc": slot["correct"] / slot["n"],
            "auc": float(roc_auc_score(y, p)),
            "n": slot["n"],
        }

    configs = sorted(
        {(f, m, a) for (_, f, m, a) in scored},
        key=lambda c: scored.get(("video", *c), {"auc": 0})["auc"],
        reverse=True,
    )

    print(f"\n{'=' * 88}")
    print("POOLED RESULTS ACROSS ALL NICHES (sorted by video-disjoint AUC)")
    print(f"{'=' * 88}")
    print(
        f"{'Features':<16} {'Model':<15} {'Aug':>4} {'#F':>4} | "
        f"{'pairAUC':>8} | {'vidAcc':>8} {'vidAUC':>8} | "
        f"{'chanAcc':>8} {'chanAUC':>8}"
    )
    print(f"{'-' * 16} {'-' * 15} {'-' * 4} {'-' * 4} -{'-' * 9} -"
          f"{'-' * 18} -{'-' * 18}")
    for cfg in configs:
        fs_name, model_name, augment = cfg
        pair = scored.get(("pair", *cfg))
        vid = scored.get(("video", *cfg))
        chan = scored.get(("channel", *cfg))
        if pair is None or vid is None:
            continue
        chan_txt = (
            f"{chan['acc'] * 100:7.1f}% {chan['auc']:8.3f}"
            if chan else f"{'-':>8} {'-':>8}"
        )
        print(
            f"{fs_name:<16} {model_name:<15} "
            f"{'Y' if augment else 'N':>4} {len(FEATURE_SETS[fs_name]):>4} | "
            f"{pair['auc']:8.3f} | "
            f"{vid['acc'] * 100:7.1f}% {vid['auc']:8.3f} | {chan_txt}"
        )

    base = scored.get(("video", "base4", "logreg", False))
    if base and configs:
        best_cfg = configs[0]
        best = scored[("video", *best_cfg)]
        print(
            f"\nHeld-out pairs: pair-split "
            f"{scored[('pair', 'base4', 'logreg', False)]['n']:,} | "
            f"video-disjoint {base['n']:,}"
        )
        print(
            f"Production today (video-disjoint): base4 + logreg = "
            f"{base['acc'] * 100:.1f}% acc / {base['auc']:.3f} AUC"
        )
        print(
            f"Best honest config: {best_cfg[0]} + {best_cfg[1]}"
            f"{' + aug' if best_cfg[2] else ''} = "
            f"{best['acc'] * 100:.1f}% acc / {best['auc']:.3f} AUC "
            f"({(best['auc'] - base['auc']):+.3f} AUC, "
            f"{(best['acc'] - base['acc']) * 100:+.1f}pp acc)"
        )
        print(
            "\npair    — same video appears in train and test; inflated by "
            "memorization.\nvideo   — no shared videos; still shares channels."
            "\nchannel — no shared channels; this is generalization to an "
            "unseen creator."
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
