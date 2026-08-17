"""
Per-niche test: do the extra title features help on an unseen channel?

Each niche is evaluated on its own, holding out one channel at a time so every
channel is unseen exactly once. Unlike feature_model_sweep.py, TF-IDF and the
High/Viral reference set are fit on TRAINING CHANNELS ONLY, so a held-out
channel's own winning titles never inform its score.

Compares three configs on identical pairs:
    base4      + logreg         (what production ships)
    title_all  + logreg         (more features, same model)
    title_all  + random_forest  (more features, stronger model)

    python niche_feature_test.py                  # all niches
    python niche_feature_test.py cooking education   # just these two
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from backend.ml.train_models import (
    MAX_AGE_GAP_DAYS,
    MIN_VIEW_RATIO,
    NICHES,
    clean_title,
)
from feature_model_sweep import (
    FEATURE_SETS,
    TITLE_FLAGS,
    TITLE_SHAPE,
    VIDEO_META,
    load_and_filter,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
SEED = 42

CONFIGS = [
    ("base4 + logreg", "base4", "logreg"),
    ("title_all + logreg", "title_all", "logreg"),
    ("title_all + forest", "title_all", "random_forest"),
]


def make_model(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=SEED)
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=10, random_state=SEED, n_jobs=-1
    )


def fit_text_on_train(train_df: pd.DataFrame):
    """Vectorizer and reference set built from training channels only."""
    cleaned = train_df.apply(
        lambda r: clean_title(r.get("title", ""), r.get("channel_title", "")),
        axis=1,
    ).fillna("").tolist()
    if len(cleaned) < 10:
        return None, None, None
    tfidf = TfidfVectorizer(
        max_features=500, ngram_range=(1, 2), stop_words="english", min_df=2
    )
    tfidf.fit(cleaned)

    high = train_df[train_df["performance_label"].isin(["High", "Viral"])]
    high = high[high["title"].str.strip() != ""]
    if high.empty:
        return None, None, None
    return tfidf, tfidf.transform(high["title"].tolist()), high["video_id"].tolist()


def sims_for(df: pd.DataFrame, tfidf, ref_vectors) -> dict:
    matrix = cosine_similarity(
        tfidf.transform(df["title"].tolist()), ref_vectors
    )
    return {vid: matrix[i] for i, vid in enumerate(df["video_id"].tolist())}


def pairs_for(df: pd.DataFrame, rng: np.random.Generator) -> list[tuple]:
    pairs = []
    for _, cdf in df.groupby("channel_id"):
        for a, b in combinations(cdf.to_dict("records"), 2):
            if abs(float(a["age_days"]) - float(b["age_days"])) > MAX_AGE_GAP_DAYS:
                continue
            va, vb = float(a["view_velocity"]), float(b["view_velocity"])
            if va == vb:
                continue
            lo, hi = min(va, vb), max(va, vb)
            if lo <= 0 or hi / lo < MIN_VIEW_RATIO:
                continue
            pairs.append((b, a) if rng.random() < 0.5 else (a, b))
    return pairs


def matrix_for(pairs, sim_lookup, ref_video_ids) -> tuple[pd.DataFrame, np.ndarray]:
    diff_cols = TITLE_FLAGS + TITLE_SHAPE + VIDEO_META + ["age_days"]
    ref_set = set(ref_video_ids)
    rows, labels = [], []
    for a, b in pairs:
        exclude = {a["video_id"], b["video_id"]}
        keep = [i for i, v in enumerate(ref_video_ids) if v not in exclude]
        sa = sim_lookup[a["video_id"]]
        sb = sim_lookup[b["video_id"]]
        # Only train videos can sit in the reference set; skip the mask when
        # neither side is there.
        if exclude & ref_set:
            sim_a = float(np.mean(sa[keep])) if keep else 0.0
            sim_b = float(np.mean(sb[keep])) if keep else 0.0
        else:
            sim_a, sim_b = float(sa.mean()), float(sb.mean())
        rec = {"tfidf_diff": sim_b - sim_a}
        for col in diff_cols:
            rec[f"{col}_diff"] = float(b[col]) - float(a[col])
        rows.append(rec)
        labels.append(
            1 if float(b["view_velocity"]) > float(a["view_velocity"]) else 0
        )
    return pd.DataFrame(rows), np.asarray(labels, dtype=int)


def main():
    os.chdir(PROJECT_ROOT)
    requested = [n for n in sys.argv[1:] if not n.startswith("-")]
    unknown = [n for n in requested if n not in NICHES]
    if unknown:
        print(f"Unknown niche(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(NICHES)}")
        sys.exit(1)
    targets = requested or list(NICHES)

    print("=" * 84)
    print("PER-NICHE TEST — LEAVE ONE CHANNEL OUT, TF-IDF FIT ON TRAIN ONLY")
    print("=" * 84)
    print(f"Niches: {', '.join(targets)}")

    per_niche = {}
    for niche in targets:
        t0 = time.time()
        df = load_and_filter(niche)
        if df is None or df.empty:
            continue
        channels = sorted(df["channel_id"].unique().tolist())
        if len(channels) < 2:
            continue

        acc = {label: {"y": [], "p": [], "correct": 0, "n": 0} for label, _, _ in CONFIGS}

        for held in channels:
            train_df = df[df["channel_id"] != held]
            test_df = df[df["channel_id"] == held]
            tfidf, ref_vectors, ref_ids = fit_text_on_train(train_df)
            if tfidf is None:
                continue

            rng = np.random.default_rng(SEED)
            train_pairs = pairs_for(train_df, rng)
            test_pairs = pairs_for(test_df, rng)
            if not train_pairs or len(test_pairs) < 20:
                continue

            sim_lookup = sims_for(df, tfidf, ref_vectors)
            X_tr, y_tr = matrix_for(train_pairs, sim_lookup, ref_ids)
            X_te, y_te = matrix_for(test_pairs, sim_lookup, ref_ids)
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            for label, fs_name, model_name in CONFIGS:
                cols = FEATURE_SETS[fs_name]
                a_tr = X_tr[cols].to_numpy(dtype=float)
                a_te = X_te[cols].to_numpy(dtype=float)
                if model_name == "logreg":
                    scaler = StandardScaler()
                    a_tr = scaler.fit_transform(a_tr)
                    a_te = scaler.transform(a_te)
                model = make_model(model_name)
                model.fit(a_tr, y_tr)
                probs = model.predict_proba(a_te)[:, 1]
                preds = (probs > 0.5).astype(int)
                slot = acc[label]
                slot["y"].append(y_te)
                slot["p"].append(probs)
                slot["correct"] += int((preds == y_te).sum())
                slot["n"] += len(y_te)

        results = {}
        for label, slot in acc.items():
            if not slot["n"]:
                continue
            y = np.concatenate(slot["y"])
            p = np.concatenate(slot["p"])
            results[label] = {
                "acc": slot["correct"] / slot["n"],
                "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
                "n": slot["n"],
            }
        if results:
            per_niche[niche] = {"n_channels": len(channels), "r": results}
        print(f"  {niche:<15} channels={len(channels):>2} ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{'=' * 84}")
    print("PER-NICHE RESULTS (unseen channel, accuracy % / ROC-AUC)")
    print(f"{'=' * 84}")
    header = f"{'Niche':<15} {'Ch':>3} {'Pairs':>8}"
    for label, _, _ in CONFIGS:
        header += f" | {label:>20}"
    print(header)
    print("-" * len(header))

    for niche, blob in per_niche.items():
        first = next(iter(blob["r"].values()))
        line = f"{niche:<15} {blob['n_channels']:>3} {first['n']:>8,}"
        for label, _, _ in CONFIGS:
            r = blob["r"].get(label)
            if r is None:
                line += f" | {'-':>20}"
            else:
                auc = f"{r['auc']:.3f}" if r["auc"] is not None else "  -  "
                line += f" | {r['acc'] * 100:12.1f}% {auc:>7}"
        print(line)

    print("-" * len(header))
    print("\nBest config per niche:")
    for niche, blob in per_niche.items():
        ranked = sorted(
            (
                (label, r)
                for label, r in blob["r"].items()
                if r["auc"] is not None
            ),
            key=lambda kv: -kv[1]["auc"],
        )
        if not ranked:
            continue
        top_label, top = ranked[0]
        base = blob["r"].get("base4 + logreg")
        delta = (
            f" ({top['auc'] - base['auc']:+.3f} AUC vs production)"
            if base and base["auc"] is not None
            else ""
        )
        print(f"  {niche:<15} {top_label:<22} {top['auc']:.3f}{delta}")

    print(
        "\nAUC below 0.500 means the model ranks worse than random on that "
        "niche's\nunseen channels — the direction it learned does not transfer."
    )


if __name__ == "__main__":
    main()
