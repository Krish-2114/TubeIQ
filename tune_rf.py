"""
Tune RandomForest on title_meta (no age_days) with video-disjoint 80/20.

Reports per-niche and pooled accuracy/AUC for a small grid of RF settings,
then the best overall config.

    python tune_rf.py
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from backend.ml.train_models import MIN_BLEND_PAIRS, NICHES
from feature_model_sweep import (
    FEATURE_SETS,
    build_matrix,
    build_pairs,
    build_sim_index,
    fit_tfidf,
    load_and_filter,
    split_indices,
    SEED,
)

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent
COLS = FEATURE_SETS["title_meta"]

GRID = [
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 10, "max_features": "sqrt"},
    {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 3, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 400, "max_depth": 20, "min_samples_leaf": 2, "max_features": 0.5},
    {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 20, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 14, "min_samples_leaf": 4, "max_features": "log2"},
]


def label(params: dict) -> str:
    return (
        f"n={params['n_estimators']} d={params['max_depth']} "
        f"leaf={params['min_samples_leaf']} mf={params['max_features']}"
    )


def main():
    os.chdir(PROJECT_ROOT)
    print("=" * 88)
    print("RANDOM FOREST TUNE — title_meta, video-disjoint 80/20")
    print("=" * 88)

    niches_data = []
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
        tr, te = split_indices(pairs, y_all, "video")
        if len(te) < 30 or len(np.unique(y_all[tr])) < 2:
            continue
        X = X_all[COLS].to_numpy(dtype=float)
        niches_data.append({
            "niche": niche,
            "X_tr": X[tr], "y_tr": y_all[tr],
            "X_te": X[te], "y_te": y_all[te],
        })
        print(f"  prepared {niche} test={len(te):,} ({time.time() - t0:.0f}s)", flush=True)

    pooled = {label(p): {"y": [], "p": [], "correct": 0, "n": 0} for p in GRID}
    per_niche_best = {}

    for nd in niches_data:
        print(f"\n  {nd['niche']}:", flush=True)
        niche_rows = []
        for params in GRID:
            clf = RandomForestClassifier(
                random_state=SEED, n_jobs=-1, **params
            )
            clf.fit(nd["X_tr"], nd["y_tr"])
            probs = clf.predict_proba(nd["X_te"])[:, 1]
            preds = (probs > 0.5).astype(int)
            acc = float((preds == nd["y_te"]).mean())
            auc = float(roc_auc_score(nd["y_te"], probs))
            key = label(params)
            slot = pooled[key]
            slot["y"].append(nd["y_te"])
            slot["p"].append(probs)
            slot["correct"] += int((preds == nd["y_te"]).sum())
            slot["n"] += len(nd["y_te"])
            niche_rows.append((key, acc, auc, params))
            print(f"    {key}: {acc * 100:.1f}% / {auc:.3f}", flush=True)
        best = max(niche_rows, key=lambda r: r[2])
        per_niche_best[nd["niche"]] = best
        print(f"    BEST -> {best[0]}  {best[1] * 100:.1f}% / {best[2]:.3f}")

    print(f"\n{'=' * 88}")
    print("POOLED (video-disjoint) — sorted by AUC")
    print(f"{'=' * 88}")
    ranked = []
    for params in GRID:
        key = label(params)
        slot = pooled[key]
        y = np.concatenate(slot["y"])
        p = np.concatenate(slot["p"])
        ranked.append({
            "key": key,
            "params": params,
            "acc": slot["correct"] / slot["n"],
            "auc": float(roc_auc_score(y, p)),
            "n": slot["n"],
        })
    ranked.sort(key=lambda r: -r["auc"])
    for r in ranked:
        print(f"  {r['key']:<48} {r['acc'] * 100:6.1f}%  {r['auc']:.3f}")

    best = ranked[0]
    print(
        f"\nBEST OVERALL: {best['key']}\n"
        f"  accuracy={best['acc'] * 100:.1f}%  AUC={best['auc']:.3f}  "
        f"n_test={best['n']:,}"
    )
    print("\nPer-niche best:")
    for niche, (key, acc, auc, _) in per_niche_best.items():
        print(f"  {niche:<15} {acc * 100:5.1f}% / {auc:.3f}  ({key})")


if __name__ == "__main__":
    main()
