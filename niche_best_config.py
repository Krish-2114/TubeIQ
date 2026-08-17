"""
Per-niche model selection on an 80/20 split (no LOCO).

For each niche independently: build the video pairs, try every feature set and
classifier, and report the best combination for that niche.

Two 80/20 splits are reported for every config:
    pair  — random over pairs. A video appears in many pairs, so the same video
            lands in train AND test. Optimistic.
    video — videos split 80/20 first, pairs kept only when both videos are on
            the same side. Same 80/20 idea, no shared videos.

age_days is excluded from every feature set: view_velocity is views/age_days,
so the younger video in a pair mechanically tends to win, and age is not
available at prediction time anyway.

    python niche_best_config.py                    # all niches
    python niche_best_config.py cooking education  # just these two
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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

# age-free feature sets only
CANDIDATE_SETS = ["base4", "title_all", "title_meta"]
NEEDS_SCALING = {"logreg"}


def make_model(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=SEED)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=150, min_samples_leaf=20,
            random_state=SEED, n_jobs=-1,
        )
    if name == "hist_gb":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=SEED
        )
    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        eval_metric="logloss", random_state=SEED, n_jobs=-1,
    )


MODEL_NAMES = ["logreg", "random_forest", "hist_gb", "xgboost"]


def evaluate(X, y, tr, te, model_name):
    X_tr, X_te = X[tr], X[te]
    if model_name in NEEDS_SCALING:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
    model = make_model(model_name)
    model.fit(X_tr, y[tr])
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs > 0.5).astype(int)
    acc = float((preds == y[te]).mean())
    auc = (
        float(roc_auc_score(y[te], probs))
        if len(np.unique(y[te])) > 1
        else None
    )
    return acc, auc


def run_niche(niche: str) -> dict | None:
    t0 = time.time()
    df = load_and_filter(niche)
    if df is None or df.empty:
        print(f"  {niche}: no usable rows")
        return None
    tfidf = fit_tfidf(df)
    if tfidf is None:
        print(f"  {niche}: too few titles")
        return None
    per_video, ref_ids = build_sim_index(df, tfidf)
    if per_video is None:
        print(f"  {niche}: no High/Viral reference titles")
        return None
    pairs = build_pairs(df, np.random.default_rng(SEED))
    if len(pairs) < MIN_BLEND_PAIRS:
        print(f"  {niche}: only {len(pairs)} pairs")
        return None

    X_all, y_all = build_matrix(pairs, per_video, ref_ids)
    splits = {p: split_indices(pairs, y_all, p) for p in ("pair", "video")}

    rows = []
    for fs_name in CANDIDATE_SETS:
        cols = FEATURE_SETS[fs_name]
        Xf = X_all[cols].to_numpy(dtype=float)
        for model_name in MODEL_NAMES:
            rec = {"features": fs_name, "model": model_name}
            for protocol, (tr, te) in splits.items():
                if len(te) < 30 or len(np.unique(y_all[tr])) < 2:
                    rec[f"{protocol}_acc"] = None
                    rec[f"{protocol}_auc"] = None
                    continue
                acc, auc = evaluate(Xf, y_all, tr, te, model_name)
                rec[f"{protocol}_acc"] = acc
                rec[f"{protocol}_auc"] = auc
            rows.append(rec)

    return {
        "niche": niche,
        "n_channels": int(df["channel_id"].nunique()),
        "n_videos": len(df),
        "n_pairs": len(pairs),
        "n_test_pair": len(splits["pair"][1]),
        "n_test_video": len(splits["video"][1]),
        "rows": rows,
        "secs": time.time() - t0,
    }


def report(result: dict):
    print(f"\n{'=' * 80}")
    print(f"NICHE: {result['niche'].upper()}")
    print(f"{'=' * 80}")
    print(
        f"  channels={result['n_channels']}  videos={result['n_videos']:,}  "
        f"pairs={result['n_pairs']:,}  "
        f"test: pair={result['n_test_pair']:,} video={result['n_test_video']:,}"
    )
    print(
        f"\n  {'Features':<12} {'Model':<15} | "
        f"{'pairAcc':>8} {'pairAUC':>8} | {'vidAcc':>8} {'vidAUC':>8}"
    )
    print(f"  {'-' * 12} {'-' * 15} -{'-' * 18} -{'-' * 18}")

    usable = [r for r in result["rows"] if r.get("video_auc") is not None]
    ranked = sorted(usable, key=lambda r: -r["video_auc"])
    for r in ranked:
        print(
            f"  {r['features']:<12} {r['model']:<15} | "
            f"{r['pair_acc'] * 100:7.1f}% {r['pair_auc']:8.3f} | "
            f"{r['video_acc'] * 100:7.1f}% {r['video_auc']:8.3f}"
        )

    if ranked:
        best = ranked[0]
        base = next(
            (
                r for r in result["rows"]
                if r["features"] == "base4" and r["model"] == "logreg"
            ),
            None,
        )
        print(
            f"\n  BEST: {best['features']} + {best['model']} = "
            f"{best['video_acc'] * 100:.1f}% acc / {best['video_auc']:.3f} AUC"
        )
        if base and base.get("video_auc") is not None:
            print(
                f"  Production (base4 + logreg): "
                f"{base['video_acc'] * 100:.1f}% acc / "
                f"{base['video_auc']:.3f} AUC "
                f"({(best['video_auc'] - base['video_auc']):+.3f} AUC)"
            )
    print(f"  ({result['secs']:.0f}s)", flush=True)


def main():
    os.chdir(PROJECT_ROOT)
    requested = [n for n in sys.argv[1:] if not n.startswith("-")]
    unknown = [n for n in requested if n not in NICHES]
    if unknown:
        print(f"Unknown niche(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(NICHES)}")
        sys.exit(1)
    targets = requested or list(NICHES)

    print("=" * 80)
    print("PER-NICHE MODEL SELECTION — 80/20 SPLIT")
    print("=" * 80)
    print(f"Niches: {', '.join(targets)}")
    print(f"Feature sets: {', '.join(CANDIDATE_SETS)} (age_days excluded)")
    print(f"Models: {', '.join(MODEL_NAMES)}")

    finished = []
    for niche in targets:
        result = run_niche(niche)
        if result:
            report(result)
            finished.append(result)

    if len(finished) < 2:
        return

    print(f"\n{'=' * 80}")
    print("BEST CONFIG PER NICHE")
    print(f"{'=' * 80}")
    print(
        f"{'Niche':<15} {'Ch':>3} {'Pairs':>8} {'Best features':<12} "
        f"{'Best model':<15} {'Acc %':>7} {'AUC':>7} {'vs prod':>8}"
    )
    print(
        f"{'-' * 15} {'-' * 3} {'-' * 8} {'-' * 12} "
        f"{'-' * 15} {'-' * 7} {'-' * 7} {'-' * 8}"
    )
    for result in finished:
        usable = [r for r in result["rows"] if r.get("video_auc") is not None]
        if not usable:
            continue
        best = max(usable, key=lambda r: r["video_auc"])
        base = next(
            (
                r for r in result["rows"]
                if r["features"] == "base4" and r["model"] == "logreg"
            ),
            None,
        )
        delta = (
            f"{best['video_auc'] - base['video_auc']:+.3f}"
            if base and base.get("video_auc") is not None
            else "-"
        )
        print(
            f"{result['niche']:<15} {result['n_channels']:>3} "
            f"{result['n_pairs']:>8,} {best['features']:<12} "
            f"{best['model']:<15} {best['video_acc'] * 100:6.1f}% "
            f"{best['video_auc']:7.3f} {delta:>8}"
        )

    print(
        "\nQuote the video-disjoint numbers. The pair-split column is higher "
        "only because\nthe same video appears on both sides of that split."
    )


if __name__ == "__main__":
    main()
