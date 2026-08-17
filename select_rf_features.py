"""
Find which blend features actually matter for the production Random Forest.

1. Build a large candidate pool (current title_meta + extra title shape features
   + a few new power-word / formatting candidates).
2. Fit RF per niche on a video-disjoint train split; average importances.
3. Re-evaluate cumulative top-k feature sets on the held-out videos.
4. Report the smallest set that matches/beats the full expanded pool.

age_days is never a candidate.

    python select_rf_features.py
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from backend.ml.train_models import (
    BLEND_FEATURE_NAMES,
    BLEND_RF_PARAMS,
    MIN_BLEND_PAIRS,
    NICHES,
)
from feature_model_sweep import (
    build_matrix,
    build_pairs,
    build_sim_index,
    fit_tfidf,
    load_and_filter,
    split_indices,
    SEED,
)
from rf_feature_expansion import EXTRA_FEATURE_NAMES, title_extras

PROJECT_ROOT = Path(__file__).resolve().parent

# Extra candidates beyond rf_feature_expansion.py
NEW_FEATURE_NAMES = [
    "hashtag_count",
    "all_caps_word_count",
    "space_count",
    "char_density",  # non-space chars / length
    "has_year",
    "has_power_word",
    "has_listicle_cue",  # top/best/N ways/tips
    "has_emotional",  # shocking / insane / crazy / unbelievable
]


def title_new(title: str) -> dict[str, float]:
    title = title or ""
    words = re.findall(r"\b[\w']+\b", title)
    lower = title.lower()
    power = (
        "secret", "ultimate", "instant", "easy", "epic", "perfect",
        "simple", "proven", "guaranteed", "insane", "crazy", "shocking",
        "unbelievable", "amazing", "must", "hack", "trick",
    )
    emotional = (
        "shocking", "insane", "crazy", "unbelievable", "wild", "brutal",
        "hilarious", "emotional", "heartbreaking",
    )
    listicle = bool(
        re.search(r"\b\d+\s*(ways|tips|things|reasons|ideas|hacks)\b", lower)
        or re.search(r"\b(top|best)\s+\d+\b", lower)
    )
    non_space = sum(not c.isspace() for c in title)
    return {
        "hashtag_count": float(title.count("#")),
        "all_caps_word_count": float(
            sum(1 for w in words if len(w) >= 2 and w.isupper())
        ),
        "space_count": float(title.count(" ")),
        "char_density": float(non_space / len(title)) if title else 0.0,
        "has_year": float(bool(re.search(r"\b20[2-3]\d\b", title))),
        "has_power_word": float(any(p in lower for p in power)),
        "has_listicle_cue": float(listicle),
        "has_emotional": float(any(e in lower for e in emotional)),
    }


def all_title_side_feats(title: str) -> dict[str, float]:
    out = title_extras(title)
    out.update(title_new(title))
    return out


SIDE_EXTRA = EXTRA_FEATURE_NAMES + NEW_FEATURE_NAMES


def build_full_matrix(pairs, per_video, ref_ids):
    wide, y = build_matrix(pairs, per_video, ref_ids)
    X_base = wide[list(BLEND_FEATURE_NAMES)].to_numpy(dtype=float)
    extra_rows = []
    for a, b in pairs:
        fa, fb = all_title_side_feats(a["title"]), all_title_side_feats(b["title"])
        extra_rows.append([fb[n] - fa[n] for n in SIDE_EXTRA])
    X_extra = np.asarray(extra_rows, dtype=float)
    names = list(BLEND_FEATURE_NAMES) + [f"{n}_diff" for n in SIDE_EXTRA]
    X = np.column_stack([X_base, X_extra])
    return names, X, y


def fit_rf(X_tr, y_tr):
    X_aug = np.vstack([X_tr, -X_tr])
    y_aug = np.concatenate([y_tr, 1 - y_tr])
    clf = RandomForestClassifier(**BLEND_RF_PARAMS)
    clf.fit(X_aug, y_aug)
    return clf


def score(clf, X_te, y_te):
    p = clf.predict_proba(X_te)[:, 1]
    pred = (p > 0.5).astype(int)
    return {
        "acc": float((pred == y_te).mean()),
        "auc": float(roc_auc_score(y_te, p)) if len(np.unique(y_te)) > 1 else 0.5,
        "correct": int((pred == y_te).sum()),
        "n": len(y_te),
        "y": y_te,
        "p": p,
    }


def main():
    os.chdir(PROJECT_ROOT)
    print("=" * 92)
    print("RF FEATURE IMPORTANCE + SELECTION (video-disjoint 80/20)")
    print("=" * 92)

    niche_packs = []
    for niche in NICHES:
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
        names, X, y = build_full_matrix(pairs, per_video, ref_ids)
        tr, te = split_indices(pairs, y, "video")
        if len(te) < 30 or len(np.unique(y[tr])) < 2:
            continue
        niche_packs.append({
            "niche": niche,
            "names": names,
            "X_tr": X[tr],
            "y_tr": y[tr],
            "X_te": X[te],
            "y_te": y[te],
        })
        print(
            f"  prepared {niche:<15} train={len(tr):,} test={len(te):,} "
            f"feats={len(names)}",
            flush=True,
        )

    if not niche_packs:
        print("No niches prepared.")
        return

    names = niche_packs[0]["names"]
    n_feat = len(names)
    print(f"\nCandidate features: {n_feat}")

    # --- Importance: mean across niches of RF feature_importances_ ---
    imp_sum = np.zeros(n_feat, dtype=float)
    for pack in niche_packs:
        clf = fit_rf(pack["X_tr"], pack["y_tr"])
        imp_sum += clf.feature_importances_
        pack["clf_full"] = clf
    mean_imp = imp_sum / len(niche_packs)
    ranked = sorted(
        zip(names, mean_imp), key=lambda kv: -kv[1]
    )

    print(f"\n{'=' * 92}")
    print("MEAN FEATURE IMPORTANCE (higher = used more by RF)")
    print(f"{'=' * 92}")
    print(f"{'Rank':>4} {'Feature':<32} {'Importance':>10}")
    for i, (name, imp) in enumerate(ranked, 1):
        mark = ""
        if name in BLEND_FEATURE_NAMES:
            mark = " [prod]"
        elif name.endswith("_diff") and name[:-5] in EXTRA_FEATURE_NAMES:
            mark = " [extra]"
        else:
            mark = " [new]"
        print(f"{i:>4} {name:<32} {imp:10.4f}{mark}")

    # --- Cumulative top-k selection ---
    order = [name for name, _ in ranked]
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Also evaluate production baseline (30) and full pool
    presets = {
        "production_30": list(BLEND_FEATURE_NAMES),
        "full_pool": names,
    }

    ks = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, n_feat]
    ks = sorted(set(k for k in ks if 1 <= k <= n_feat))

    print(f"\n{'=' * 92}")
    print("CUMULATIVE TOP-K (pooled video-disjoint)")
    print(f"{'=' * 92}")
    print(f"{'Set':<20} {'#F':>4} {'Acc %':>8} {'ROC-AUC':>9}")
    print(f"{'-' * 20} {'-' * 4} {'-' * 8} {'-' * 9}")

    results = []

    def eval_cols(label: str, cols: list[str]):
        idxs = [name_to_idx[c] for c in cols]
        slots = []
        for pack in niche_packs:
            clf = fit_rf(pack["X_tr"][:, idxs], pack["y_tr"])
            slots.append(score(clf, pack["X_te"][:, idxs], pack["y_te"]))
        y = np.concatenate([s["y"] for s in slots])
        p = np.concatenate([s["p"] for s in slots])
        correct = sum(s["correct"] for s in slots)
        n = sum(s["n"] for s in slots)
        row = {
            "label": label,
            "n_feat": len(cols),
            "cols": cols,
            "acc": correct / n,
            "auc": float(roc_auc_score(y, p)),
            "n": n,
        }
        results.append(row)
        print(
            f"{label:<20} {len(cols):>4} "
            f"{row['acc'] * 100:7.1f}% {row['auc']:9.3f}",
            flush=True,
        )
        return row

    for label, cols in presets.items():
        eval_cols(label, cols)

    for k in ks:
        eval_cols(f"top_{k}", order[:k])

    # Importance threshold: keep features that sum to 95% of importance
    cum = 0.0
    keep_95 = []
    for name, imp in ranked:
        keep_95.append(name)
        cum += imp
        if cum >= 0.95:
            break
    eval_cols("imp_95pct", keep_95)

    # Drop near-zero: importance >= median of non-zero or >= 0.005
    nonzero = mean_imp[mean_imp > 0]
    thr = max(0.005, float(np.percentile(nonzero, 40)))
    keep_thr = [n for n, i in zip(names, mean_imp) if i >= thr]
    # preserve ranking order
    keep_thr = [n for n, _ in ranked if n in set(keep_thr)]
    eval_cols(f"imp>={thr:.3f}", keep_thr)

    best = max(results, key=lambda r: (r["auc"], r["acc"], -r["n_feat"]))
    # Prefer a set that beats production by AUC and is not huge
    prod = next(r for r in results if r["label"] == "production_30")
    challengers = [
        r for r in results
        if r["auc"] >= prod["auc"] - 0.001
        and r["acc"] >= prod["acc"] - 0.002
    ]
    # Among non-worse than production, pick best AUC then fewest features
    preferred = max(
        challengers,
        key=lambda r: (r["auc"], r["acc"], -r["n_feat"]),
    )

    print(f"\n{'=' * 92}")
    print("VERDICT")
    print(f"{'=' * 92}")
    print(
        f"Production (30): {prod['acc'] * 100:.1f}% / {prod['auc']:.3f}"
    )
    print(
        f"Best overall:    {best['label']} ({best['n_feat']} feats) = "
        f"{best['acc'] * 100:.1f}% / {best['auc']:.3f}"
    )
    print(
        f"Recommended:     {preferred['label']} ({preferred['n_feat']} feats) = "
        f"{preferred['acc'] * 100:.1f}% / {preferred['auc']:.3f} "
        f"(AUC {(preferred['auc'] - prod['auc']):+.3f}, "
        f"acc {(preferred['acc'] - prod['acc']) * 100:+.1f}pp)"
    )
    print("\nRecommended feature list:")
    for i, name in enumerate(preferred["cols"], 1):
        print(f"  {i:>2}. {name}")

    # Write selection for a follow-up ship step
    out = PROJECT_ROOT / "selected_blend_features.txt"
    out.write_text("\n".join(preferred["cols"]) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
