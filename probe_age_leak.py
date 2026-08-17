"""
Single-feature probe: how much does each candidate feature know on its own?

A feature that predicts the winner by itself, with no title information at all,
is not measuring title quality. Run under the video-disjoint protocol so
memorization is not what we are measuring.

    python probe_age_leak.py
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from backend.ml.train_models import MIN_BLEND_PAIRS, NICHES
from feature_model_sweep import (
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

PROBES = {
    "age_days_diff only": ["age_days_diff"],
    "duration_seconds_diff only": ["duration_seconds_diff"],
    "hour_of_day_diff only": ["hour_of_day_diff"],
    "day_of_week_diff only": ["day_of_week_diff"],
    "tfidf_diff only": ["tfidf_diff"],
    "title_length_diff only": ["title_length_diff"],
}


def main():
    os.chdir(PROJECT_ROOT)
    print("=" * 72)
    print("SINGLE-FEATURE PROBE (video-disjoint split, pooled all niches)")
    print("=" * 72)

    pooled = {name: {"y": [], "p": []} for name in PROBES}

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

        X_all, y_all = build_matrix(pairs, per_video, ref_ids)
        tr, te = split_indices(pairs, y_all, "video")
        if len(te) < 50 or len(np.unique(y_all[tr])) < 2:
            continue

        for name, cols in PROBES.items():
            Xf = X_all[cols].to_numpy(dtype=float)
            scaler = StandardScaler()
            model = LogisticRegression(max_iter=1000, random_state=SEED)
            model.fit(scaler.fit_transform(Xf[tr]), y_all[tr])
            probs = model.predict_proba(scaler.transform(Xf[te]))[:, 1]
            pooled[name]["y"].append(y_all[te])
            pooled[name]["p"].append(probs)

    print(f"\n{'Feature':<30} {'ROC-AUC':>9}  interpretation")
    print(f"{'-' * 30} {'-' * 9}  {'-' * 30}")
    results = []
    for name, slot in pooled.items():
        if not slot["y"]:
            continue
        y = np.concatenate(slot["y"])
        p = np.concatenate(slot["p"])
        results.append((name, float(roc_auc_score(y, p))))
    results.sort(key=lambda r: -abs(r[1] - 0.5))

    # Title-derived features are allowed to be predictive on their own; that is
    # the signal we want. Metadata features are not — if they predict the
    # winner alone, they are encoding the outcome rather than title quality.
    title_derived = {"tfidf_diff only", "title_length_diff only"}
    for name, auc in results:
        strength = abs(auc - 0.5)
        if name in title_derived:
            note = "title signal (legitimate)"
        elif strength > 0.08:
            note = "LEAKAGE - predicts the winner with no title input"
        elif strength > 0.03:
            note = "metadata, weak signal"
        else:
            note = "essentially no signal"
        print(f"{name:<30} {auc:9.3f}  {note}")

    print(
        "\nview_velocity = views / age_days, so the younger video in a pair "
        "mechanically\ntends to win. age_days must stay out of the feature "
        "set. It is also unavailable\nat prediction time, since /compare "
        "receives two title strings and nothing else."
    )


if __name__ == "__main__":
    main()
