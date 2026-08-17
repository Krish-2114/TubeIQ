"""
Test additional title-derived features with the production Random Forest only.

The split is video-disjoint 80/20 within each niche, so a video cannot appear
in both train and test. Every candidate feature is known before publishing;
age_days and every post-publish engagement/view metric are intentionally
excluded.

Run from the tubeiq project root:
    python rf_feature_expansion.py
"""

from __future__ import annotations

import os
import re
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

PROJECT_ROOT = Path(__file__).resolve().parent

# These all derive solely from the title text available at comparison time.
EXTRA_FEATURE_NAMES = [
    "digit_count",
    "uppercase_count",
    "uppercase_ratio",
    "exclamation_count",
    "question_count",
    "punctuation_count",
    "emoji_count",
    "word_length_mean",
    "word_length_max",
    "long_word_count",
    "unique_word_ratio",
    "starts_with_number",
    "starts_with_question",
    "has_colon",
    "has_parentheses",
    "has_brackets",
    "has_quotes",
    "has_ellipsis",
]


def title_extras(title: str) -> dict[str, float]:
    title = title or ""
    words = re.findall(r"\b[\w']+\b", title.lower())
    word_lengths = [len(word) for word in words]
    letters = [char for char in title if char.isalpha()]
    # Broad Unicode ranges, deliberately only used as a count—not language ID.
    emoji_count = len(
        re.findall(
            r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", title
        )
    )
    return {
        "digit_count": float(sum(char.isdigit() for char in title)),
        "uppercase_count": float(sum(char.isupper() for char in title)),
        "uppercase_ratio": (
            sum(char.isupper() for char in letters) / len(letters)
            if letters
            else 0.0
        ),
        "exclamation_count": float(title.count("!")),
        "question_count": float(title.count("?")),
        "punctuation_count": float(
            sum(char in "!?.,:;—-()[]{}\"'" for char in title)
        ),
        "emoji_count": float(emoji_count),
        "word_length_mean": (
            float(np.mean(word_lengths)) if word_lengths else 0.0
        ),
        "word_length_max": float(max(word_lengths)) if word_lengths else 0.0,
        "long_word_count": float(sum(length >= 8 for length in word_lengths)),
        "unique_word_ratio": (
            len(set(words)) / len(words) if words else 0.0
        ),
        "starts_with_number": float(bool(re.match(r"\s*\d", title))),
        "starts_with_question": float(title.lower().startswith(
            ("how", "why", "what", "when", "where", "who", "is ", "are ")
        )),
        "has_colon": float(":" in title),
        "has_parentheses": float("(" in title or ")" in title),
        "has_brackets": float("[" in title or "]" in title),
        "has_quotes": float('"' in title or "'" in title),
        "has_ellipsis": float("..." in title or "…" in title),
    }


def append_extra_diffs(X_base, pairs):
    rows = []
    for row_a, row_b in pairs:
        a = title_extras(row_a["title"])
        b = title_extras(row_b["title"])
        rows.append([b[name] - a[name] for name in EXTRA_FEATURE_NAMES])
    return np.column_stack([X_base, np.asarray(rows, dtype=float)])


def evaluate(X, y, train_idx, test_idx):
    # Matches current production RF training: both A-B directions.
    X_train = np.vstack([X[train_idx], -X[train_idx]])
    y_train = np.concatenate([y[train_idx], 1 - y[train_idx]])
    model = RandomForestClassifier(**BLEND_RF_PARAMS)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X[test_idx])[:, 1]
    predictions = (probabilities > 0.5).astype(int)
    return {
        "correct": int((predictions == y[test_idx]).sum()),
        "n": len(test_idx),
        "y": y[test_idx],
        "p": probabilities,
    }


def summarize(name, slots):
    y = np.concatenate([slot["y"] for slot in slots])
    probabilities = np.concatenate([slot["p"] for slot in slots])
    correct = sum(slot["correct"] for slot in slots)
    n = sum(slot["n"] for slot in slots)
    return {
        "name": name,
        "acc": correct / n,
        "auc": float(roc_auc_score(y, probabilities)),
        "n": n,
    }


def main():
    os.chdir(PROJECT_ROOT)
    print("=" * 88)
    print("RF FEATURE EXPANSION — VIDEO-DISJOINT 80/20")
    print("=" * 88)
    print(
        f"Baseline: title_meta RF ({len(BLEND_FEATURE_NAMES)} features) | "
        f"Expansion: +{len(EXTRA_FEATURE_NAMES)} title-only features"
    )

    pooled = {"baseline": [], "expanded": []}
    per_niche = []

    for niche in NICHES:
        df = load_and_filter(niche)
        if df is None or df.empty:
            continue
        tfidf = fit_tfidf(df)
        if tfidf is None:
            continue
        per_video, reference_ids = build_sim_index(df, tfidf)
        if per_video is None:
            continue
        pairs = build_pairs(df, np.random.default_rng(SEED))
        if len(pairs) < MIN_BLEND_PAIRS:
            continue

        wide, y = build_matrix(pairs, per_video, reference_ids)
        X_base = wide[list(BLEND_FEATURE_NAMES)].to_numpy(dtype=float)
        X_expanded = append_extra_diffs(X_base, pairs)
        train_idx, test_idx = split_indices(pairs, y, "video")
        if len(test_idx) < 30 or len(np.unique(y[train_idx])) < 2:
            continue

        base = evaluate(X_base, y, train_idx, test_idx)
        expanded = evaluate(X_expanded, y, train_idx, test_idx)
        pooled["baseline"].append(base)
        pooled["expanded"].append(expanded)

        base_auc = roc_auc_score(base["y"], base["p"])
        expanded_auc = roc_auc_score(expanded["y"], expanded["p"])
        per_niche.append(
            (
                niche,
                len(test_idx),
                base["correct"] / base["n"],
                base_auc,
                expanded["correct"] / expanded["n"],
                expanded_auc,
            )
        )
        print(
            f"  {niche:<15} baseline={base['correct'] / base['n'] * 100:5.1f}%"
            f"/{base_auc:.3f}  expanded="
            f"{expanded['correct'] / expanded['n'] * 100:5.1f}%"
            f"/{expanded_auc:.3f}"
        )

    base_total = summarize("baseline", pooled["baseline"])
    expanded_total = summarize("expanded", pooled["expanded"])

    print(f"\n{'=' * 88}")
    print("RESULTS")
    print(f"{'=' * 88}")
    print(
        f"{'Niche':<15} {'Test':>8} | {'Base acc':>8} {'Base AUC':>8} | "
        f"{'Extra acc':>9} {'Extra AUC':>9} | {'AUC lift':>8}"
    )
    print("-" * 88)
    for niche, n, ba, bu, ea, eu in per_niche:
        print(
            f"{niche:<15} {n:>8,} | {ba * 100:7.1f}% {bu:8.3f} | "
            f"{ea * 100:8.1f}% {eu:9.3f} | {eu - bu:+8.3f}"
        )
    print("-" * 88)
    print(
        f"{'OVERALL':<15} {base_total['n']:>8,} | "
        f"{base_total['acc'] * 100:7.1f}% {base_total['auc']:8.3f} | "
        f"{expanded_total['acc'] * 100:8.1f}% {expanded_total['auc']:9.3f} | "
        f"{expanded_total['auc'] - base_total['auc']:+8.3f}"
    )
    print(
        "\nShip only if expanded improves pooled AUC and does not materially "
        "harm most niches."
    )


if __name__ == "__main__":
    main()
