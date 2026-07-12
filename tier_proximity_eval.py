"""
Evaluate trained XGBoost models with ordinal tier proximity metrics.

Run from the tubeiq project root:
    python tier_proximity_eval.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

from backend.ml.train_models import FEATURE_COLUMNS, NICHES

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "backend" / "saved_models"
DATA_DIR = PROJECT_ROOT / "backend" / "data"

# Explicit ordinal tier order (matches LabelEncoder.fit order in train_models.py)
TIER_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Viral": 3}


def load_niche_dataset(niche: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{niche}_features.csv"
    if not path.exists():
        print(f"  Skipping {niche} — CSV not found")
        return None

    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURE_COLUMNS + ["performance_label"])

    if len(df) < 50:
        print(f"  Skipping {niche} — only {len(df)} rows after dropna")
        return None

    return df


def get_train_test_split(df: pd.DataFrame):
    """
    Same fixed split as backend/ml/train_models.py:
    Channel-based holdout split using GroupShuffleSplit on `channel_id`,
    with stratified random fallback only when fewer than 4 unique channels exist.
    """
    X = df[FEATURE_COLUMNS].fillna(0)
    y_raw = df["performance_label"]

    le = LabelEncoder()
    le.fit(["Low", "Medium", "High", "Viral"])
    y = le.transform(y_raw)

    groups = df["channel_id"]
    unique_channels = int(groups.nunique(dropna=True))

    if unique_channels < 4:
        idx = np.arange(len(df))
        _, test_idx = train_test_split(
            idx,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    else:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=42,
        )
        _, test_idx = next(
            splitter.split(X, y, groups=groups)
        )

    # Preserve ordering for any downstream indexing assumptions.
    test_idx = np.sort(test_idx)
    X_test = X.iloc[test_idx]
    y_test = y[test_idx]

    return X_test, y_test


def load_trained_model(niche: str):
    model_path = MODELS_DIR / f"xgboost_{niche}.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def top2_accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    top2_indices = np.argsort(proba, axis=1)[:, -2:]
    hits = [y_true[i] in top2_indices[i] for i in range(len(y_true))]
    return float(np.mean(hits))


def offbyone_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    hits = np.abs(y_true - y_pred) <= 1
    return float(np.mean(hits))


def evaluate_niche(niche: str) -> dict | None:
    print(f"\n>>> Niche: {niche}")

    model = load_trained_model(niche)
    if model is None:
        print("  Skipping — trained model not found")
        return None

    df = load_niche_dataset(niche)
    if df is None:
        return None

    X_test, y_test = get_train_test_split(df)
    print(f"  Test set size: {len(X_test):,}")

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    exact = accuracy_score(y_test, y_pred)
    top2 = top2_accuracy(y_test, proba)
    offbyone = offbyone_accuracy(y_test, y_pred)

    print(f"  Exact-match accuracy:  {exact * 100:.1f}%")
    print(f"  Top-2 accuracy:        {top2 * 100:.1f}%")
    print(f"  Off-by-one accuracy:   {offbyone * 100:.1f}%")

    return {
        "niche": niche,
        "exact_accuracy": round(exact * 100, 2),
        "top2_accuracy": round(top2 * 100, 2),
        "offbyone_accuracy": round(offbyone * 100, 2),
    }


def print_summary_table(results: list[dict]):
    print(f"\n{'=' * 62}")
    print("TIER PROXIMITY EVAL — SUMMARY")
    print(f"{'=' * 62}")
    print(
        f"{'Niche':<15} {'Exact %':>10} {'Top-2 %':>10} {'Off-by-1 %':>12}"
    )
    print(f"{'-' * 15} {'-' * 10} {'-' * 10} {'-' * 12}")

    for row in results:
        print(
            f"{row['niche']:<15} "
            f"{row['exact_accuracy']:>9.1f}% "
            f"{row['top2_accuracy']:>9.1f}% "
            f"{row['offbyone_accuracy']:>11.1f}%"
        )

    avg_exact = np.mean([r["exact_accuracy"] for r in results])
    avg_top2 = np.mean([r["top2_accuracy"] for r in results])
    avg_offbyone = np.mean([r["offbyone_accuracy"] for r in results])

    print(f"{'-' * 15} {'-' * 10} {'-' * 10} {'-' * 12}")
    print(
        f"{'AVERAGE':<15} "
        f"{avg_exact:>9.1f}% "
        f"{avg_top2:>9.1f}% "
        f"{avg_offbyone:>11.1f}%"
    )

    return avg_exact, avg_top2, avg_offbyone


def main():
    os.chdir(PROJECT_ROOT)

    print("=" * 62)
    print("TUBEIQ TIER PROXIMITY EVAL")
    print("=" * 62)
    print("Tier order: Low=0, Medium=1, High=2, Viral=3")
    print("Loading pre-trained XGBoost models (no retraining)")

    results: list[dict] = []

    for niche in NICHES:
        row = evaluate_niche(niche)
        if row:
            results.append(row)

    if not results:
        print("\nNo niches evaluated.")
        sys.exit(1)

    csv_path = PROJECT_ROOT / "tier_proximity_results.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)

    avg_exact, avg_top2, avg_offbyone = print_summary_table(results)

    print(f"\nAverage exact-match accuracy:  {avg_exact:.1f}%")
    print(f"Average top-2 accuracy:        {avg_top2:.1f}%")
    print(f"Average off-by-one accuracy:   {avg_offbyone:.1f}%")
    print(f"\nSaved results: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
