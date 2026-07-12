"""
Learning curve analysis for TubeIQ XGBoost classifiers.

Trains on increasing chronological subsets of each niche's
training split and evaluates on the same held-out test set.

Run from the tubeiq project root:
    python learning_curve.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from backend.ml.feature_engineering import get_performance_label
from backend.ml.train_models import FEATURE_COLUMNS, NICHES

PROJECT_ROOT = Path(__file__).resolve().parent
SUBSET_PCTS = [0.2, 0.4, 0.6, 0.8, 1.0]

# Same hyperparameters as train_models.train_xgboost_for_niche
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}


def load_niche_dataset(niche: str) -> pd.DataFrame | None:
    """Load pre-exported feature CSV (same source as train_models.py)."""
    path = PROJECT_ROOT / "backend" / "data" / f"{niche}_features.csv"
    if not path.exists():
        print(f"  Skipping {niche} — CSV not found at {path}")
        return None

    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURE_COLUMNS + ["performance_label"])

    if len(df) < 50:
        print(f"  Skipping {niche} — only {len(df)} rows after dropna")
        return None

    return df


def time_series_split(df: pd.DataFrame):
    """
    Mirror backend/ml/train_models.py:
    Channel-based holdout split using GroupShuffleSplit on `channel_id`.
    Falls back to stratified random split only when fewer than 4 unique channels exist.
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
        train_idx, test_idx = train_test_split(
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
        train_idx, test_idx = next(
            splitter.split(X, y, groups=groups)
        )

    # Preserve chronological ordering within each fold.
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def build_xgboost_classifier() -> XGBClassifier:
    return XGBClassifier(**XGB_PARAMS)


def evaluate_subset(
    X_train,
    y_train,
    X_test,
    y_test,
    subset_pct: float,
):
    """Train on the first `subset_pct` of training data (chronological)."""
    n_subset = max(1, int(len(X_train) * subset_pct))
    X_sub = X_train.iloc[:n_subset]
    y_sub = y_train[:n_subset]

    model = build_xgboost_classifier()
    model.fit(X_sub, y_sub, verbose=False)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return n_subset, accuracy


def print_niche_table(niche: str, niche_results: list[dict]):
    print(f"\n{'=' * 56}")
    print(f"NICHE: {niche.upper()}")
    print(f"{'=' * 56}")
    print(f"{'Subset':>8}  {'Examples':>10}  {'Test accuracy':>14}")
    print(f"{'-' * 8}  {'-' * 10}  {'-' * 14}")
    for row in niche_results:
        print(
            f"{row['subset_pct'] * 100:>7.0f}%  "
            f"{row['num_examples']:>10,}  "
            f"{row['test_accuracy']:>14.3f}"
        )


def save_plot(results_df: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(10, 6))

    for niche in results_df["niche"].unique():
        niche_df = results_df[results_df["niche"] == niche].sort_values(
            "num_examples"
        )
        plt.plot(
            niche_df["num_examples"],
            niche_df["test_accuracy"],
            marker="o",
            linewidth=2,
            label=niche,
        )

    plt.xlabel("Number of training examples")
    plt.ylabel("Test accuracy")
    plt.title("TubeIQ Learning Curves by Niche")
    plt.legend(title="Niche", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    os.chdir(PROJECT_ROOT)

    # Reference imported preprocessing helper (labels in CSV were built with this).
    _ = get_performance_label

    print("=" * 56)
    print("TUBEIQ LEARNING CURVE ANALYSIS")
    print("=" * 56)
    print(f"Niches: {', '.join(NICHES)}")
    print(f"Subset sizes: {[int(p * 100) for p in SUBSET_PCTS]}% of training data")
    print(f"Output dir: {PROJECT_ROOT}")

    all_results: list[dict] = []

    for niche in NICHES:
        print(f"\n>>> Processing niche: {niche}")

        df = load_niche_dataset(niche)
        if df is None:
            continue

        print(f"    Loaded {len(df):,} videos")
        X_train, X_test, y_train, y_test = time_series_split(df)
        print(
            f"    Train: {len(X_train):,} | "
            f"Test: {len(X_test):,} (fixed held-out set)"
        )

        niche_results: list[dict] = []

        for subset_pct in SUBSET_PCTS:
            print(
                f"    Training subset {int(subset_pct * 100):>3}% ...",
                end=" ",
                flush=True,
            )
            num_examples, test_accuracy = evaluate_subset(
                X_train,
                y_train,
                X_test,
                y_test,
                subset_pct,
            )
            print(
                f"done — {num_examples:,} examples, "
                f"test accuracy {test_accuracy:.3f}"
            )

            row = {
                "niche": niche,
                "subset_pct": subset_pct,
                "num_examples": num_examples,
                "test_accuracy": round(test_accuracy, 4),
            }
            niche_results.append(row)
            all_results.append(row)

        print_niche_table(niche, niche_results)

    if not all_results:
        print("\nNo results generated. Check that backend/data CSVs exist.")
        sys.exit(1)

    results_df = pd.DataFrame(all_results)
    csv_path = PROJECT_ROOT / "learning_curve_results.csv"
    plot_path = PROJECT_ROOT / "learning_curve_plot.png"

    results_df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 56}")
    print("ALL NICHES COMPLETE")
    print(f"{'=' * 56}")
    print(f"Saved results: {csv_path}")
    print(f"Saved plot:    {plot_path}")

    save_plot(results_df, plot_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
