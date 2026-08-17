"""Retrain only blend_{niche}.pkl models (TF-IDF + reference set must already exist)."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd

from backend.ml.train_models import NICHES, train_blend_for_niche

ROOT = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    results = []
    for niche in NICHES:
        path = ROOT / "backend" / "data" / f"{niche}_features.csv"
        if not path.exists():
            print(f"Skipping {niche} — no CSV")
            continue
        tfidf_path = ROOT / "backend" / "saved_models" / f"tfidf_{niche}.pkl"
        if not tfidf_path.exists():
            print(f"Skipping {niche} — no TF-IDF model")
            continue
        print(f"\n{'=' * 40}\nNICHE: {niche.upper()}\n{'=' * 40}")
        df = pd.read_csv(path)
        tfidf = joblib.load(tfidf_path)
        row = train_blend_for_niche(df, niche, tfidf=tfidf)
        if row:
            results.append(row)

    print(f"\nRetrained {len(results)} blend models.")
    for row in results:
        print(
            f"  {row['niche']:<15} train={row['train_accuracy'] * 100:5.1f}% "
            f"held-out≈{(row['held_out_accuracy'] or 0) * 100:5.1f}% "
            f"feats={row['n_features']}"
        )


if __name__ == "__main__":
    main()
