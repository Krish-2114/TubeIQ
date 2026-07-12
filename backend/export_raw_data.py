"""
Export raw channels + videos join from Postgres to a single CSV.

Run from the tubeiq project root:
    python -m backend.export_raw_data
"""

from __future__ import annotations

import os

import pandas as pd

from backend.database import engine

OUTPUT_PATH = os.path.join("backend", "data", "raw_export.csv")

QUERY = """
SELECT c.channel_id, c.title AS channel_title, c.niche, c.subscriber_count,
       v.video_id, v.title AS video_title, v.description, v.tags,
       v.published_at, v.duration_seconds, v.view_count,
       v.like_count, v.comment_count, v.category_id
FROM videos v
JOIN channels c ON v.channel_id = c.id
ORDER BY c.niche, c.channel_id, v.published_at
"""


def export_raw_data() -> pd.DataFrame:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df = pd.read_sql(QUERY, engine)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Total rows: {len(df):,}")

    print("\nRows per niche:")
    niche_counts = df.groupby("niche").size()
    for niche, count in niche_counts.items():
        print(f"  {niche}: {count:,}")

    null_cols = ["view_count", "duration_seconds", "published_at"]
    print("\nNull counts:")
    for col in null_cols:
        nulls = int(df[col].isna().sum())
        print(f"  {col}: {nulls:,}")

    return df


if __name__ == "__main__":
    export_raw_data()
