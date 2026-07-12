import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timezone
from sqlalchemy import text
from backend.database import SessionLocal
from backend.models import Video, Channel
from backend.ml.train_models import clean_title


def parse_duration_category(seconds):
    if seconds is None or seconds <= 0:
        return "unknown"
    if seconds < 120:
        return "short"
    if seconds < 600:
        return "medium"
    if seconds < 1800:
        return "long"
    return "very_long"


def extract_title_features(title):
    if not title:
        return {
            "title_length": 0,
            "word_count": 0,
            "has_number": 0,
            "has_dollar": 0,
            "has_rupee": 0,
            "has_question": 0,
            "has_exclamation": 0,
            "has_vs": 0,
            "has_challenge": 0,
            "has_hours": 0,
            "has_last": 0,
            "has_win": 0,
            "has_free": 0,
            "has_extreme": 0,
            "has_best": 0,
            "has_worst": 0,
            "has_every": 0,
            "has_first": 0,
            "has_only": 0,
            "has_never": 0,
            "has_days": 0,
            "has_review": 0,
            "has_how": 0,
            "has_why": 0,
            "has_what": 0,
        }
    title_lower = title.lower()
    return {
        "title_length": len(title),
        "word_count": len(title.split()),
        "has_number": int(bool(re.search(r'\d', title))),
        "has_dollar": int("$" in title),
        "has_rupee": int("₹" in title or
                         "rs." in title_lower or
                         "rupee" in title_lower),
        "has_question": int("?" in title),
        "has_exclamation": int("!" in title),
        "has_vs": int(" vs" in title_lower or
                      "vs " in title_lower),
        "has_challenge": int(
            "challenge" in title_lower
        ),
        "has_hours": int(
            "hours" in title_lower or
            "hour" in title_lower
        ),
        "has_last": int("last" in title_lower),
        "has_win": int(
            " win" in title_lower or
            "wins" in title_lower
        ),
        "has_free": int("free" in title_lower),
        "has_extreme": int(
            "extreme" in title_lower
        ),
        "has_best": int("best" in title_lower),
        "has_worst": int("worst" in title_lower),
        "has_every": int("every" in title_lower),
        "has_first": int("first" in title_lower),
        "has_only": int("only" in title_lower),
        "has_never": int("never" in title_lower),
        "has_days": int(
            "days" in title_lower or
            "day" in title_lower
        ),
        "has_review": int("review" in title_lower),
        "has_how": int(
            title_lower.startswith("how")
        ),
        "has_why": int(
            title_lower.startswith("why")
        ),
        "has_what": int(
            title_lower.startswith("what")
        ),
    }


def compute_channel_thresholds(velocities):
    """Per-channel percentile cutoffs on age-corrected view velocity."""
    arr = np.array(velocities)
    return {
        "q25": np.percentile(arr, 25),
        "q60": np.percentile(arr, 60),
        "q90": np.percentile(arr, 90),
    }


def get_performance_label(velocity, thresholds):
    if velocity < thresholds["q25"]:
        return "Low"
    if velocity < thresholds["q60"]:
        return "Medium"
    if velocity < thresholds["q90"]:
        return "High"
    return "Viral"


def video_age_days(published_at, now=None):
    """Days since publish, floored at 1. Returns None if published_at missing."""
    if not published_at:
        return None
    now = now or datetime.utcnow()
    pub = published_at
    if getattr(pub, "tzinfo", None) is not None:
        pub = pub.replace(tzinfo=None)
    elif isinstance(pub, str):
        try:
            pub = datetime.fromisoformat(
                pub.replace("Z", "+00:00")
            ).replace(tzinfo=None)
        except Exception:
            return None
    try:
        return max((now - pub).days, 1)
    except Exception:
        return None


def engineer_features_for_niche(niche):
    db = SessionLocal()
    try:
        channels = db.query(Channel).filter(
            Channel.niche == niche
        ).all()

        if not channels:
            print(f"No channels for niche: {niche}")
            return None

        all_rows = []

        for channel in channels:
            videos = db.query(Video).filter(
                Video.channel_id == channel.id
            ).all()

            if not videos:
                continue

            now = datetime.utcnow()
            eligible = []
            for v in videos:
                if not v.view_count or v.view_count <= 0:
                    continue
                if not v.title:
                    continue
                age_days = video_age_days(v.published_at, now=now)
                if age_days is None:
                    continue
                view_velocity = v.view_count / age_days
                eligible.append((v, age_days, view_velocity))

            if not eligible:
                continue

            view_counts = [v.view_count for v, _, _ in eligible]
            velocities = [vel for _, _, vel in eligible]
            channel_median = np.median(view_counts)
            thresholds = compute_channel_thresholds(velocities)

            for v, age_days, view_velocity in eligible:
                title_feats = extract_title_features(
                    v.title
                )

                # Parse published_at
                day_of_week = -1
                hour_of_day = -1
                if v.published_at:
                    try:
                        if hasattr(
                            v.published_at, 'weekday'
                        ):
                            day_of_week = \
                                v.published_at.weekday()
                            hour_of_day = \
                                v.published_at.hour
                        else:
                            dt = datetime.fromisoformat(
                                str(v.published_at
                                    ).replace(
                                        'Z', '+00:00'
                                    )
                            )
                            day_of_week = dt.weekday()
                            hour_of_day = dt.hour
                    except Exception:
                        pass

                # Cap duration at 3600 seconds
                duration = min(
                    v.duration_seconds or 0, 3600
                )

                # Engagement rate — handle zero likes
                if v.like_count and \
                   v.like_count > 0 and \
                   v.view_count > 0:
                    eng_rate = (
                        v.like_count +
                        (v.comment_count or 0)
                    ) / v.view_count * 100
                else:
                    eng_rate = 0.0

                row = {
                    "video_id": v.video_id,
                    "channel_id": channel.id,
                    "channel_title": channel.title,
                    "niche": niche,
                    "title": v.title,
                    "view_count": v.view_count,
                    "like_count": v.like_count or 0,
                    "comment_count":
                        v.comment_count or 0,
                    "duration_seconds": duration,
                    "engagement_rate": eng_rate,
                    "day_of_week": day_of_week,
                    "hour_of_day": hour_of_day,
                    "age_days": age_days,
                    "view_velocity": view_velocity,
                    "duration_category":
                        parse_duration_category(
                            duration
                        ),
                    "channel_median_views":
                        channel_median,
                    "performance_ratio":
                        v.view_count / max(
                            channel_median, 1
                        ),
                    "performance_label":
                        get_performance_label(
                            view_velocity,
                            thresholds
                        ),
                    **title_feats
                }
                all_rows.append(row)

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)

        # Encode duration category
        duration_map = {
            "short": 0, "medium": 1,
            "long": 2, "very_long": 3,
            "unknown": -1
        }
        df['duration_encoded'] = df[
            'duration_category'
        ].map(duration_map).fillna(-1)

        print(f"\n{niche.upper()}")
        print(f"  Videos: {len(df)}")
        print(f"  Channels: "
              f"{df['channel_id'].nunique()}")
        print(f"  Performance distribution:")
        dist = df['performance_label'].value_counts()
        for label, count in dist.items():
            pct = count / len(df) * 100
            print(f"    {label}: {count} ({pct:.1f}%)")

        return df

    finally:
        db.close()


def compute_niche_insights(df, niche):
    insights = {}

    # 1 — Best upload day
    day_names = [
        'Monday', 'Tuesday', 'Wednesday',
        'Thursday', 'Friday', 'Saturday', 'Sunday'
    ]
    day_perf = df[df['day_of_week'] >= 0].groupby(
        'day_of_week'
    )['performance_ratio'].mean()
    if not day_perf.empty:
        best_day_idx = int(day_perf.idxmax())
        insights['best_upload_day'] = \
            day_names[best_day_idx]
        insights['upload_day_performance'] = {
            day_names[int(k)]: round(float(v), 2)
            for k, v in day_perf.items()
        }

    # 2 — Optimal duration range (exclude shorts under 2 min)
    dur_df = df[df['duration_seconds'] >= 120].copy()
    dur_df['dur_bucket'] = pd.cut(
        dur_df['duration_seconds'],
        bins=[120, 600, 1200, 1800, 3600],
        labels=[
            '2-10min', '10-20min',
            '20-30min', '30-60min'
        ]
    )
    dur_perf = dur_df.groupby(
        'dur_bucket', observed=True
    )['performance_ratio'].mean()
    if not dur_perf.empty:
        best_dur = str(dur_perf.idxmax())
        insights['optimal_duration'] = best_dur
        insights['duration_performance'] = {
            str(k): round(float(v), 2)
            for k, v in dur_perf.items()
        }

    # 3 — Number effect
    with_num = df[df['has_number'] == 1][
        'performance_ratio'
    ].mean()
    without_num = df[df['has_number'] == 0][
        'performance_ratio'
    ].mean()
    insights['number_in_title_boost'] = round(
        float(with_num / max(without_num, 0.01)), 2
    )

    # 4 — Question mark effect
    with_q = df[df['has_question'] == 1][
        'performance_ratio'
    ].mean()
    without_q = df[df['has_question'] == 0][
        'performance_ratio'
    ].mean()
    insights['question_title_boost'] = round(
        float(with_q / max(without_q, 0.01)), 2
    )

    # 5 — Top keywords from high performing titles
    high_perf_df = df[
        df['performance_label'].isin(['High', 'Viral'])
    ].copy()
    high_perf_df['clean_title'] = high_perf_df.apply(
        lambda row: clean_title(
            row.get('title', ''),
            row.get('channel_title', '')
        ),
        axis=1
    )
    high_perf = (
        high_perf_df['clean_title']
        .str.lower()
        .str.split()
        .explode()
    )

    stop_words = {
        'the', 'a', 'an', 'in', 'on', 'at', 'to',
        'for', 'of', 'and', 'or', 'but', 'is', 'i',
        'my', 'me', 'we', 'you', 'it', 'this', 'that',
        'with', 'from', 'by', 'as', 'be', 'was', 'are',
        'have', 'has', 'had', 'do', 'did', 'not', 'so',
        'if', 'he', 'she', 'they', 'his', 'her', 'our'
    }
    keywords = high_perf[
        ~high_perf.isin(stop_words) &
        (high_perf.str.len() > 3)
    ].value_counts().head(20)
    insights['top_keywords'] = keywords.to_dict()

    # 6 — Niche difficulty score
    # Lower median = easier to stand out
    median_views = df['view_count'].median()
    if median_views > 5000000:
        difficulty = "Very Hard"
    elif median_views > 1000000:
        difficulty = "Hard"
    elif median_views > 100000:
        difficulty = "Medium"
    else:
        difficulty = "Easy"
    insights['niche_difficulty'] = difficulty
    insights['niche_median_views'] = int(
        median_views
    )

    # 7 — Engagement quality score
    avg_engagement = df[
        df['engagement_rate'] > 0
    ]['engagement_rate'].mean()
    insights['avg_engagement_rate'] = round(
        float(avg_engagement), 2
    )

    # 8 — Upload frequency benchmark
    insights['total_videos_analyzed'] = len(df)
    insights['channels_analyzed'] = int(
        df['channel_id'].nunique()
    )

    # 9 — Title length sweet spot
    df['title_len_bucket'] = pd.cut(
        df['title_length'],
        bins=[0, 30, 50, 70, 100, 200],
        labels=[
            'very_short', 'short', 'medium',
            'long', 'very_long'
        ]
    )
    title_len_perf = df.groupby(
        'title_len_bucket', observed=True
    )['performance_ratio'].mean()
    if not title_len_perf.empty:
        best_title_len = str(
            title_len_perf.idxmax()
        )
        insights['optimal_title_length'] = \
            best_title_len
        insights['title_length_performance'] = {
            str(k): round(float(v), 2)
            for k, v in title_len_perf.items()
        }

    return insights


NICHES = [
    "gaming", "entertainment", "cooking",
    "finance", "fitness", "education", "tech", "streaming"
]

FEATURE_COLUMNS = [
    "duration_seconds", "duration_encoded",
    "day_of_week", "hour_of_day",
    "engagement_rate",
    "title_length", "word_count",
    "has_number", "has_dollar", "has_rupee",
    "has_question", "has_exclamation",
    "has_vs", "has_challenge", "has_hours",
    "has_last", "has_win", "has_free",
    "has_extreme", "has_best", "has_worst",
    "has_every", "has_first", "has_only",
    "has_never", "has_days", "has_review",
    "has_how", "has_why", "has_what"
]


def export_niche_data():
    import os
    import json
    os.makedirs("backend/data", exist_ok=True)
    os.makedirs("backend/insights", exist_ok=True)

    all_insights = {}

    for niche in NICHES:
        print(f"\nProcessing {niche}...")
        df = engineer_features_for_niche(niche)

        if df is None or len(df) < 50:
            print(f"  Skipping — insufficient data")
            continue

        # Save features CSV for training
        df.to_csv(
            f"backend/data/{niche}_features.csv",
            index=False
        )
        print(f"  Saved: backend/data/"
              f"{niche}_features.csv")

        # Compute and save insights
        insights = compute_niche_insights(df, niche)
        all_insights[niche] = insights

        with open(
            f"backend/insights/{niche}_insights.json",
            'w'
        ) as f:
            json.dump(insights, f, indent=2)
        print(f"  Saved: backend/insights/"
              f"{niche}_insights.json")

    # Save combined insights
    with open(
        "backend/insights/all_insights.json", 'w'
    ) as f:
        json.dump(all_insights, f, indent=2)

    print(f"\n=== Feature Engineering Complete ===")
    print(f"CSVs saved to backend/data/")
    print(f"Insights saved to backend/insights/")
    return all_insights


if __name__ == "__main__":
    export_niche_data()
