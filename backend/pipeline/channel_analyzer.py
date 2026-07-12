from datetime import datetime

import pandas as pd
import numpy as np
from backend.database import SessionLocal
from backend.models import Channel, Video
from backend.pipeline.youtube_fetcher import (
    fetch_and_store_channel
)
from backend.ml.feature_engineering import (
    extract_title_features,
    parse_duration_category,
    compute_channel_thresholds,
    get_performance_label,
    video_age_days,
)


def analyze_channel(identifier):
    
    # Step 1 - fetch channel if not in DB
    db = SessionLocal()
    try:
        channel_info_resp = None
        from backend.pipeline.youtube_fetcher import (
            get_channel_info
        )
        channel_info_resp = get_channel_info(
            identifier
        )
        
        if not channel_info_resp:
            return {"error": "Channel not found"}
        
        channel = db.query(Channel).filter(
            Channel.channel_id == 
            channel_info_resp["channel_id"]
        ).first()
        
        if not channel:
            db.close()
            channel = fetch_and_store_channel(
                identifier
            )
            if not channel:
                return {"error": "Could not fetch channel"}
            db = SessionLocal()
            channel = db.query(Channel).filter(
                Channel.channel_id == 
                channel_info_resp["channel_id"]
            ).first()
        
        # Step 2 - get all videos for this channel
        videos = db.query(Video).filter(
            Video.channel_id == channel.id
        ).all()
        
        if len(videos) < 10:
            return {
                "error": "Not enough videos to analyze"
            }
        
        # Step 3 - build dataframe with age-corrected velocity
        now = datetime.utcnow()
        rows = []
        for v in videos:
            if not v.view_count or v.view_count <= 0:
                continue

            age_days = video_age_days(v.published_at, now=now)
            if age_days is None:
                continue

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
                except Exception:
                    pass

            view_velocity = v.view_count / age_days
            rows.append({
                "title": v.title or "",
                "view_count": v.view_count,
                "like_count": v.like_count or 0,
                "comment_count": v.comment_count or 0,
                "duration_seconds": min(
                    v.duration_seconds or 0, 3600
                ),
                "day_of_week": day_of_week,
                "hour_of_day": hour_of_day,
                "published_at": v.published_at,
                "age_days": age_days,
                "view_velocity": view_velocity,
            })

        df = pd.DataFrame(rows)

        if len(df) == 0:
            return {"error": "No valid videos found"}

        # Step 4 - percentile labels on view velocity; keep median on raw views
        median_views = df['view_count'].median()
        thresholds = compute_channel_thresholds(
            df['view_velocity'].tolist()
        )
        df['performance_label'] = df[
            'view_velocity'
        ].apply(
            lambda x: get_performance_label(
                x, thresholds
            )
        )
        df['performance_ratio'] = df[
            'view_count'
        ] / max(median_views, 1)
        
        # Step 5 - best upload day
        day_names = [
            'Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 
            'Saturday', 'Sunday'
        ]
        day_perf = df[
            df['day_of_week'] >= 0
        ].groupby('day_of_week')[
            'performance_ratio'
        ].mean()
        
        best_day = None
        day_performance = {}
        if not day_perf.empty:
            best_day = day_names[
                int(day_perf.idxmax())
            ]
            day_performance = {
                day_names[int(k)]: round(float(v), 2)
                for k, v in day_perf.items()
            }
        
        # Step 6 - optimal duration
        df['dur_bucket'] = pd.cut(
            df['duration_seconds'],
            bins=[0, 120, 600, 1200, 1800, 3600],
            labels=[
                '0-2min', '2-10min', '10-20min',
                '20-30min', '30-60min'
            ]
        )
        dur_perf = df.groupby(
            'dur_bucket', observed=True
        )['performance_ratio'].mean()
        
        optimal_duration = None
        if not dur_perf.empty:
            optimal_duration = str(
                dur_perf.idxmax()
            )
        
        # Step 7 - top performing content types
        high_df = df[
            df['performance_label'].isin(
                ['High', 'Viral']
            )
        ]
        
        top_keywords = []
        if len(high_df) > 0:
            import re
            stop_words = {
                'the', 'a', 'an', 'in', 'on', 
                'at', 'to', 'for', 'of', 'and', 
                'or', 'but', 'is', 'i', 'my', 
                'me', 'we', 'you', 'it', 'this',
                'that', 'with', 'from', 'by'
            }
            words = high_df['title'].str.lower(
            ).str.split().explode()
            top_kw = words[
                ~words.isin(stop_words) & 
                (words.str.len() > 3)
            ].value_counts().head(10)
            top_keywords = top_kw.to_dict()
        
        # Step 8 - performance distribution
        dist = df[
            'performance_label'
        ].value_counts().to_dict()
        
        # Step 9 - top 5 best videos
        top_videos = df.nlargest(
            5, 'view_count'
        )[[
            'title', 'view_count', 
            'performance_label',
            'duration_seconds'
        ]].to_dict('records')
        
        # Step 10 - upload frequency
        if 'published_at' in df.columns:
            dates = pd.to_datetime(
                df['published_at'], 
                errors='coerce'
            ).dropna().sort_values()
            if len(dates) > 1:
                date_range = (
                    dates.max() - dates.min()
                ).days
                upload_freq = round(
                    len(dates) / 
                    max(date_range / 7, 1), 1
                )
            else:
                upload_freq = None
        else:
            upload_freq = None
        
        # Step 11 - engagement quality
        eng_df = df[df['view_count'] > 0].copy()
        eng_df['eng_rate'] = (
            eng_df['like_count'] + 
            eng_df['comment_count']
        ) / eng_df['view_count'] * 100
        avg_engagement = round(
            float(eng_df['eng_rate'].mean()), 2
        )
        
        return {
            "channel": {
                "title": channel.title,
                "channel_id": channel.channel_id,
                "subscriber_count": 
                    channel.subscriber_count,
                "thumbnail_url": 
                    channel.thumbnail_url,
                "total_videos_analyzed": len(df)
            },
            "performance": {
                "median_views": int(median_views),
                "avg_engagement_rate": avg_engagement,
                "distribution": dist,
                "top_videos": top_videos
            },
            "upload_insights": {
                "best_upload_day": best_day,
                "day_performance": day_performance,
                "optimal_duration": optimal_duration,
                "uploads_per_week": upload_freq
            },
            "content_insights": {
                "top_keywords_in_best_videos": 
                    top_keywords,
                "high_performing_count": len(high_df),
                "viral_count": len(df[
                    df['performance_label'] == 'Viral'
                ])
            }
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    import json
    identifier = sys.argv[1] \
        if len(sys.argv) > 1 else "@mkbhd"
    result = analyze_channel(identifier)
    print(json.dumps(result, indent=2, 
                     default=str))
