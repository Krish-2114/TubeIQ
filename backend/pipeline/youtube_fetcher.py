import requests
import time
import isodate
from datetime import datetime, timezone
from backend.database import SessionLocal
from backend.models import Channel, Video
from backend.config import YOUTUBE_API_KEY

YOUTUBE_BASE_URL = "https://www.googleapis.com/youtube/v3"

TRAINING_CHANNELS = {
    # Reconciled against DB custom_url / channel_title, then expanded
    # for LOCO diversity (education prioritized; cooking/fitness unchanged).
    "gaming": [
        "@CarryMinati",
        "@TechnoGamerzOfficial",
        "@IShowSpeed",
        "@GyanGaming",
        "@Markiplier",
        "@PewDiePie",
        "@jacksepticeye",
        "@VanossGaming",  # multiplayer comedy-montage (new)
    ],
    "entertainment": [
        "@MrBeast",
        "@DudePerfect",
        "@rug",  # FaZe Rug
        "@KSI",
        "@loganpaulvlogs",  # Logan Paul Vlogs (corrected)
        "@Airrack",  # short-form challenge/experiment
        "@ZHCYT",  # Zach Hsieh / ZHC art challenges (corrected)
        "@ryan",  # Ryan Trahan endurance/challenge docs
        "@emmachamberlain",  # Emma Chamberlain low-key vlogs
    ],
    "cooking": [
        "@gordonramsay",
        "@bingingwithbabish",
        "@ranveerbrar",  # Chef Ranveer Brar
        "@nickdigiovanni",
        "@SortedFood",
        "@joshuaweissman",
        "@buzzfeedtasty",  # Tasty
    ],
    "tech": [
        "@mkbhd",
        "@LinusTechTips",
        "@Mrwhosetheboss",
        "@unboxtherapy",  # unboxing-first (new)
        "@JerryRigEverything",  # durability/teardown (new)
        "@Dave2D",  # short aesthetic reviews (new)
    ],
    "finance": [
        "@GrahamStephan",
        "@AndreiJikh",
        "@CARachanaRanade",
        "@warikoo",  # Indian career-and-finance (new)
        "@MeetKevin",  # real estate/stocks hot takes (new)
    ],
    "education": [
        "@kurzgesagt",
        "@veritasium",
        "@Vsauce",  # philosophical essay / live-action (new)
        "@TEDEd",  # short curriculum animated lessons (new)
        "@CGPGrey",  # minimalist maps/diagram essays (new)
        "@SmarterEveryDay",  # hands-on experiment format (new)
    ],
    "fitness": [
        "@JeffNippard",
        "@athleanx",  # ATHLEAN-X™
    ],
    "streaming": [
        "@Ludwig",
        "@Pokimane",
        "@penguinz0",
        "@xqcow",  # xQc
    ],
}


def get_channel_info(identifier):
    # Try by handle first
    if identifier.startswith("@"):
        params = {
            "part": "snippet,statistics",
            "forHandle": identifier,
            "key": YOUTUBE_API_KEY
        }
    # Try by channel ID
    elif identifier.startswith("UC"):
        params = {
            "part": "snippet,statistics",
            "id": identifier,
            "key": YOUTUBE_API_KEY
        }
    # Try by username
    else:
        params = {
            "part": "snippet,statistics",
            "forHandle": f"@{identifier}",
            "key": YOUTUBE_API_KEY
        }

    resp = requests.get(
        f"{YOUTUBE_BASE_URL}/channels",
        params=params,
        timeout=10
    )

    if resp.status_code != 200:
        return None

    data = resp.json()
    items = data.get("items", [])

    if not items:
        return None

    item = items[0]
    stats = item.get("statistics", {})
    snippet = item.get("snippet", {})

    return {
        "channel_id": item["id"],
        "title": snippet.get("title"),
        "description": snippet.get("description"),
        "custom_url": snippet.get("customUrl"),
        "thumbnail_url": snippet.get(
            "thumbnails", {}
        ).get("high", {}).get("url"),
        "subscriber_count": int(
            stats.get("subscriberCount", 0)
        ),
        "view_count": int(
            stats.get("viewCount", 0)
        ),
        "video_count": int(
            stats.get("videoCount", 0)
        ),
        "country": snippet.get("country"),
        "published_at": snippet.get("publishedAt")
    }


def get_uploads_playlist_id(channel_id):
    resp = requests.get(
        f"{YOUTUBE_BASE_URL}/channels",
        params={
            "part": "contentDetails",
            "id": channel_id,
            "key": YOUTUBE_API_KEY
        },
        timeout=10
    )

    if resp.status_code != 200:
        return None

    data = resp.json()
    items = data.get("items", [])

    if not items:
        return None

    return items[0].get(
        "contentDetails", {}
    ).get(
        "relatedPlaylists", {}
    ).get("uploads")


def get_video_ids(playlist_id, max_videos=200):
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_videos:
        params = {
            "part": "contentDetails",
            "playlistId": playlist_id,
            "maxResults": 50,
            "key": YOUTUBE_API_KEY
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        resp = requests.get(
            f"{YOUTUBE_BASE_URL}/playlistItems",
            params=params,
            timeout=10
        )

        if resp.status_code != 200:
            break

        data = resp.json()
        items = data.get("items", [])

        for item in items:
            vid_id = item.get(
                "contentDetails", {}
            ).get("videoId")
            if vid_id:
                video_ids.append(vid_id)

        next_page_token = data.get("nextPageToken")

        if not next_page_token or \
           len(video_ids) >= max_videos:
            break

        time.sleep(0.1)

    return video_ids[:max_videos]


def get_video_details(video_ids):
    if not video_ids:
        return []

    resp = requests.get(
        f"{YOUTUBE_BASE_URL}/videos",
        params={
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(video_ids),
            "key": YOUTUBE_API_KEY
        },
        timeout=15
    )

    if resp.status_code != 200:
        return []

    videos = []
    for item in resp.json().get("items", []):
        snippet = item.get("snippet", {})
        stats   = item.get("statistics", {})
        details = item.get("contentDetails", {})

        duration_str = details.get("duration", "PT0S")
        try:
            duration_seconds = int(
                isodate.parse_duration(
                    duration_str
                ).total_seconds()
            )
        except Exception:
            duration_seconds = 0

        tags = snippet.get("tags", [])

        view_count    = int(stats.get("viewCount", 0))
        like_count    = int(stats.get("likeCount", 0))
        comment_count = int(
            stats.get("commentCount", 0)
        )

        engagement_rate = round(
            (like_count + comment_count) /
            max(view_count, 1) * 100, 4
        )

        videos.append({
            "video_id": item["id"],
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "thumbnail_url": snippet.get(
                "thumbnails", {}
            ).get("high", {}).get("url"),
            "published_at": snippet.get("publishedAt"),
            "duration_seconds": duration_seconds,
            "view_count": view_count,
            "like_count": like_count,
            "comment_count": comment_count,
            "tags": ",".join(tags) if tags else None,
            "category_id": snippet.get("categoryId"),
            "engagement_rate": engagement_rate
        })

    return videos


def fetch_and_store_channel(
    identifier, max_videos=200
):
    db = SessionLocal()
    try:
        print(f"Fetching channel: {identifier}")

        # Step 1 - get channel info
        channel_info = get_channel_info(identifier)
        if not channel_info:
            print("Channel not found")
            return None

        print(f"Found: {channel_info['title']}")
        print(f"Subscribers: "
              f"{channel_info['subscriber_count']:,}")

        # Step 2 - store or update channel in DB
        channel = db.query(Channel).filter(
            Channel.channel_id ==
            channel_info["channel_id"]
        ).first()

        if not channel:
            channel = Channel(**channel_info)
            db.add(channel)
            db.commit()
            db.refresh(channel)
            print(f"Channel stored — ID: {channel.id}")
        else:
            for k, v in channel_info.items():
                setattr(channel, k, v)
            db.commit()
            print(f"Channel updated — ID: {channel.id}")

        # Step 3 - get uploads playlist ID
        playlist_id = get_uploads_playlist_id(
            channel_info["channel_id"]
        )
        if not playlist_id:
            print("Could not get uploads playlist")
            return channel

        # Step 4 - get video IDs
        print(f"Fetching up to {max_videos} "
              f"video IDs...")
        video_ids = get_video_ids(
            playlist_id, max_videos
        )
        print(f"Found {len(video_ids)} videos")

        # Step 5 - fetch video details in batches
        # of 50
        total_stored = 0
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            videos = get_video_details(batch_ids)

            for v in videos:
                existing = db.query(Video).filter(
                    Video.video_id == v["video_id"]
                ).first()

                if existing:
                    for k, val in v.items():
                        setattr(existing, k, val)
                else:
                    video = Video(
                        channel_id=channel.id,
                        **v
                    )
                    db.add(video)

                total_stored += 1

            db.commit()
            print(f"  Stored batch "
                  f"{i//50 + 1}: "
                  f"{len(videos)} videos")
            time.sleep(0.2)

        print(f"\n=== Fetch Complete ===")
        print(f"Channel: {channel_info['title']}")
        print(f"Videos stored: {total_stored}")
        return channel

    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        db.close()


def fetch_all_training_channels(max_videos=200):
    print("="*50)
    print("FETCHING ALL TRAINING CHANNELS")
    print("="*50)

    total_channels = sum(
        len(v) for v in TRAINING_CHANNELS.values()
    )
    print(f"Total channels to fetch: "
          f"{total_channels}")
    print(f"Max videos per channel: {max_videos}")
    print(f"Estimated total videos: "
          f"{total_channels * max_videos}")

    results = {}
    completed = 0

    for niche, handles in TRAINING_CHANNELS.items():
        print(f"\n--- {niche.upper()} ---")
        results[niche] = []

        for handle in handles:
            try:
                print(f"\nFetching {handle}...")
                channel = fetch_and_store_channel(
                    handle, max_videos
                )

                if channel:
                    db2 = SessionLocal()
                    try:
                        ch = db2.query(Channel).filter(
                            Channel.channel_id ==
                            channel.channel_id
                        ).first()
                        if ch:
                            ch.niche = niche
                            db2.commit()
                            print(f"  Niche set: {niche}")
                    except Exception as ne:
                        db2.rollback()
                        print(f"  Niche update error: {ne}")
                    finally:
                        db2.close()

                    results[niche].append({
                        "handle": handle,
                        "title": channel.title,
                        "status": "success"
                    })
                    completed += 1
                else:
                    results[niche].append({
                        "handle": handle,
                        "status": "failed"
                    })

                # Sleep between channels to
                # avoid rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"Error fetching {handle}: {e}")
                results[niche].append({
                    "handle": handle,
                    "status": f"error: {str(e)}"
                })
                continue

    print(f"\n{'='*50}")
    print(f"FETCH COMPLETE")
    print(f"{'='*50}")

    for niche, channels in results.items():
        print(f"\n{niche.upper()}:")
        for ch in channels:
            status = ch['status']
            title = ch.get('title', ch['handle'])
            print(f"  {title}: {status}")

    # Print DB summary
    db = SessionLocal()
    try:
        from sqlalchemy import func
        summary = db.query(
            Channel.niche,
            func.count(Channel.id).label('channels'),
            func.count(
                Video.id
            ).label('videos')
        ).join(
            Video,
            Video.channel_id == Channel.id,
            isouter=True
        ).group_by(Channel.niche).all()

        print(f"\n{'='*50}")
        print(f"DATABASE SUMMARY")
        print(f"{'='*50}")
        for row in summary:
            print(f"{row.niche}: "
                  f"{row.channels} channels, "
                  f"{row.videos} videos")
    finally:
        db.close()

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m backend.pipeline."
              "youtube_fetcher @handle")
        print("  python -m backend.pipeline."
              "youtube_fetcher all")
    elif sys.argv[1] == "all":
        fetch_all_training_channels()
    else:
        result = fetch_and_store_channel(sys.argv[1])
        if result:
            print(f"Success!")
        else:
            print("Failed to fetch channel")
