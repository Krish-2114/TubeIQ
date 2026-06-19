import joblib
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

FEATURE_COLUMNS = [
    "duration_seconds", "duration_encoded",
    "day_of_week", "hour_of_day",
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

import re
def extract_title_features(title):
    if not title:
        return {col: 0 for col in FEATURE_COLUMNS}
    title_lower = title.lower()
    return {
        "title_length": len(title),
        "word_count": len(title.split()),
        "has_number": int(
            bool(re.search(r'\d', title))
        ),
        "has_dollar": int("$" in title),
        "has_rupee": int(
            "₹" in title or
            "rs." in title_lower or
            "rupee" in title_lower
        ),
        "has_question": int("?" in title),
        "has_exclamation": int("!" in title),
        "has_vs": int(
            " vs" in title_lower or
            "vs " in title_lower
        ),
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

def parse_duration_encoded(seconds):
    if seconds < 120:   return 0
    if seconds < 600:   return 1
    if seconds < 1800:  return 2
    return 3

def predict_title(
    title, niche,
    duration_seconds=600,
    day_of_week=5,
    hour_of_day=18
):
    model_path = (
        f"backend/saved_models/"
        f"xgboost_{niche}.pkl"
    )
    le_path = (
        f"backend/saved_models/"
        f"label_encoder_{niche}.pkl"
    )
    tfidf_path = (
        f"backend/saved_models/"
        f"tfidf_{niche}.pkl"
    )

    if not os.path.exists(model_path):
        print(f"No model found for {niche}")
        return None

    model = joblib.load(model_path)
    le    = joblib.load(le_path)
    tfidf = joblib.load(tfidf_path)

    # Build feature row
    feats = extract_title_features(title)
    feats['duration_seconds'] = duration_seconds
    feats['duration_encoded'] = parse_duration_encoded(
        duration_seconds
    )
    feats['day_of_week'] = day_of_week
    feats['hour_of_day'] = hour_of_day

    X = pd.DataFrame([feats])[FEATURE_COLUMNS]

    # XGBoost prediction
    pred_encoded = model.predict(X)[0]
    pred_proba   = model.predict_proba(X)[0]
    pred_label   = le.inverse_transform(
        [pred_encoded]
    )[0]
    confidence   = round(
        float(max(pred_proba)) * 100, 1
    )

    # Class probabilities
    class_probs = {
        str(le.inverse_transform([i])[0]):
        round(float(p) * 100, 1)
        for i, p in enumerate(pred_proba)
    }

    # Similar titles from DB
    df_path = f"backend/data/{niche}_features.csv"
    similar_titles = []
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        high_df = df[
            df['performance_label'].isin(
                ['High', 'Viral']
            )
        ].copy()

        if len(high_df) > 0:
            title_vec = tfidf.transform([title])
            corpus_vec = tfidf.transform(
                high_df['title'].fillna('').tolist()
            )
            sims = cosine_similarity(
                title_vec, corpus_vec
            )[0]
            top_indices = sims.argsort()[-5:][::-1]

            for idx in top_indices:
                if sims[idx] > 0.05:
                    similar_titles.append({
                        "title": high_df.iloc[
                            idx
                        ]['title'],
                        "views": int(
                            high_df.iloc[idx][
                                'view_count'
                            ]
                        ),
                        "label": high_df.iloc[
                            idx
                        ]['performance_label'],
                        "similarity": round(
                            float(sims[idx]), 3
                        )
                    })

    # What signals were detected
    signals = []
    if feats['has_number']:
        signals.append("Has number in title")
    if feats['has_dollar']:
        signals.append("Has dollar amount")
    if feats['has_rupee']:
        signals.append("Has rupee/price point")
    if feats['has_question']:
        signals.append("Question format")
    if feats['has_vs']:
        signals.append("Comparison/VS format")
    if feats['has_challenge']:
        signals.append("Challenge format")
    if feats['has_best']:
        signals.append("Contains 'best'")
    if feats['has_every']:
        signals.append("Contains 'every'")
    if feats['has_last']:
        signals.append("Contains 'last'")
    if feats['has_win']:
        signals.append("Contains 'win'")
    if feats['title_length'] > 70:
        signals.append("Title might be too long")
    if feats['word_count'] < 3:
        signals.append("Title might be too short")

    return {
        "title": title,
        "niche": niche,
        "prediction": pred_label,
        "confidence": confidence,
        "class_probabilities": class_probs,
        "signals_detected": signals,
        "similar_high_performing": similar_titles,
        "input_features": {
            "duration_seconds": duration_seconds,
            "day_of_week": day_of_week,
            "hour_of_day": hour_of_day
        }
    }

def run_test(niche, test_titles):
    print(f"\n{'='*55}")
    print(f"TESTING: {niche.upper()} NICHE")
    print(f"{'='*55}")

    for title in test_titles:
        result = predict_title(title, niche)
        if not result:
            continue

        print(f"\nTitle: {title}")
        print(f"Prediction: {result['prediction']} "
              f"({result['confidence']}% confidence)")
        print(f"Probabilities: "
              f"{result['class_probabilities']}")

        if result['signals_detected']:
            print(f"Signals: "
                  f"{', '.join(result['signals_detected'])}")

        if result['similar_high_performing']:
            print(f"Similar high-performing titles:")
            for s in result[
                'similar_high_performing'
            ][:3]:
                print(f"  • {s['title'][:60]} "
                      f"— {s['views']:,} views")
        print("-"*55)

if __name__ == "__main__":
    import sys

    niche = sys.argv[1] if len(sys.argv) > 1 \
            else "tech"

    test_titles_by_niche = {
        "tech": [
            "I Tested Every Budget Laptop Under ₹30,000",
            "iPhone 17 Pro vs Samsung Galaxy S26 Ultra",
            "Best Wireless Earbuds of 2026",
            "Why I Switched From Mac to Windows",
            "The Only Laptop You Need in 2026"
        ],
        "entertainment": [
            "Last To Leave Wins $100,000",
            "I Survived 24 Hours In The Amazon",
            "We Played Among Us In Real Life",
            "Giving $10,000 To Random Strangers",
            "World's Largest Pizza Challenge"
        ],
        "gaming": [
            "I Spent 100 Days In Minecraft",
            "Roasting Every Type of Gamer",
            "GTA 6 First Look — Everything You Need To Know",
            "Playing PUBG With Strangers Goes Wrong",
            "Last To Stop Playing Wins $10,000"
        ],
        "cooking": [
            "Gordon Ramsay Reacts To My Cooking",
            "I Made A 5 Star Burger At Home",
            "Every Michelin Star Dish In One Video",
            "The Perfect Biryani Recipe — Restaurant Style",
            "30 Minute Meals That Actually Taste Amazing"
        ],
        "finance": [
            "How I Made ₹1 Lakh From Stocks In 30 Days",
            "The Best Mutual Funds for Beginners 2026",
            "Why 90% Of People Fail At Investing",
            "I Tried Every Credit Card for 1 Year",
            "How To Retire Early With ₹50 Lakhs"
        ],
        "fitness": [
            "The Only Ab Workout You Need",
            "I Trained Like The Rock For 30 Days",
            "Best vs Worst Exercises For Bigger Arms",
            "Why You Are Not Building Muscle",
            "Full Body Workout In 20 Minutes"
        ],
        "education": [
            "What Happens When A Black Hole Dies",
            "Why We Have Not Found Aliens Yet",
            "How The Universe Will End",
            "The Science Behind Sleep",
            "What If Earth Had Two Moons"
        ]
    }

    titles = test_titles_by_niche.get(
        niche,
        test_titles_by_niche["tech"]
    )
    run_test(niche, titles)
