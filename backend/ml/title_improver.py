import joblib
import json
import pandas as pd
import re
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent

STOP_WORDS = {
    'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or',
    'but', 'is', 'i', 'my', 'we', 'you', 'it', 'this', 'that', 'with',
    'from', 'by', 'your', 'our', 'their', 'what', 'how', 'why', 'when',
}


def _title_words(title):
    return [
        re.sub(r'[^\w]', '', w.lower())
        for w in title.lower().split()
        if re.sub(r'[^\w]', '', w.lower()) not in STOP_WORDS
        and len(re.sub(r'[^\w]', '', w.lower())) > 2
    ]


def _top_titles_for_keyword(high_df, keyword, limit=3):
    if not keyword or high_df.empty:
        return []

    mask = high_df['title'].fillna('').str.lower().str.contains(
        re.escape(keyword.lower()),
        regex=True,
        na=False,
    )
    matches = high_df[mask].nlargest(limit, 'view_count')
    results = []
    for _, row in matches.iterrows():
        results.append({
            "title": row['title'],
            "views": int(row['view_count']),
            "label": row['performance_label'],
        })
    return results


def improve_title(title, niche):
    insights_path = BASE_DIR / "insights" / f"{niche}_insights.json"
    if not insights_path.exists():
        return {"error": f"No insights for {niche}"}

    with open(insights_path) as f:
        insights = json.load(f)

    df_path = BASE_DIR / "data" / f"{niche}_features.csv"
    if not df_path.exists():
        return {"error": "No training data found"}

    df = pd.read_csv(df_path)
    top_keywords = list(insights.get('top_keywords', {}).keys())[:15]
    title_lower = title.lower()
    title_words = _title_words(title)

    high_df = df[df['performance_label'].isin(['High', 'Viral'])].copy()

    # Keywords to look up: words from the title + niche top keywords already in title
    matched_keywords = []
    for word in title_words:
        if word not in matched_keywords:
            matched_keywords.append(word)

    for kw in top_keywords[:10]:
        if kw in title_lower and kw not in matched_keywords:
            matched_keywords.append(kw)

    if not matched_keywords and top_keywords:
        matched_keywords = top_keywords[:3]

    top_titles_by_keyword = []
    for kw in matched_keywords[:5]:
        titles = _top_titles_for_keyword(high_df, kw, limit=3)
        if titles:
            top_titles_by_keyword.append({
                "keyword": kw,
                "titles": titles,
            })

    similar = []
    tfidf_path = BASE_DIR / "saved_models" / f"tfidf_{niche}.pkl"
    if tfidf_path.exists() and len(high_df) > 0:
        tfidf = joblib.load(tfidf_path)
        title_vec = tfidf.transform([title])
        corpus_vec = tfidf.transform(high_df['title'].fillna('').tolist())
        sims = cosine_similarity(title_vec, corpus_vec)[0]
        top_idx = sims.argsort()[-5:][::-1]
        for idx in top_idx:
            if sims[idx] > 0.05:
                similar.append({
                    "title": high_df.iloc[idx]['title'],
                    "views": int(high_df.iloc[idx]['view_count']),
                    "label": high_df.iloc[idx]['performance_label'],
                    "similarity_score": round(float(sims[idx]), 4),
                })

    return {
        "original_title": title,
        "niche": niche,
        "matched_keywords": matched_keywords[:5],
        "top_titles_by_keyword": top_titles_by_keyword,
        "inspiration_titles": similar,
        "top_niche_keywords": top_keywords[:10],
    }


if __name__ == "__main__":
    import sys

    title_arg = sys.argv[1] if len(sys.argv) > 1 else "Laptop Review"
    niche_arg = sys.argv[2] if len(sys.argv) > 2 else "tech"
    result = improve_title(title_arg, niche_arg)
    print(json.dumps(result, indent=2, default=str))
