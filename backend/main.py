import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from backend.ml.train_models import clean_title

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "saved_models"
DATA_DIR = BASE_DIR / "data"
INSIGHTS_DIR = BASE_DIR / "insights"

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
    "has_how", "has_why", "has_what",
]

_cache: dict = {}

app = FastAPI(title="TubeIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    title: str
    niche: str
    duration_seconds: int
    upload_day: int
    upload_hour: int


class SimilarRequest(BaseModel):
    title: str
    niche: str
    top_n: int = 5


class GapsRequest(BaseModel):
    niche: str


class ChannelAnalyzeRequest(BaseModel):
    identifier: str


class ImproveTitleRequest(BaseModel):
    title: str
    niche: str


def get_available_niches() -> list[str]:
    niches = []
    for path in MODELS_DIR.glob("xgboost_*.pkl"):
        niche = path.stem.replace("xgboost_", "")
        niches.append(niche)
    return sorted(niches)


def _cache_get(key: str, loader):
    if key not in _cache:
        _cache[key] = loader()
    return _cache[key]


def _model_path(name: str) -> Path:
    return MODELS_DIR / name


def _require_niche(niche: str) -> None:
    model_path = _model_path(f"xgboost_{niche}.pkl")
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for niche '{niche}'",
        )


def load_xgboost(niche: str):
    return _cache_get(
        f"xgboost_{niche}",
        lambda: joblib.load(_model_path(f"xgboost_{niche}.pkl")),
    )


def load_label_encoder(niche: str):
    return _cache_get(
        f"label_encoder_{niche}",
        lambda: joblib.load(
            _model_path(f"label_encoder_{niche}.pkl")
        ),
    )


def load_tfidf(niche: str):
    path = _model_path(f"tfidf_{niche}.pkl")
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No TF-IDF model found for niche '{niche}'",
        )
    return _cache_get(
        f"tfidf_{niche}",
        lambda: joblib.load(path),
    )


def load_kmeans(niche: str):
    path = _model_path(f"kmeans_{niche}.pkl")
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No K-Means model found for niche '{niche}'",
        )
    return _cache_get(
        f"kmeans_{niche}",
        lambda: joblib.load(path),
    )


def load_clusters(niche: str) -> dict:
    path = _model_path(f"clusters_{niche}.json")
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No cluster data found for niche '{niche}'",
        )

    def loader():
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    return _cache_get(f"clusters_{niche}", loader)


def load_niche_csv(niche: str) -> pd.DataFrame:
    path = DATA_DIR / f"{niche}_features.csv"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No feature data found for niche '{niche}'",
        )
    return _cache_get(
        f"csv_{niche}",
        lambda: pd.read_csv(path),
    )


def load_insights(niche: str) -> dict:
    path = INSIGHTS_DIR / f"{niche}_insights.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No insights found for niche '{niche}'",
        )

    def loader():
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    return _cache_get(f"insights_{niche}", loader)


def parse_duration_encoded(seconds: int) -> int:
    if seconds < 120:
        return 0
    if seconds < 600:
        return 1
    if seconds < 1800:
        return 2
    return 3


def extract_title_features(title: str) -> dict:
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
        "has_number": int(bool(re.search(r"\d", title))),
        "has_dollar": int("$" in title),
        "has_rupee": int(
            "₹" in title
            or "rs." in title_lower
            or "rupee" in title_lower
        ),
        "has_question": int("?" in title),
        "has_exclamation": int("!" in title),
        "has_vs": int(
            " vs" in title_lower or "vs " in title_lower
        ),
        "has_challenge": int("challenge" in title_lower),
        "has_hours": int(
            "hours" in title_lower or "hour" in title_lower
        ),
        "has_last": int("last" in title_lower),
        "has_win": int(
            " win" in title_lower or "wins" in title_lower
        ),
        "has_free": int("free" in title_lower),
        "has_extreme": int("extreme" in title_lower),
        "has_best": int("best" in title_lower),
        "has_worst": int("worst" in title_lower),
        "has_every": int("every" in title_lower),
        "has_first": int("first" in title_lower),
        "has_only": int("only" in title_lower),
        "has_never": int("never" in title_lower),
        "has_days": int(
            "days" in title_lower or "day" in title_lower
        ),
        "has_review": int("review" in title_lower),
        "has_how": int(title_lower.startswith("how")),
        "has_why": int(title_lower.startswith("why")),
        "has_what": int(title_lower.startswith("what")),
    }


def build_feature_row(
    title: str,
    duration_seconds: int,
    upload_day: int,
    upload_hour: int,
) -> pd.DataFrame:
    feats = extract_title_features(title)
    feats["duration_seconds"] = duration_seconds
    feats["duration_encoded"] = parse_duration_encoded(
        duration_seconds
    )
    feats["day_of_week"] = upload_day
    feats["hour_of_day"] = upload_hour
    return pd.DataFrame([feats])[FEATURE_COLUMNS]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "niches": get_available_niches(),
    }


@app.get("/")
def root():
    return {"message": "TubeIQ API is running"}


@app.post("/predict")
def predict(body: PredictRequest):
    _require_niche(body.niche)

    model = load_xgboost(body.niche)
    label_encoder = load_label_encoder(body.niche)

    X = build_feature_row(
        body.title,
        body.duration_seconds,
        body.upload_day,
        body.upload_hour,
    )

    pred_encoded = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    label = str(label_encoder.inverse_transform([pred_encoded])[0])
    confidence = round(float(max(pred_proba)), 4)

    probabilities = {
        str(label_encoder.inverse_transform([i])[0]): round(
            float(p), 4
        )
        for i, p in enumerate(pred_proba)
    }

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
    }


@app.post("/similar")
def similar(body: SimilarRequest):
    _require_niche(body.niche)

    tfidf = load_tfidf(body.niche)
    df = load_niche_csv(body.niche)

    high_df = df[
        df["performance_label"].isin(["High", "Viral"])
    ].copy()

    if high_df.empty:
        return {"similar_titles": []}

    query_vec = tfidf.transform([body.title])
    corpus_vec = tfidf.transform(
        high_df["title"].fillna("").tolist()
    )
    sims = cosine_similarity(query_vec, corpus_vec)[0]

    high_df = high_df.copy()
    high_df["similarity_score"] = sims
    high_df = high_df.sort_values(
        "similarity_score", ascending=False
    )

    top_n = max(1, body.top_n)
    results = []
    for _, row in high_df.head(top_n).iterrows():
        if row["similarity_score"] <= 0:
            continue
        results.append({
            "title": row["title"],
            "channel": row.get("channel_title", ""),
            "views": int(row["view_count"]),
            "performance_label": row["performance_label"],
            "similarity_score": round(
                float(row["similarity_score"]), 4
            ),
        })

    return {"similar_titles": results}


@app.get("/insights/{niche}")
def insights(niche: str):
    return load_insights(niche)


@app.post("/gaps")
def gaps(body: GapsRequest):
    _require_niche(body.niche)

    tfidf = load_tfidf(body.niche)
    kmeans = load_kmeans(body.niche)
    cluster_keywords = load_clusters(body.niche)
    df = load_niche_csv(body.niche)

    cleaned_titles = df.apply(
        lambda row: clean_title(
            row.get("title", ""),
            row.get("channel_title", ""),
        ),
        axis=1,
    )
    vectors = tfidf.transform(cleaned_titles.fillna("").tolist())
    cluster_labels = kmeans.predict(vectors)

    df = df.copy()
    df["cluster_id"] = cluster_labels

    cluster_avgs = (
        df.groupby("cluster_id")["view_count"]
        .mean()
        .to_dict()
    )
    overall_avg = float(np.mean(list(cluster_avgs.values())))

    gap_results = []
    for cluster_id, avg_views in cluster_avgs.items():
        if avg_views >= overall_avg:
            continue

        opportunity_score = round(
            float((overall_avg - avg_views) / max(overall_avg, 1)),
            4,
        )
        keywords = cluster_keywords.get(str(cluster_id), [])
        gap_results.append({
            "cluster_id": int(cluster_id),
            "keywords": keywords,
            "avg_views": round(float(avg_views), 2),
            "opportunity_score": opportunity_score,
        })

    gap_results.sort(
        key=lambda item: item["opportunity_score"],
        reverse=True,
    )

    return {"gaps": gap_results}


@app.post("/channel/analyze")
def channel_analyze(body: ChannelAnalyzeRequest):
    from backend.pipeline.channel_analyzer import analyze_channel

    identifier = body.identifier.strip()
    if not identifier:
        raise HTTPException(
            status_code=400,
            detail="Channel identifier is required",
        )

    result = analyze_channel(identifier)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/improve")
def improve_title_endpoint(body: ImproveTitleRequest):
    from backend.ml.title_improver import improve_title

    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")

    _require_niche(body.niche)
    result = improve_title(title, body.niche)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
