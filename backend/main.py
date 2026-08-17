import json
import re
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

from backend.ml.train_models import (
    blend_pair_feature_vector,
    extract_blend_side_features,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "saved_models"
DATA_DIR = BASE_DIR / "data"
INSIGHTS_DIR = BASE_DIR / "insights"

_cache: dict = {}

app = FastAPI(title="TubeIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompareRequest(BaseModel):
    niche: str
    title_a: str
    title_b: str
    # Optional pre-publish metadata — defaults keep title-only compares working
    duration_seconds_a: Optional[float] = Field(default=None)
    duration_seconds_b: Optional[float] = Field(default=None)
    day_of_week_a: Optional[int] = Field(default=None, ge=0, le=6)
    day_of_week_b: Optional[int] = Field(default=None, ge=0, le=6)
    hour_of_day_a: Optional[int] = Field(default=None, ge=0, le=23)
    hour_of_day_b: Optional[int] = Field(default=None, ge=0, le=23)


class SimilarRequest(BaseModel):
    title: str
    niche: str
    top_n: int = 5


class GapsRequest(BaseModel):
    niche: str


class ChannelAnalyzeRequest(BaseModel):
    identifier: str


def get_available_niches() -> list[str]:
    niches = []
    for path in MODELS_DIR.glob("tfidf_*.pkl"):
        niche = path.stem.replace("tfidf_", "")
        niches.append(niche)
    return sorted(niches)


def _cache_get(key: str, loader):
    if key not in _cache:
        _cache[key] = loader()
    return _cache[key]


def _clear_cache() -> int:
    n = len(_cache)
    _cache.clear()
    return n


def _model_path(name: str) -> Path:
    return MODELS_DIR / name


def struct_features(title: str) -> dict:
    """Deprecated alias — use extract_blend_side_features."""
    return extract_blend_side_features(title)


def _require_niche(niche: str) -> None:
    model_path = _model_path(f"tfidf_{niche}.pkl")
    ref_path = _model_path(f"reference_set_{niche}.pkl")
    blend_path = _model_path(f"blend_{niche}.pkl")
    if (
        not model_path.exists()
        or not ref_path.exists()
        or not blend_path.exists()
    ):
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for niche '{niche}'",
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


def load_reference_set(niche: str):
    path = _model_path(f"reference_set_{niche}.pkl")
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No reference set found for niche '{niche}'"
            ),
        )
    return _cache_get(
        f"reference_set_{niche}",
        lambda: joblib.load(path),
    )


def load_blend(niche: str):
    path = _model_path(f"blend_{niche}.pkl")
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No blend model found for niche '{niche}'",
        )
    return _cache_get(
        f"blend_{niche}",
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


def load_gaps_csv(niche: str) -> list:
    path = MODELS_DIR / f"gaps_{niche}.csv"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No gaps data found for niche '{niche}'",
        )

    def loader():
        df = pd.read_csv(path)
        results = []
        for _, row in df.iterrows():
            raw_keywords = row.get("keywords", "")
            if pd.isna(raw_keywords) or str(raw_keywords).strip() == "":
                keywords = []
            else:
                keywords = [
                    k for k in str(raw_keywords).split(";") if k
                ]
            results.append({
                "cluster_id": int(row["cluster_id"]),
                "keywords": keywords,
                "avg_views": float(row["avg_views"]),
                "opportunity_score": float(row["opportunity_score"]),
            })
        return results

    return _cache_get(f"gaps_{niche}", loader)


def _reference_metadata(niche: str, ref: dict) -> tuple[list, list, list]:
    """channel / views / label aligned to reference rows."""
    n = len(ref.get("titles") or [])
    channels = ref.get("channel_titles")
    views = ref.get("view_counts")
    labels = ref.get("performance_labels")
    if (
        channels is not None
        and views is not None
        and labels is not None
        and len(channels) == n
        and len(views) == n
        and len(labels) == n
    ):
        return list(channels), list(views), list(labels)

    video_ids = ref.get("video_ids") or []
    if len(video_ids) != n:
        raise HTTPException(
            status_code=500,
            detail=(
                f"reference_set_{niche}.pkl missing metadata "
                f"and video_ids length mismatch"
            ),
        )

    df = load_niche_csv(niche)
    by_id = {
        str(row["video_id"]): row
        for _, row in df.iterrows()
    }
    channels_out = []
    views_out = []
    labels_out = []
    for vid in video_ids:
        row = by_id.get(str(vid))
        if row is None:
            channels_out.append("")
            views_out.append(0)
            labels_out.append("")
        else:
            channels_out.append(row.get("channel_title", "") or "")
            views_out.append(row["view_count"])
            labels_out.append(row["performance_label"])
    return channels_out, views_out, labels_out


@app.get("/health")
def health():
    return {
        "status": "ok",
        "niches": get_available_niches(),
    }


@app.get("/")
def root():
    return {"message": "TubeIQ API is running"}


@app.post("/admin/clear-cache")
def clear_cache():
    cleared = _clear_cache()
    return {"status": "ok", "cleared_entries": cleared}


@app.post("/compare")
def compare_titles(req: CompareRequest):
    _require_niche(req.niche)

    tfidf = load_tfidf(req.niche)
    ref = load_reference_set(req.niche)
    blend = load_blend(req.niche)

    vec_a = tfidf.transform([req.title_a])
    vec_b = tfidf.transform([req.title_b])
    score_a = float(
        cosine_similarity(vec_a, ref["vectors"]).mean()
    )
    score_b = float(
        cosine_similarity(vec_b, ref["vectors"]).mean()
    )

    feats_a = extract_blend_side_features(
        req.title_a,
        duration_seconds=req.duration_seconds_a,
        day_of_week=req.day_of_week_a,
        hour_of_day=req.hour_of_day_a,
    )
    feats_b = extract_blend_side_features(
        req.title_b,
        duration_seconds=req.duration_seconds_b,
        day_of_week=req.day_of_week_b,
        hour_of_day=req.hour_of_day_b,
    )
    x = [
        blend_pair_feature_vector(
            score_a, score_b, feats_a, feats_b
        )
    ]

    model = blend["model"]
    # Legacy pickles used a StandardScaler; new RF pickles do not
    if "scaler" in blend and blend["scaler"] is not None:
        x = blend["scaler"].transform(x)

    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    winner = "title_b" if pred == 1 else "title_a"
    confidence = round(float(max(proba)), 4)

    return {
        "winner": winner,
        "title_a_score": score_a,
        "title_b_score": score_b,
        "confidence": confidence,
        "model_type": blend.get("model_type", "unknown"),
    }


@app.post("/similar")
def similar(body: SimilarRequest):
    _require_niche(body.niche)

    tfidf = load_tfidf(body.niche)
    ref = load_reference_set(body.niche)
    titles = ref.get("titles") or []
    corpus_vec = ref.get("vectors")
    if not titles or corpus_vec is None or corpus_vec.shape[0] == 0:
        return {"similar_titles": []}

    channels, views, labels = _reference_metadata(body.niche, ref)

    query_vec = tfidf.transform([body.title])
    sims = cosine_similarity(query_vec, corpus_vec)[0]

    high_df = pd.DataFrame({
        "title": titles,
        "channel_title": channels,
        "view_count": views,
        "performance_label": labels,
        "similarity_score": sims,
    })
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
    return {"gaps": load_gaps_csv(body.niche)}


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
