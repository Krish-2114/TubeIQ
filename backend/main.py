import json
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


class CompareRequest(BaseModel):
    niche: str
    title_a: str
    title_b: str


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
    for path in MODELS_DIR.glob("tfidf_*.pkl"):
        niche = path.stem.replace("tfidf_", "")
        niches.append(niche)
    return sorted(niches)


def _cache_get(key: str, loader):
    if key not in _cache:
        _cache[key] = loader()
    return _cache[key]


def _model_path(name: str) -> Path:
    return MODELS_DIR / name


def _require_niche(niche: str) -> None:
    model_path = _model_path(f"tfidf_{niche}.pkl")
    ref_path = _model_path(f"reference_set_{niche}.pkl")
    if not model_path.exists() or not ref_path.exists():
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


def load_all_vecs(niche: str):
    """Precompute TF-IDF vectors for every title in the niche CSV."""
    def loader():
        tfidf = load_tfidf(niche)
        df = load_niche_csv(niche)
        titles = df["title"].fillna("").tolist()
        return tfidf.transform(titles)

    return _cache_get(f"all_vecs_{niche}", loader)


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
def predict_title(req: PredictRequest):
    _require_niche(req.niche)

    tfidf = load_tfidf(req.niche)
    ref = load_reference_set(req.niche)
    all_vecs = load_all_vecs(req.niche)

    vec = tfidf.transform([req.title])
    score = float(cosine_similarity(vec, ref["vectors"]).mean())
    all_scores = cosine_similarity(
        all_vecs, ref["vectors"]
    ).mean(axis=1)
    percentile = float((all_scores < score).mean() * 100)

    return {
        "score": score,
        "percentile": percentile,
    }


@app.post("/compare")
def compare_titles(req: CompareRequest):
    _require_niche(req.niche)

    tfidf = load_tfidf(req.niche)
    ref = load_reference_set(req.niche)

    vec_a = tfidf.transform([req.title_a])
    vec_b = tfidf.transform([req.title_b])
    score_a = float(
        cosine_similarity(vec_a, ref["vectors"]).mean()
    )
    score_b = float(
        cosine_similarity(vec_b, ref["vectors"]).mean()
    )

    winner = "title_a" if score_a > score_b else "title_b"
    return {
        "winner": winner,
        "title_a_score": score_a,
        "title_b_score": score_b,
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
