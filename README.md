# TubeIQ

TubeIQ is a full-stack ML application that helps you understand what makes a YouTube title work. Give it two titles and it predicts which one is more likely to perform better, based on patterns learned from real top-performing videos in that niche.

## What it does

- **Title Comparison** — Pit two titles against each other and get a prediction on which performs better, with a confidence score.
- **Similarity Search** — Find the most similar high-performing titles to yours in a given niche.
- **Content Gap Analysis** — Surface topic clusters in a niche that are underrepresented relative to demand.
- **Niche Insights** — Get aggregate patterns (title length, use of numbers, question framing, etc.) for what tends to work in a given niche.

## How it works

For each niche, TubeIQ builds four artifacts from real YouTube data pulled via the YouTube Data API:

1. A fitted **TF-IDF vectorizer** over that niche's title corpus
2. A **reference set** of High/Viral title vectors to compare new titles against
3. A **KMeans clusterer** for grouping titles by topic
4. A **blended Random Forest model** that scores head-to-head title comparisons

### Training

Production models are trained per niche in `backend/ml/train_models.py` on **all channels** in that niche (maximum data for the shipped model).

**Pair construction**

- Same channel only (controls for channel size / subscriber count)
- View-velocity ratio ≥ 2× (clear winner, less label noise)
- Age gap ≤ 120 days (blocks old-vs-new era comparisons)
- Pair orientation is randomized so labels are ~50/50

**Label**

- For pair `(A, B)`, label = 1 if `B` has higher `view_velocity` than `A`, else 0

**Features (25 selected pairwise diffs, B − A)**

Selected by Random Forest feature importance (weak keyword flags dropped):

- TF-IDF similarity difference vs High/Viral reference titles
- Title shape / style: length, word count, uppercase stats, digit count, punctuation, emoji, hashtags, quotes, year mentions, etc.
- Pre-publish video meta: planned duration, duration bucket, upload day, upload hour

**Leakage controls during training**

- `age_days` is **not** used as a feature — `view_velocity ≈ views / age`, so age alone can leak the label
- When computing TF-IDF similarity, each pair's own videos are excluded from the reference average so a video cannot match itself
- Post-publish engagement metrics (views, likes, comments) are never used as inputs at inference

**Classifier**

- Random Forest (`n_estimators=500`, `max_depth=14`, `min_samples_leaf=4`, `max_features=log2`)
- Trained with antisymmetry augmentation: both `(A,B)` and `(B,A)` with flipped labels
- Saved as `backend/saved_models/blend_{niche}.pkl`

### Evaluation

Held-out accuracy is measured by `blend_eval.py` with a **simple random 80/20 split over title pairs** (stratified, fixed seed):

1. Build all valid pairs for a niche
2. Train Random Forest on 80% of pairs
3. Score accuracy and ROC-AUC on the remaining 20%
4. Report per-niche and overall pooled results

**Result: ~79% average accuracy** (range about **72–86%** across niches) vs a **50%** random baseline. Overall ROC-AUC ≈ **0.88**.

Leakage-aware pair rules from training also apply at eval time: same channel, ≥2× velocity gap, ≤120-day age gap, and no `age_days` feature.

## Tech Stack

**Backend:** FastAPI, SQLAlchemy, Alembic, PostgreSQL  
**Frontend:** React  
**ML:** scikit-learn (TF-IDF, Random Forest, KMeans)  
**Data:** YouTube Data API v3

## API Overview

| Endpoint | Description |
|---|---|
| `POST /compare` | Compare two titles, returns predicted winner + confidence |
| `GET /similar` | Find similar high-performing titles in a niche |
| `GET /gaps` | Underrepresented topic clusters for a niche |
| `GET /insights` | Aggregate title patterns for a niche |

> Older `/predict` and `/improve` endpoints were deprecated in favor of the upgraded `/compare` endpoint.

## Project Status

TubeIQ is substantially complete and actively maintained as a portfolio project.

## Getting Started

```bash
# Clone the repo
git clone https://github.com/Krish-2114/TubeIQ.git
cd TubeIQ

# Backend setup
cd backend
pip install -r requirements.txt
alembic upgrade head
uvicorn main:app --reload

# Frontend setup
cd ../frontend
npm install
npm run dev
```

You'll need a YouTube Data API v3 key and a running PostgreSQL instance — see `.env.example` for required environment variables.

### Retrain / evaluate models

From the `tubeiq` project root:

```bash
# Train all niche models (TF-IDF, KMeans, blend RF)
python -m backend.ml.train_models

# Or retrain blend models only (requires existing TF-IDF + reference sets)
python retrain_blend.py

# Evaluate blend model (random 80/20 pair split)
python blend_eval.py
```
