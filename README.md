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



### Evaluation

Model accuracy is validated with **leave-one-channel-out (LOCO) cross-validation**, holding out entire channels rather than random title pairs, to make sure the model generalizes to creators it hasn't seen. Pairs are also restricted to a 120-day age gap to eliminate temporal leakage (title style drifts platform-wide over time, and view velocity naturally favors newer videos).

**Result: ~79% Average Accuracy 

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


