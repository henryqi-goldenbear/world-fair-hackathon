# FastAPI Ingestor Phase

Public-facing ingestion API for FerbAI session outputs.

## Endpoints

- `GET /health` - DigitalOcean App Platform health check.
- `POST /sessions` - ingest a full FerbAI session payload.
- `POST /sessions/{session_id}/event` - stream incremental interaction events.
- `GET /sessions/{session_id}/report` - fetch the generated verdict report.
- `GET /demo` - run a full synthetic agent-student session through ingest, extraction, persistence, and report generation.
- `GET /generation/status` - inspect the continuous FerbAI agent-student generator.
- `POST /generation/start?interval_seconds=15&limit=0` - continuously generate sessions. `limit=0` means keep running.
- `POST /generation/stop` - stop the continuous generator.

## Feature Extraction

`feature_extraction.py` runs inside the same DigitalOcean container as the API.
It produces a `FeatureBundle` that is attached to every verdict report and saved
with the report document:

- `claims`: sentence-level transcript claims tagged as verifiable or not.
- `behavior.rewatch_rate`: rewatch events per minute.
- `behavior.hesitation_ms`: average long pause before student responses.
- `behavior.drawing_score`: 0-1 complexity score from strokes and SVG coverage.

The extractor uses a lightweight spaCy English pipeline with an EntityRuler plus
rule-based filters. No GPU or external service is required.

## Runtime

```powershell
$env:PYTHONPATH = "$PWD\.deps"
python -m uvicorn fastapi_session_receiver:app --host 127.0.0.1 --port 8899
```

## Continuous Agent Generation

The ingestor can continuously produce FerbAI synthetic agent-student sessions
inside the same DigitalOcean container. Generated sessions use the normal
`POST /sessions` storage path, replay a few interaction events through the
event stream storage, run feature extraction, and save a report.

Useful demo calls:

```text
GET  /generation/status
POST /generation/start?interval_seconds=10&limit=0
POST /generation/start?interval_seconds=1&limit=3
POST /generation/stop
```

Use `limit=0` for a persistent synthetic swarm. Use a small positive `limit`
when you want a bounded smoke test during deployment checks.

## DigitalOcean App Platform

This repo has a root `Dockerfile`. Push the repo to GitHub and create a DigitalOcean App Platform app from it.

Suggested environment variables:

```text
PORT=8080
DATA_DIR=/app/ingestor_data
MONGODB_URI=<your MongoDB Atlas URI>
MONGODB_DB=ferbai
LOG_LEVEL=INFO
```

If `MONGODB_URI` is absent, the app still works with local JSON persistence. When MongoDB is configured, `sessions`, `events`, and `reports` are upserted into MongoDB.

## Verified Local Flow

The local proof exercised:

```text
GET /health
POST /sessions
POST /sessions/{id}/event
GET /sessions/{id}/report
GET /demo
```

After streaming an extra event, the report showed:

```json
{
  "verdict": "needs_targeted_support",
  "score": 0.62,
  "evidence": {
    "transcript_turns": 8,
    "student_turns": 4,
    "event_count": 10,
    "drawing_count": 1,
    "verifiable_claim_count": 2,
    "rewatch_rate": 1.667,
    "hesitation_ms": 3940.0,
    "drawing_score": 0.604,
    "human_student": false
  },
  "features": {
    "claims": [{ "claim": "Chlorophyll captures light energy.", "sentence_idx": 7, "verifiable": true }],
    "behavior": { "rewatch_rate": 1.667, "hesitation_ms": 3940.0, "drawing_score": 0.604 }
  }
}
```

## Demo URL

During judging, hit:

```text
https://<your-app>.ondigitalocean.app/demo
```

It returns verdict JSON and persists the generated session/report. With `MONGODB_URI` set, the response includes `saved_to_mongodb: true`.
