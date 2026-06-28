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
- `GET /evaluators/status` - show Gemini/MiniMax configuration and agreement logic.
- `GET /sessions/{session_id}/disagreement-seed` - fetch a stored self-improvement seed when models disagree.

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

## Dual LLM Evaluators

`llm_evaluators.py` runs Gemini and MiniMax as outbound API calls from the
DigitalOcean app. The calls are made in parallel with `asyncio.gather()` and an
8 second default timeout, matching the architecture diagram's MiniMax guidance.

- Gemini is the primary evaluator.
- MiniMax MoE is the independent cross-check evaluator.
- Agreement produces higher confidence.
- Disagreement or score delta above `0.3` produces an `uncertain` verdict.
- If either API is not configured or fails, the report falls back gracefully to
  the local feature-based verdict.

Reports include:

```json
{
  "verdict_document": {
    "overall_score": 0.66,
    "confidence": "high",
    "flagged_claims": [{ "claim": "Sunlight turns directly into glucose.", "reason": "incorrect" }],
    "behavioral_notes": {
      "rewatch": { "rate": 1.667, "note": "high_rewatch" },
      "hesitation": { "average_ms": 3940.0, "note": "high_hesitation" }
    }
  },
  "llm_evaluation": {
    "mode": "dual_external | gemini_only | minimax_only | local_fallback",
    "confidence": "low | medium | high",
    "score_delta": 0.12,
    "agreement": true,
    "evaluators": [{ "provider": "gemini" }, { "provider": "minimax" }]
  }
}
```

The verdict engine itself is pure Python. It performs no I/O while deciding the
final score, confidence, flagged claims, behavioral notes, or self-improvement
seed. After the document is produced, the FastAPI layer saves the report and,
when present, writes disagreement seeds to local JSON and MongoDB
`disagreements`.

Agreement logic:

- both evaluators agree on verdict, score, and flagged claims -> high confidence;
- claim-level disagreement -> `flagged_for_review` and a prompt-tuning seed;
- score delta above `0.3` -> `uncertain`;
- high rewatch or hesitation dampens confidence;
- strong drawing evidence can amplify confidence.

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
GEMINI_API_KEY=<your Gemini API key>
GEMINI_MODEL=gemini-3.5-flash
MINIMAX_API_KEY=<your MiniMax token>
MINIMAX_MODEL=MiniMax-Text-01
MINIMAX_API_URL=https://api.minimax.io/v1/text/chatcompletion_v2
EVALUATOR_TIMEOUT_SECONDS=8
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
