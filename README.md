# FerbAI Synthetic Student Swarm

FerbAI is a DigitalOcean-hosted FastAPI system that ingests tutoring-session
outputs from agent students, extracts learning signals, sends the session to
Gemini and MiniMax as parallel evaluators, and produces a final verdict.

The current live app is:

```text
https://ferbai-fastapi-ingestor-ej6yt.ondigitalocean.app
```

## What Is Running

The system has five main parts:

- Synthetic agent-student sessions generate FerbAI-style transcripts, events,
  drawings, and metadata.
- FastAPI ingests full sessions and streamed interaction events.
- `feature_extraction.py` extracts claims, rewatch rate, hesitation, and drawing
  complexity.
- Gemini and MiniMax evaluate the same session in parallel from the
  DigitalOcean container.
- The verdict engine checks agreement, flags disagreement, adjusts confidence
  with behavioral signals, and stores disagreement cases as self-improvement
  seeds.

No human students are accepted. Sessions must have:

```json
{
  "student_kind": "agent_student",
  "agent_generated": true,
  "human_student": false
}
```

## Human-Readable Views

Open these in a browser:

```text
https://ferbai-fastapi-ingestor-ej6yt.ondigitalocean.app/
https://ferbai-fastapi-ingestor-ej6yt.ondigitalocean.app/live
https://ferbai-fastapi-ingestor-ej6yt.ondigitalocean.app/generation/status.txt
https://ferbai-fastapi-ingestor-ej6yt.ondigitalocean.app/demo.txt
```

The `/live` dashboard auto-refreshes and shows:

- whether continuous generation is running;
- generated session count;
- latest verdict and score;
- latest session/report links;
- evaluator agreement;
- confidence;
- behavioral notes;
- flagged claims;
- self-improvement seed status.

## JSON API Views

Use these when you want raw machine-readable data:

```text
GET /api
GET /health
GET /generation/status
GET /evaluators/status
GET /demo
GET /sessions/{session_id}/report
GET /sessions/{session_id}/disagreement-seed
```

## Start, Stop, And Watch Progress

Set the base URL once:

```bash
BASE="https://ferbai-fastapi-ingestor-ej6yt.ondigitalocean.app"
```

Start continuous generation:

```bash
curl -X POST "$BASE/generation/start?interval_seconds=120&limit=0&stream_events=2"
```

Run a short bounded smoke test:

```bash
curl -X POST "$BASE/generation/start?interval_seconds=5&limit=3&stream_events=2"
```

Stop generation:

```bash
curl -X POST "$BASE/generation/stop"
```

View human-readable progress:

```bash
curl "$BASE/generation/status.txt"
```

View JSON progress:

```bash
curl "$BASE/generation/status"
```

Run one full synthetic session and read the result:

```bash
curl "$BASE/demo.txt"
```

Fetch the latest report:

1. Open or curl `/generation/status.txt`.
2. Copy the latest readable report path, for example:

```text
/sessions/continuous_20260628000331_1/report.txt
```

3. Fetch it:

```bash
curl "$BASE/sessions/continuous_20260628000331_1/report.txt"
```

## Local Development

Install dependencies:

```bash
python -m pip install -r requirements-fastapi.txt
```

Run the app locally:

```bash
uvicorn fastapi_session_receiver:app --host 127.0.0.1 --port 8899
```

Then open:

```text
http://127.0.0.1:8899/live
```

Run tests:

```bash
python -m unittest test_feature_extraction.py test_llm_evaluators.py
```

## Verdict Engine

The verdict engine produces a `verdict_document`:

```json
{
  "overall_score": 0.66,
  "confidence": "low",
  "flagged_claims": [{ "claim": "...", "reason": "..." }],
  "behavioral_notes": {
    "rewatch": { "rate": 1.667, "note": "high_rewatch" },
    "hesitation": { "average_ms": 3940.0, "note": "high_hesitation" },
    "drawing": { "score": 0.604, "note": "strong_board_work" }
  }
}
```

Agreement rules:

- both models agree -> high confidence;
- claim disagreement -> `flagged_for_review`;
- score delta above `0.3` -> `uncertain`;
- high rewatch or hesitation lowers confidence;
- strong drawing work can raise confidence;
- disagreement cases become self-improvement seeds.

Judge pitch:

```text
Two independent LLMs evaluate each session. When they disagree, FerbAI learns:
those cases become the training signal for continuous prompt improvement.
```
