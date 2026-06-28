from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import suppress
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Response, status
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from atlas_integration import (
    atlas_search_index_definitions,
    ensure_atlas_indexes,
    load_personas,
    prepare_disagreement_document,
    prepare_session_document,
    prepare_verdict_document,
    seed_personas,
    swarm_analytics,
    verdict_drift,
)
from feature_extraction import dump_model, extract_feature_bundle
from llm_evaluators import build_verdict_document, evaluate_with_external_llms


DATA_DIR = Path(os.getenv("DATA_DIR", "ingestor_data"))
SESSIONS_DIR = DATA_DIR / "sessions"
REPORTS_DIR = DATA_DIR / "reports"
DISAGREEMENTS_DIR = DATA_DIR / "disagreements"
STREAM_EVENTS_DIR = DATA_DIR / "stream_events"
EVENTS_PATH = DATA_DIR / "events.jsonl"
RECEIVED_PATH = Path("received_ferbai_sessions.jsonl")
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "ferbai")
REPORT_WEBHOOK_URL = os.getenv("REPORT_WEBHOOK_URL", "")
ATLAS_BOOTSTRAP_ON_STARTUP = os.getenv("ATLAS_BOOTSTRAP_ON_STARTUP", "true").lower() == "true"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        for key in ("session_id", "event_type", "route", "status_code", "duration_ms"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), handlers=[handler], force=True)
logger = logging.getLogger("ferbai.ingestor")


class TranscriptTurn(BaseModel):
    t: int | float = Field(..., description="Milliseconds from session start")
    speaker: str
    text: str


class InteractionEvent(BaseModel):
    type: str
    t: int | float = Field(..., description="Milliseconds from session start")
    payload: dict[str, Any] = Field(default_factory=dict)


class SessionMetadata(BaseModel):
    duration_ms: int | float
    subject: str
    student_id: str
    student_kind: str = "agent_student"
    agent_generated: bool = True
    human_student: bool = False
    recording_id: str | None = None
    understanding_band: str | None = None
    confidence: float | None = None


class FerbAISessionOutput(BaseModel):
    session_id: str
    timestamp: str
    transcript: list[TranscriptTurn]
    events: list[InteractionEvent]
    drawings: list[str]
    metadata: SessionMetadata


class StreamEvent(BaseModel):
    type: str
    t: int | float
    payload: dict[str, Any] = Field(default_factory=dict)


class VerdictReport(BaseModel):
    session_id: str
    status: str
    verdict: str
    score: float
    generated_at: str
    summary: str
    evidence: dict[str, Any]
    saved_to_mongodb: bool = False


app = FastAPI(
    title="FerbAI FastAPI Ingestor",
    description="Public-facing ingestion API for FerbAI session outputs.",
    version="0.7.0",
)

_mongo_client: Any = None
_mongo_failed = False
_mongo_last_error: str | None = None
_generation_task: asyncio.Task[None] | None = None
_generation_started_at: str | None = None
_generation_last_session_id: str | None = None
_generation_last_report: dict[str, Any] | None = None
_generation_count = 0
_generation_stop_reason = "not_started"
_generation_interval_seconds = float(os.getenv("FERBAI_GENERATION_INTERVAL_SECONDS", "15"))


def model_dump(model: BaseModel) -> dict[str, Any]:
    return dump_model(model)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_dirs() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DISAGREEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    STREAM_EVENTS_DIR.mkdir(parents=True, exist_ok=True)


def session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def report_path(session_id: str) -> Path:
    return REPORTS_DIR / f"{session_id}.json"


def disagreement_path(session_id: str) -> Path:
    return DISAGREEMENTS_DIR / f"{session_id}.json"


def stream_events_path(session_id: str) -> Path:
    return STREAM_EVENTS_DIR / f"{session_id}.jsonl"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


async def write_json(path: Path, record: dict[str, Any]) -> None:
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_text, json.dumps(record, indent=2), "utf-8")


async def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(await asyncio.to_thread(path.read_text, "utf-8"))


async def get_mongo() -> Any:
    global _mongo_client, _mongo_failed, _mongo_last_error
    if not MONGODB_URI or _mongo_failed:
        return None
    if _mongo_client is not None:
        return _mongo_client[MONGODB_DB]
    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        _mongo_client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=1500)
        await _mongo_client.admin.command("ping")
        logger.info("mongodb_connected")
        _mongo_last_error = None
        return _mongo_client[MONGODB_DB]
    except Exception as exc:  # pragma: no cover - depends on external service
        _mongo_failed = True
        _mongo_last_error = str(exc).replace(MONGODB_URI, "<MONGODB_URI>")[:1200]
        logger.warning("mongodb_unavailable_using_local_fallback: %s", exc)
        return None


async def reset_mongo_connection() -> Any:
    global _mongo_client, _mongo_failed, _mongo_last_error
    if _mongo_client is not None:
        _mongo_client.close()
    _mongo_client = None
    _mongo_failed = False
    _mongo_last_error = None
    return await get_mongo()


async def require_mongo() -> Any:
    db = await get_mongo()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MongoDB Atlas is not configured. Set MONGODB_URI in DigitalOcean App Platform.",
        )
    return db


async def save_session(record: dict[str, Any]) -> bool:
    ensure_dirs()
    await write_json(session_path(record["session_id"]), record)
    await asyncio.to_thread(append_jsonl, RECEIVED_PATH, record)
    db = await get_mongo()
    if db is None:
        return False
    await db.sessions.update_one(
        {"session_id": record["session_id"]},
        {"$set": prepare_session_document(record)},
        upsert=True,
    )
    return True


async def save_event(session_id: str, event: dict[str, Any]) -> None:
    envelope = {"session_id": session_id, "received_at": utc_now(), **event}
    await asyncio.to_thread(append_jsonl, EVENTS_PATH, envelope)
    await asyncio.to_thread(append_jsonl, stream_events_path(session_id), envelope)
    db = await get_mongo()
    if db is not None:
        await db.events.insert_one(envelope)


async def read_streamed_events(session_id: str) -> list[dict[str, Any]]:
    path = stream_events_path(session_id)
    if not path.exists():
        return []
    text = await asyncio.to_thread(path.read_text, "utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


async def save_report(report: dict[str, Any]) -> bool:
    await write_json(report_path(report["session_id"]), report)
    db = await get_mongo()
    saved_to_mongo = False
    if db is not None:
        verdict_doc = prepare_verdict_document(report)
        await db.verdicts.update_one({"session_id": report["session_id"]}, {"$set": verdict_doc}, upsert=True)
        await db.reports.update_one({"session_id": report["session_id"]}, {"$set": verdict_doc}, upsert=True)
        await db.sessions.update_one(
            {"session_id": report["session_id"]},
            {
                "$set": {
                    "features": report.get("features", {}),
                    "latest_verdict": report.get("verdict"),
                    "latest_score": report.get("overall_score", report.get("score")),
                    "latest_confidence": report.get("confidence"),
                }
            },
            upsert=False,
        )
        saved_to_mongo = True
    if REPORT_WEBHOOK_URL:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(REPORT_WEBHOOK_URL, json=report)
    return saved_to_mongo


async def save_disagreement_seed(session_id: str, seed: dict[str, Any]) -> bool:
    created_at = utc_now()
    record = prepare_disagreement_document(session_id, seed, created_at)
    await write_json(disagreement_path(session_id), record)
    db = await get_mongo()
    if db is None:
        return True
    await db.disagreements.update_one({"session_id": session_id}, {"$set": record}, upsert=True)
    return True


async def generate_report_for_session(session_id: str) -> dict[str, Any]:
    session = await read_json(session_path(session_id))
    report = extract_verdict(session, await read_streamed_events(session_id))
    await attach_external_evaluation(session, report)
    report["saved_to_mongodb"] = await save_report(report)
    return report


async def attach_external_evaluation(session: dict[str, Any], report: dict[str, Any]) -> None:
    agreement = await evaluate_with_external_llms(session, report)
    verdict_document = build_verdict_document(report["session_id"], agreement)
    agreement_data = model_dump(agreement)
    verdict_document_data = model_dump(verdict_document)
    report["llm_evaluation"] = agreement_data
    report["verdict_document"] = verdict_document_data
    report["verdict"] = agreement.final_verdict
    report["score"] = agreement.final_score
    report["overall_score"] = agreement.final_score
    report["confidence"] = agreement.confidence
    report["flagged_claims"] = agreement.flagged_claims
    report["behavioral_notes"] = agreement.behavioral_notes
    report["evidence"]["llm_mode"] = agreement.mode
    report["evidence"]["llm_confidence"] = agreement.confidence
    report["evidence"]["llm_agreement"] = agreement.agreement
    report["evidence"]["llm_score_delta"] = agreement.score_delta
    report["evidence"]["llm_claim_disagreement"] = agreement.claim_disagreement
    seed = verdict_document.self_improvement_seed
    if seed:
        saved_seed = await save_disagreement_seed(report["session_id"], seed)
        report["self_improvement_seed_saved"] = saved_seed
    else:
        report["self_improvement_seed_saved"] = False


def extract_verdict(session: dict[str, Any], streamed_events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    transcript = session.get("transcript") or []
    events = [model_dump(event) if isinstance(event, BaseModel) else event for event in session.get("events") or []]
    if streamed_events:
        events.extend(streamed_events)
    drawings = session.get("drawings") or []
    metadata = session.get("metadata") or {}
    feature_bundle = extract_feature_bundle(session, streamed_events)
    feature_data = model_dump(feature_bundle)
    student_turns = [turn for turn in transcript if str(turn.get("speaker", "")).lower() == "student"]
    text = " ".join(str(turn.get("text", "")) for turn in transcript).lower()
    misconception_hits = sum(
        phrase in text
        for phrase in (
            "maybe",
            "not sure",
            "stuck",
            "mix up",
            "wrong",
            "confused",
            "i think",
        )
    )
    revision_hits = sum(phrase in text for phrase in ("revise", "wait", "repair", "stronger explanation"))
    board_events = sum(1 for event in events if event.get("type") in {"add", "click", "pause", "rewatch"})
    base = 0.45
    base += min(0.2, len(student_turns) * 0.025)
    base += min(0.15, len(drawings) * 0.08)
    base += min(0.1, board_events * 0.01)
    base += min(0.1, revision_hits * 0.05)
    base += min(0.08, feature_bundle.metadata["verifiable_claim_count"] * 0.01)
    base += min(0.05, feature_bundle.behavior.drawing_score * 0.05)
    base -= min(0.2, misconception_hits * 0.04)
    score = round(max(0.0, min(1.0, base)), 3)
    if score >= 0.72:
        verdict = "on_track"
    elif score >= 0.5:
        verdict = "needs_targeted_support"
    else:
        verdict = "needs_review"
    return {
        "session_id": session["session_id"],
        "status": "complete",
        "verdict": verdict,
        "score": score,
        "generated_at": utc_now(),
        "summary": (
            f"{metadata.get('student_kind', 'student')} session on {metadata.get('subject', 'unknown subject')} "
            f"with {len(transcript)} transcript turns, {len(events)} events, and {len(drawings)} drawing artifact(s)."
        ),
        "evidence": {
            "transcript_turns": len(transcript),
            "student_turns": len(student_turns),
            "event_count": len(events),
            "drawing_count": len(drawings),
            "misconception_hits": misconception_hits,
            "revision_hits": revision_hits,
            "duration_ms": metadata.get("duration_ms"),
            "student_id": metadata.get("student_id"),
            "human_student": metadata.get("human_student", False),
            "verifiable_claim_count": feature_bundle.metadata["verifiable_claim_count"],
            "rewatch_rate": feature_bundle.behavior.rewatch_rate,
            "hesitation_ms": feature_bundle.behavior.hesitation_ms,
            "drawing_score": feature_bundle.behavior.drawing_score,
        },
        "features": feature_data,
        "saved_to_mongodb": False,
    }


async def process_session(session_id: str) -> None:
    try:
        await generate_report_for_session(session_id)
        logger.info("session_processed", extra={"session_id": session_id})
    except Exception:
        logger.exception("session_processing_failed", extra={"session_id": session_id})


def assert_agent_session(payload: FerbAISessionOutput) -> None:
    if payload.metadata.human_student or payload.metadata.student_kind != "agent_student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This ingestor accepts agent-student FerbAI sessions only.",
        )


def generation_running() -> bool:
    return _generation_task is not None and not _generation_task.done()


def generation_status() -> dict[str, Any]:
    return {
        "running": generation_running(),
        "started_at": _generation_started_at,
        "generated_count": _generation_count,
        "last_session_id": _generation_last_session_id,
        "last_report_url": (
            f"/sessions/{_generation_last_session_id}/report" if _generation_last_session_id else None
        ),
        "last_verdict": _generation_last_report.get("verdict") if _generation_last_report else None,
        "last_score": _generation_last_report.get("score") if _generation_last_report else None,
        "interval_seconds": _generation_interval_seconds,
        "stop_reason": _generation_stop_reason,
    }


def human_bool(value: Any) -> str:
    return "yes" if value else "no"


def format_generation_status_text(status_data: dict[str, Any]) -> str:
    lines = [
        "FerbAI Live Generation",
        "======================",
        f"Running: {human_bool(status_data.get('running'))}",
        f"Generated sessions: {status_data.get('generated_count', 0)}",
        f"Interval: {status_data.get('interval_seconds')} seconds",
        f"Stop reason: {status_data.get('stop_reason')}",
        f"Started at: {status_data.get('started_at') or 'not started'}",
        f"Latest session: {status_data.get('last_session_id') or 'none yet'}",
        f"Latest verdict: {status_data.get('last_verdict') or 'none yet'}",
        f"Latest score: {status_data.get('last_score') if status_data.get('last_score') is not None else 'none yet'}",
    ]
    if status_data.get("last_report_url"):
        lines.append(f"Latest report: {status_data['last_report_url']}")
        lines.append(f"Readable report: {status_data['last_report_url']}.txt")
    return "\n".join(lines) + "\n"


def format_report_text(report: dict[str, Any]) -> str:
    evidence = report.get("evidence", {})
    doc = report.get("verdict_document") or {}
    behavior = report.get("behavioral_notes") or doc.get("behavioral_notes") or {}
    flagged_claims = report.get("flagged_claims") or doc.get("flagged_claims") or []
    lines = [
        "FerbAI Verdict Report",
        "====================",
        f"Session: {report.get('session_id')}",
        f"Verdict: {report.get('verdict')}",
        f"Overall score: {report.get('overall_score', report.get('score'))}",
        f"Confidence: {report.get('confidence', doc.get('confidence', 'unknown'))}",
        f"Generated at: {report.get('generated_at')}",
        "",
        "Evaluator Agreement",
        "-------------------",
        f"Mode: {evidence.get('llm_mode', 'unknown')}",
        f"Agreement: {evidence.get('llm_agreement')}",
        f"Score delta: {evidence.get('llm_score_delta')}",
        f"Claim disagreement: {evidence.get('llm_claim_disagreement')}",
        f"Self-improvement seed saved: {human_bool(report.get('self_improvement_seed_saved'))}",
        "",
        "Behavioral Signals",
        "------------------",
        f"Rewatch: {behavior.get('rewatch', {})}",
        f"Hesitation: {behavior.get('hesitation', {})}",
        f"Drawing: {behavior.get('drawing', {})}",
        "",
        "Flagged Claims",
        "--------------",
    ]
    if flagged_claims:
        for index, item in enumerate(flagged_claims, start=1):
            providers = ", ".join(item.get("providers") or [])
            suffix = f" [{providers}]" if providers else ""
            lines.append(f"{index}. {item.get('claim')} - {item.get('reason')}{suffix}")
    else:
        lines.append("No flagged claims.")
    lines.extend(
        [
            "",
            "Student Source",
            "--------------",
            f"Student ID: {evidence.get('student_id')}",
            f"Human student: {human_bool(evidence.get('human_student'))}",
            f"Transcript turns: {evidence.get('transcript_turns')}",
            f"Events: {evidence.get('event_count')}",
            f"Drawings: {evidence.get('drawing_count')}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_dashboard_html(status_data: dict[str, Any], latest_report: dict[str, Any] | None = None) -> str:
    latest_url = status_data.get("last_report_url")
    latest_id = status_data.get("last_session_id")
    report = latest_report or {}
    doc = report.get("verdict_document") or {}
    llm_evaluation = report.get("llm_evaluation") or {}
    behavior = report.get("behavioral_notes") or doc.get("behavioral_notes") or {}
    flagged_claims = report.get("flagged_claims") or doc.get("flagged_claims") or []
    report_links = ""
    if latest_url:
        report_links = (
            f'<a href="{escape(latest_url)}">JSON report</a>'
            f'<a href="{escape(latest_url)}.txt">Readable report</a>'
        )
    claims_html = "<li>No flagged claims yet.</li>"
    if flagged_claims:
        claims_html = "".join(
            "<li><strong>{claim}</strong><span>{reason}</span></li>".format(
                claim=escape(str(item.get("claim", ""))),
                reason=escape(str(item.get("reason", ""))),
            )
            for item in flagged_claims
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="10">
  <title>FerbAI Live Dashboard</title>
  <style>
    :root {{ color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f6f7f9; color: #1b1d22; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 32px 20px 48px; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 24px; }}
    h1 {{ margin: 0; font-size: 30px; line-height: 1.1; }}
    h2 {{ margin: 0 0 12px; font-size: 17px; }}
    p {{ margin: 6px 0; color: #4b5563; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .wide {{ grid-column: span 2; }}
    section, .metric {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 16px; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 4px; }}
    .pill {{ display: inline-flex; align-items: center; padding: 4px 9px; border-radius: 999px; background: #e8f2ef; color: #116149; font-size: 13px; font-weight: 700; }}
    .pill.off {{ background: #f3e8e8; color: #8a2727; }}
    .actions {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    form {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
    input {{ width: 88px; padding: 8px; border: 1px solid #cfd6df; border-radius: 6px; }}
    button, a {{ display: inline-flex; align-items: center; min-height: 36px; padding: 0 12px; border-radius: 6px; border: 1px solid #b9c3cf; background: #fff; color: #16324f; text-decoration: none; font-weight: 700; }}
    button.primary {{ background: #0d6b63; border-color: #0d6b63; color: #fff; }}
    ul {{ padding-left: 18px; }}
    li span {{ display: block; color: #5b6472; }}
    code {{ background: #eef1f5; padding: 2px 5px; border-radius: 4px; }}
    @media (prefers-color-scheme: dark) {{
      body {{ background: #101215; color: #f3f4f6; }}
      p, li span {{ color: #a8b0bc; }}
      section, .metric, button, a {{ background: #181b20; border-color: #2c333d; color: #d9eefb; }}
      input, code {{ background: #111419; border-color: #2c333d; color: #f3f4f6; }}
    }}
    @media (max-width: 780px) {{ .grid {{ grid-template-columns: 1fr; }} .wide {{ grid-column: span 1; }} header {{ display: block; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>FerbAI Live Dashboard</h1>
        <p>Human-readable view of the DigitalOcean app, synthetic student swarm, and dual-evaluator verdict engine. Auto-refreshes every 10 seconds.</p>
      </div>
      <span class="pill {' ' if status_data.get('running') else 'off'}">{'RUNNING' if status_data.get('running') else 'STOPPED'}</span>
    </header>
    <div class="grid">
      <div class="metric"><span>Generated sessions</span><strong>{escape(str(status_data.get('generated_count', 0)))}</strong></div>
      <div class="metric"><span>Latest verdict</span><strong>{escape(str(status_data.get('last_verdict') or 'none yet'))}</strong></div>
      <div class="metric"><span>Latest score</span><strong>{escape(str(status_data.get('last_score') if status_data.get('last_score') is not None else 'none yet'))}</strong></div>
      <section class="wide">
        <h2>Controls</h2>
        <form method="post" action="/generation/start?interval_seconds=120&limit=0&stream_events=2">
          <button class="primary" type="submit">Start continuous generation</button>
        </form>
        <form method="post" action="/generation/start?interval_seconds=5&limit=3&stream_events=2" style="margin-top:8px">
          <button type="submit">Run 3-session smoke test</button>
        </form>
        <form method="post" action="/generation/stop" style="margin-top:8px">
          <button type="submit">Stop generation</button>
        </form>
      </section>
      <section>
        <h2>Links</h2>
        <div class="actions">
          <a href="/generation/status.txt">Readable status</a>
          <a href="/generation/status">JSON status</a>
          <a href="/demo.txt">Run readable demo</a>
          <a href="/evaluators/status">Evaluator status</a>
          {report_links}
        </div>
      </section>
      <section class="wide">
        <h2>Latest Session</h2>
        <p><strong>ID:</strong> <code>{escape(str(latest_id or 'none yet'))}</code></p>
        <p><strong>Started:</strong> {escape(str(status_data.get('started_at') or 'not started'))}</p>
        <p><strong>Stop reason:</strong> {escape(str(status_data.get('stop_reason')))}</p>
        <p><strong>Confidence:</strong> {escape(str(report.get('confidence') or doc.get('confidence') or 'none yet'))}</p>
        <p><strong>Agreement:</strong> {escape(str(llm_evaluation.get('agreement', 'none yet')))}</p>
        <p><strong>Self-improvement seed saved:</strong> {escape(human_bool(report.get('self_improvement_seed_saved')) if report else 'none yet')}</p>
      </section>
      <section>
        <h2>Behavior</h2>
        <p><strong>Rewatch:</strong> {escape(str(behavior.get('rewatch', 'none yet')))}</p>
        <p><strong>Hesitation:</strong> {escape(str(behavior.get('hesitation', 'none yet')))}</p>
        <p><strong>Drawing:</strong> {escape(str(behavior.get('drawing', 'none yet')))}</p>
      </section>
      <section class="wide">
        <h2>Flagged Claims</h2>
        <ul>{claims_html}</ul>
      </section>
    </div>
  </main>
</body>
</html>"""


async def ingest_generated_session(payload: FerbAISessionOutput, stream_events: int) -> dict[str, Any]:
    assert_agent_session(payload)
    record = model_dump(payload)
    await save_session(record)
    for event in record.get("events", [])[:stream_events]:
        streamed_event = {
            **event,
            "payload": {
                **event.get("payload", {}),
                "generated_stream": True,
                "source": "continuous_ferbai_generator",
            },
        }
        await save_event(payload.session_id, streamed_event)
    return await generate_report_for_session(payload.session_id)


async def continuous_generation_loop(interval_seconds: float, limit: int, stream_events: int) -> None:
    global _generation_count, _generation_interval_seconds, _generation_last_report
    global _generation_last_session_id, _generation_stop_reason, _generation_task

    run_count = 0
    _generation_interval_seconds = interval_seconds
    _generation_stop_reason = "running"
    try:
        while limit == 0 or run_count < limit:
            payload = load_demo_payload()
            next_count = _generation_count + 1
            payload.session_id = f"continuous_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{next_count}"
            payload.timestamp = utc_now()
            report = await ingest_generated_session(payload, stream_events)
            _generation_count = next_count
            _generation_last_session_id = payload.session_id
            _generation_last_report = report
            run_count += 1
            logger.info("continuous_session_generated", extra={"session_id": payload.session_id})
            if limit == 0 or run_count < limit:
                await asyncio.sleep(interval_seconds)
        _generation_stop_reason = "limit_reached"
    except asyncio.CancelledError:
        _generation_stop_reason = "stopped"
        raise
    except Exception:
        _generation_stop_reason = "error"
        logger.exception("continuous_generation_failed")
    finally:
        _generation_task = None


@app.on_event("startup")
async def startup() -> None:
    ensure_dirs()
    db = await get_mongo()
    if db is not None and ATLAS_BOOTSTRAP_ON_STARTUP:
        try:
            await ensure_atlas_indexes(db)
            persona_count = await seed_personas(db, load_personas())
            logger.info("atlas_bootstrapped", extra={"duration_ms": persona_count})
        except Exception:
            logger.exception("atlas_bootstrap_failed")
    logger.info("ingestor_started")


@app.on_event("shutdown")
async def shutdown() -> None:
    if generation_running():
        _generation_task.cancel()
        with suppress(asyncio.CancelledError):
            await _generation_task
    if _mongo_client is not None:
        _mongo_client.close()
    logger.info("ingestor_stopped")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "ferbai-fastapi-ingestor",
        "mongodb_configured": bool(MONGODB_URI),
        "storage": str(DATA_DIR),
        "generation_running": generation_running(),
    }


async def latest_report_for_dashboard() -> dict[str, Any] | None:
    status_data = generation_status()
    session_id = status_data.get("last_session_id")
    if not session_id:
        return None
    path = report_path(str(session_id))
    if not path.exists():
        return None
    return await read_json(path)


async def mongo_status_payload() -> dict[str, Any]:
    configured = bool(MONGODB_URI)
    db = await get_mongo()
    if configured and db is None and _mongo_failed:
        db = await reset_mongo_connection()
    if db is None:
        return {
            "configured": configured,
            "connected": False,
            "database": MONGODB_DB,
            "provider": "MongoDB Atlas on GCP",
            "last_error": _mongo_last_error,
            "collections": ["sessions", "events", "verdicts", "disagreements", "personas"],
        }
    names = await db.list_collection_names()
    return {
        "configured": configured,
        "connected": True,
        "database": MONGODB_DB,
        "provider": "MongoDB Atlas on GCP",
        "last_error": None,
        "collections": sorted(name for name in names if not name.startswith("system.")),
    }


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    status_data = generation_status()
    latest_report = await latest_report_for_dashboard()
    return HTMLResponse(render_dashboard_html(status_data, latest_report))


@app.get("/live", response_class=HTMLResponse)
async def live_dashboard() -> HTMLResponse:
    status_data = generation_status()
    latest_report = await latest_report_for_dashboard()
    return HTMLResponse(render_dashboard_html(status_data, latest_report))


@app.get("/api")
async def api_root() -> dict[str, Any]:
    return {
        "service": "ferbai-fastapi-ingestor",
        "ok": True,
        "routes": {
            "health": "/health",
            "demo": "/demo",
            "ingest_session": "POST /sessions",
            "stream_event": "POST /sessions/{session_id}/event",
            "report": "GET /sessions/{session_id}/report",
            "generation_status": "/generation/status",
            "generation_start": "POST /generation/start?interval_seconds=15&limit=0",
            "generation_stop": "POST /generation/stop",
            "evaluator_status": "/evaluators/status",
            "disagreement_seed": "GET /sessions/{session_id}/disagreement-seed",
            "live_dashboard": "/live",
            "readable_generation_status": "/generation/status.txt",
            "readable_demo": "/demo.txt",
            "atlas_status": "/atlas/status",
            "atlas_bootstrap": "POST /atlas/bootstrap",
            "atlas_analytics": "/atlas/analytics",
            "atlas_drift": "/atlas/verdict-drift",
        },
        "generation": generation_status(),
    }


@app.get("/atlas/status")
async def atlas_status() -> dict[str, Any]:
    payload = await mongo_status_payload()
    payload["search_indexes"] = atlas_search_index_definitions()
    payload["connection_notes"] = {
        "driver": "motor async PyMongo",
        "pooling": "global AsyncIOMotorClient reused across requests",
        "write_order": "sessions first, verdicts after consensus, disagreements only when seeds exist",
    }
    return payload


@app.post("/atlas/bootstrap")
async def atlas_bootstrap() -> dict[str, Any]:
    db = await require_mongo()
    indexes = await ensure_atlas_indexes(db)
    personas = load_personas()
    persona_count = await seed_personas(db, personas)
    return {
        "ok": True,
        "database": MONGODB_DB,
        "provider": "MongoDB Atlas on GCP",
        "indexes": indexes,
        "personas_seeded": persona_count,
        "search_index_definitions": atlas_search_index_definitions(),
        "note": "Create the Atlas Search and Vector Search definitions in Atlas if your cluster does not allow driver-managed search index creation.",
    }


@app.post("/atlas/reconnect")
async def atlas_reconnect() -> dict[str, Any]:
    db = await reset_mongo_connection()
    if db is None:
        payload = await mongo_status_payload()
        payload["ok"] = False
        return payload
    if ATLAS_BOOTSTRAP_ON_STARTUP:
        await ensure_atlas_indexes(db)
        await seed_personas(db, load_personas())
    payload = await mongo_status_payload()
    payload["ok"] = True
    return payload


@app.get("/atlas/analytics")
async def atlas_analytics() -> dict[str, Any]:
    db = await require_mongo()
    return await swarm_analytics(db)


@app.get("/atlas/verdict-drift")
async def atlas_verdict_drift(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    db = await require_mongo()
    return {"items": await verdict_drift(db, limit)}


@app.get("/personas")
async def list_personas(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    db = await get_mongo()
    if db is not None:
        cursor = db.personas.find({}, {"_id": False}).sort("persona_id", 1).limit(limit)
        return {"source": "mongodb_atlas", "items": [doc async for doc in cursor]}
    return {"source": "local_file", "items": load_personas()[:limit]}


@app.get("/disagreements")
async def list_disagreements(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    db = await get_mongo()
    if db is not None:
        cursor = db.disagreements.find({}, {"_id": False}).sort("created_at", -1).limit(limit)
        return {"source": "mongodb_atlas", "items": [doc async for doc in cursor]}
    items = []
    for path in sorted(DISAGREEMENTS_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        items.append(await read_json(path))
    return {"source": "local_json", "items": items}


@app.get("/evaluators/status")
async def evaluator_status() -> dict[str, Any]:
    return {
        "gemini": {
            "configured": bool(os.getenv("GEMINI_API_KEY", "")),
            "model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
            "role": "primary evaluator via external Google API call from DigitalOcean",
        },
        "minimax": {
            "configured": bool(os.getenv("MINIMAX_API_KEY") or os.getenv("MINIMAX_ACCESS_TOKEN", "")),
            "model": os.getenv("MINIMAX_MODEL", "MiniMax-Text-01"),
            "role": "parallel MoE cross-check evaluator",
            "timeout_seconds": float(os.getenv("EVALUATOR_TIMEOUT_SECONDS", "8")),
        },
        "agreement_logic": {
            "parallel": True,
            "score_delta_uncertain_threshold": 0.3,
            "claim_disagreement": "flagged_for_review and saved as a self-improvement seed",
            "behavioral_confidence_adjustment": "rewatch and hesitation dampen confidence; strong drawing can amplify it",
            "fallback": "local feature verdict when external evaluators are unavailable",
        },
    }


@app.get("/generation/status")
async def get_generation_status() -> dict[str, Any]:
    return generation_status()


@app.get("/generation/status.txt", response_class=PlainTextResponse)
async def get_generation_status_text() -> PlainTextResponse:
    return PlainTextResponse(format_generation_status_text(generation_status()))


@app.post("/generation/start", status_code=status.HTTP_202_ACCEPTED)
async def start_generation(
    interval_seconds: float = Query(
        default=15,
        ge=0.2,
        le=3600,
        description="Seconds between generated FerbAI agent-student sessions.",
    ),
    limit: int = Query(
        default=0,
        ge=0,
        le=100000,
        description="Number of sessions to generate. Use 0 for continuous generation.",
    ),
    stream_events: int = Query(
        default=2,
        ge=0,
        le=10,
        description="How many interaction events to replay through POST /sessions/{id}/event storage.",
    ),
) -> dict[str, Any]:
    global _generation_interval_seconds, _generation_started_at, _generation_stop_reason, _generation_task

    if generation_running():
        return {"ok": True, "already_running": True, "generation": generation_status()}
    _generation_interval_seconds = interval_seconds
    _generation_started_at = utc_now()
    _generation_stop_reason = "starting"
    _generation_task = asyncio.create_task(continuous_generation_loop(interval_seconds, limit, stream_events))
    logger.info("continuous_generation_started")
    return {
        "ok": True,
        "started": True,
        "continuous": limit == 0,
        "generation": generation_status(),
    }


@app.post("/generation/stop")
async def stop_generation() -> dict[str, Any]:
    global _generation_stop_reason

    task = _generation_task
    if task is None or task.done():
        _generation_stop_reason = "not_running"
        return {"ok": True, "stopped": False, "generation": generation_status()}
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    return {"ok": True, "stopped": True, "generation": generation_status()}


@app.post("/sessions", status_code=status.HTTP_202_ACCEPTED)
async def ingest_session(payload: FerbAISessionOutput, background_tasks: BackgroundTasks) -> dict[str, Any]:
    assert_agent_session(payload)
    record = model_dump(payload)
    saved_to_mongo = await save_session(record)
    background_tasks.add_task(process_session, payload.session_id)
    logger.info("session_ingested", extra={"session_id": payload.session_id})
    return {
        "ok": True,
        "accepted": payload.session_id,
        "report_url": f"/sessions/{payload.session_id}/report",
        "saved_to_mongodb": saved_to_mongo,
        "counts": {
            "transcript": len(payload.transcript),
            "events": len(payload.events),
            "drawings": len(payload.drawings),
        },
    }


@app.post("/sessions/{session_id}/event", status_code=status.HTTP_202_ACCEPTED)
async def stream_event(session_id: str, event: StreamEvent, background_tasks: BackgroundTasks) -> dict[str, Any]:
    if not session_path(session_id).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown session_id")
    event_record = model_dump(event)
    await save_event(session_id, event_record)
    background_tasks.add_task(process_session, session_id)
    logger.info("event_ingested", extra={"session_id": session_id, "event_type": event.type})
    return {"ok": True, "accepted": session_id, "event_type": event.type}


@app.get("/sessions/{session_id}/report")
async def get_report(session_id: str, response: Response) -> dict[str, Any]:
    path = report_path(session_id)
    if not path.exists():
        if not session_path(session_id).exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown session_id")
        response.status_code = status.HTTP_202_ACCEPTED
        return {"session_id": session_id, "status": "processing"}
    return await read_json(path)


@app.get("/sessions/{session_id}/report.txt", response_class=PlainTextResponse)
async def get_report_text(session_id: str) -> PlainTextResponse:
    path = report_path(session_id)
    if not path.exists():
        if not session_path(session_id).exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown session_id")
        return PlainTextResponse(f"Session {session_id} is still processing.\n", status_code=status.HTTP_202_ACCEPTED)
    return PlainTextResponse(format_report_text(await read_json(path)))


@app.get("/sessions/{session_id}/disagreement-seed")
async def get_disagreement_seed(session_id: str) -> dict[str, Any]:
    path = disagreement_path(session_id)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No disagreement seed for session_id")
    return await read_json(path)


def load_demo_payload() -> FerbAISessionOutput:
    payload_path = Path("ferbai_session_outputs.json")
    if not payload_path.exists():
        raise HTTPException(status_code=500, detail="Demo payload file missing. Run ferbai_session_outputs.py first.")
    payloads = json.loads(payload_path.read_text(encoding="utf-8"))
    if not payloads:
        raise HTTPException(status_code=500, detail="Demo payload file is empty.")
    payload = payloads[0]
    payload["session_id"] = f"demo_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    payload["timestamp"] = utc_now()
    return FerbAISessionOutput(**payload)


@app.get("/demo")
async def demo() -> dict[str, Any]:
    payload = load_demo_payload()
    assert_agent_session(payload)
    record = model_dump(payload)
    saved_session_to_mongo = await save_session(record)
    report = extract_verdict(record)
    await attach_external_evaluation(record, report)
    report["saved_to_mongodb"] = await save_report(report)
    logger.info("demo_completed", extra={"session_id": payload.session_id})
    return {
        "ok": True,
        "session_id": payload.session_id,
        "saved_to_mongodb": saved_session_to_mongo or report["saved_to_mongodb"],
        "verdict": report,
    }


@app.post("/demo")
async def demo_post() -> dict[str, Any]:
    return await demo()


@app.get("/demo.txt", response_class=PlainTextResponse)
async def demo_text() -> PlainTextResponse:
    result = await demo()
    return PlainTextResponse(format_report_text(result["verdict"]))
