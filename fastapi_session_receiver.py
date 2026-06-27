from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

from feature_extraction import dump_model, extract_feature_bundle


DATA_DIR = Path(os.getenv("DATA_DIR", "ingestor_data"))
SESSIONS_DIR = DATA_DIR / "sessions"
REPORTS_DIR = DATA_DIR / "reports"
STREAM_EVENTS_DIR = DATA_DIR / "stream_events"
EVENTS_PATH = DATA_DIR / "events.jsonl"
RECEIVED_PATH = Path("received_ferbai_sessions.jsonl")
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "ferbai")
REPORT_WEBHOOK_URL = os.getenv("REPORT_WEBHOOK_URL", "")


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
    version="0.2.0",
)

_mongo_client: Any = None
_mongo_failed = False


def model_dump(model: BaseModel) -> dict[str, Any]:
    return dump_model(model)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_dirs() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    STREAM_EVENTS_DIR.mkdir(parents=True, exist_ok=True)


def session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def report_path(session_id: str) -> Path:
    return REPORTS_DIR / f"{session_id}.json"


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
    global _mongo_client, _mongo_failed
    if not MONGODB_URI or _mongo_failed:
        return None
    if _mongo_client is not None:
        return _mongo_client[MONGODB_DB]
    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        _mongo_client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=1500)
        await _mongo_client.admin.command("ping")
        logger.info("mongodb_connected")
        return _mongo_client[MONGODB_DB]
    except Exception as exc:  # pragma: no cover - depends on external service
        _mongo_failed = True
        logger.warning("mongodb_unavailable_using_local_fallback: %s", exc)
        return None


async def save_session(record: dict[str, Any]) -> bool:
    ensure_dirs()
    await write_json(session_path(record["session_id"]), record)
    await asyncio.to_thread(append_jsonl, RECEIVED_PATH, record)
    db = await get_mongo()
    if db is None:
        return False
    await db.sessions.update_one({"session_id": record["session_id"]}, {"$set": record}, upsert=True)
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
        await db.reports.update_one({"session_id": report["session_id"]}, {"$set": report}, upsert=True)
        saved_to_mongo = True
    if REPORT_WEBHOOK_URL:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(REPORT_WEBHOOK_URL, json=report)
    return saved_to_mongo


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
        session = await read_json(session_path(session_id))
        report = extract_verdict(session, await read_streamed_events(session_id))
        report["saved_to_mongodb"] = await save_report(report)
        logger.info("session_processed", extra={"session_id": session_id})
    except Exception:
        logger.exception("session_processing_failed", extra={"session_id": session_id})


def assert_agent_session(payload: FerbAISessionOutput) -> None:
    if payload.metadata.human_student or payload.metadata.student_kind != "agent_student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This ingestor accepts agent-student FerbAI sessions only.",
        )


@app.on_event("startup")
async def startup() -> None:
    ensure_dirs()
    await get_mongo()
    logger.info("ingestor_started")


@app.on_event("shutdown")
async def shutdown() -> None:
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
    }


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
