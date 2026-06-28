from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


SWARM_PATH = Path("synthetic_student_swarm.json")


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(text)


def _elapsed_ms(start: datetime, value: Any) -> int:
    return max(0, int((_parse_time(value) - start).total_seconds() * 1000))


def _xml_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _drawing_svg(drawing_sequence: list[dict[str, Any]]) -> str:
    parts = []
    for index, item in enumerate(drawing_sequence):
        path = item.get("svg_path")
        if not path:
            continue
        color = "#1f7a5b" if item.get("type") == "node" else "#334155"
        parts.append(
            f'<path d="{_xml_escape(path)}" fill="none" stroke="{color}" '
            f'stroke-width="3" stroke-linecap="round"/>'
        )
        if item.get("label") and item.get("type") == "node":
            x = 36 + (index * 82) % 720
            y = 124 + ((index // 5) * 72)
            parts.append(
                f'<text x="{x}" y="{y}" font-size="15" fill="#111827">'
                f'{_xml_escape(item["label"])}</text>'
            )
    body = "\n  ".join(parts)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640" '
        'viewBox="0 0 960 640">\n  '
        f"{body}\n</svg>"
    )


def _events(session: dict[str, Any], start: datetime) -> list[dict[str, Any]]:
    persona = session.get("persona", {})
    behavior = session.get("behavioral_signals", {})
    drawing_sequence = session.get("drawing_sequence") or []
    ability = float(persona.get("ability") or 0.5)
    hesitation = float(behavior.get("hesitation_score") or 0.0)
    base_pause = int(900 + hesitation * 9000 + (1.0 - ability) * 1600)
    events: list[dict[str, Any]] = [
        {
            "type": "pause",
            "t": 3200,
            "payload": {"duration_ms": base_pause, "reason": "agent_student_thinking"},
        },
        {
            "type": "click",
            "t": 6200,
            "payload": {"target": "whiteboard", "tool": "concept_map", "agent_generated": True},
        },
    ]
    for index, item in enumerate(drawing_sequence):
        events.append(
            {
                "type": "add",
                "t": 7200 + index * 850,
                "payload": {
                    "element_type": item.get("type", "path"),
                    "label": item.get("label"),
                    "stroke_index": index,
                },
            }
        )
    if behavior.get("self_revision"):
        events.append(
            {
                "type": "revision",
                "t": 21000,
                "payload": {"reason": "self_corrected_misconception", "agent_generated": True},
            }
        )
    rewatch_count = 2 if ability < 0.42 else 1 if ability < 0.68 or hesitation > 0.45 else 0
    for index in range(rewatch_count):
        events.append(
            {
                "type": "rewatch",
                "t": 24000 + index * 3600,
                "payload": {
                    "from_ms": 4000,
                    "to_ms": 16000,
                    "reason": "agent_reviewed_gap",
                    "segment": index + 1,
                },
            }
        )
    last_turn_time = max((_elapsed_ms(start, turn["t"]) for turn in session.get("transcript", [])), default=0)
    return sorted((event for event in events if event["t"] <= last_turn_time + 10000), key=lambda event: event["t"])


def convert_swarm_session(session: dict[str, Any]) -> dict[str, Any]:
    persona = session.get("persona", {})
    behavior = session.get("behavioral_signals", {})
    ground_truth = session.get("ground_truth", {})
    start = _parse_time(session.get("created_at") or datetime.now(timezone.utc))
    transcript = [
        {
            "t": _elapsed_ms(start, turn.get("t")),
            "speaker": str(turn.get("speaker", "student")).title(),
            "text": str(turn.get("text", "")),
        }
        for turn in session.get("transcript", [])
    ]
    duration_ms = max((int(turn["t"]) for turn in transcript), default=0) + 6000
    misconceptions = persona.get("misconceptions") or []
    return {
        "session_id": session.get("session_id"),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "transcript": transcript,
        "events": _events(session, start),
        "drawings": [_drawing_svg(session.get("drawing_sequence") or [])],
        "metadata": {
            "duration_ms": duration_ms,
            "subject": session.get("subject") or persona.get("subject") or "unknown subject",
            "student_id": session.get("student_id") or persona.get("persona_id"),
            "student_kind": "agent_student",
            "agent_generated": True,
            "human_student": False,
            "recording_id": session.get("recording_id"),
            "understanding_band": persona.get("understanding_band"),
            "confidence": behavior.get("confidence") or persona.get("confidence"),
            "ability": persona.get("ability"),
            "support_need": persona.get("support_need"),
            "ground_truth_verdict": ground_truth.get("verdict"),
            "misconception_count": len(misconceptions),
            "source_swarm_id": session.get("session_id"),
        },
    }


@lru_cache(maxsize=1)
def load_swarm_payloads(path: str = str(SWARM_PATH)) -> list[dict[str, Any]]:
    payload_path = Path(path)
    if not payload_path.exists():
        return []
    source = json.loads(payload_path.read_text(encoding="utf-8"))
    return [convert_swarm_session(session) for session in source.get("sessions", [])]


def clone_swarm_payload(index: int) -> dict[str, Any] | None:
    payloads = load_swarm_payloads()
    if not payloads:
        return None
    return copy.deepcopy(payloads[index % len(payloads)])
