#!/usr/bin/env python3
"""
Export FerbAI agent-session data as the external session-output payload.

Shape:
{
  "session_id": "...",
  "timestamp": "...",
  "transcript": [{"t": 0, "speaker": "Student", "text": "..."}],
  "events": [{"type": "pause", "t": 12000, "payload": {...}}],
  "drawings": ["<svg ...>...</svg>"],
  "metadata": {"duration_ms": 36000, "subject": "...", "student_id": "..."}
}
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TURN_TIMES_MS = [0, 4200, 8700, 13100, 18100, 23000, 27700, 32100]


def xml_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def element_to_svg_fragment(element: dict[str, Any]) -> str:
    color = xml_escape(element.get("color", "black"))
    width = xml_escape(element.get("width", 3))
    kind = element.get("type")
    if kind == "text":
        return (
            f'<text x="{xml_escape(element.get("x", 0))}" y="{xml_escape(element.get("y", 0))}" '
            f'font-size="{xml_escape(element.get("size", 24))}" fill="{color}">'
            f'{xml_escape(element.get("text", ""))}</text>'
        )
    if kind in {"line", "arrow"}:
        marker = ' marker-end="url(#arrow)"' if kind == "arrow" else ""
        return (
            f'<line x1="{xml_escape(element.get("x1", 0))}" y1="{xml_escape(element.get("y1", 0))}" '
            f'x2="{xml_escape(element.get("x2", 0))}" y2="{xml_escape(element.get("y2", 0))}" '
            f'stroke="{color}" stroke-width="{width}" stroke-linecap="round"{marker}/>'
        )
    if kind in {"rect", "highlight"}:
        opacity = ' opacity="0.25"' if kind == "highlight" else ""
        fill = color if kind == "highlight" else "none"
        return (
            f'<rect x="{xml_escape(element.get("x", 0))}" y="{xml_escape(element.get("y", 0))}" '
            f'width="{xml_escape(element.get("w", 0))}" height="{xml_escape(element.get("h", 0))}" '
            f'fill="{fill}" stroke="{color}" stroke-width="{width}"{opacity}/>'
        )
    if kind == "ellipse":
        x = float(element.get("x", 0))
        y = float(element.get("y", 0))
        w = float(element.get("w", 0))
        h = float(element.get("h", 0))
        return (
            f'<ellipse cx="{x + w / 2:g}" cy="{y + h / 2:g}" rx="{w / 2:g}" ry="{h / 2:g}" '
            f'fill="none" stroke="{color}" stroke-width="{width}"/>'
        )
    if kind == "path":
        points = element.get("points") or []
        if not points:
            return ""
        path = "M " + " L ".join(f'{p.get("x", 0)} {p.get("y", 0)}' for p in points)
        return f'<path d="{xml_escape(path)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>'
    return f"<!-- unsupported FerbAI element: {xml_escape(kind)} -->"


def elements_to_svg(elements: list[dict[str, Any]]) -> str:
    body = "\n  ".join(fragment for fragment in (element_to_svg_fragment(el) for el in elements) if fragment)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640" viewBox="0 0 960 640">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="currentColor"/>
    </marker>
  </defs>
  {body}
</svg>"""


def chat_transcript(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transcript = []
    turn_index = 0
    for message in messages:
        if message.get("pending") or message.get("error"):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        transcript.append(
            {
                "t": TURN_TIMES_MS[turn_index] if turn_index < len(TURN_TIMES_MS) else turn_index * 4200,
                "speaker": "Student" if role == "user" else "Tutor",
                "text": str(message.get("text", "")).strip(),
            }
        )
        turn_index += 1
    return transcript


def interaction_events(session: dict[str, Any]) -> list[dict[str, Any]]:
    recording = session["ferbai"]["recording"]
    events = []
    for event in recording.get("events", []):
        payload = {k: v for k, v in event.items() if k not in {"type", "t"}}
        events.append({"type": event.get("type", "recording_event"), "t": event.get("t", 0), "payload": payload})

    duration = int(recording.get("durationMs") or 0)
    events.extend(
        [
            {
                "type": "click",
                "t": 1200,
                "payload": {"target": "whiteboard", "tool": "text", "agent_generated": True},
            },
            {
                "type": "pause",
                "t": min(12000, duration),
                "payload": {"reason": "agent_student_thinking", "duration_ms": 1800},
            },
            {
                "type": "rewatch",
                "t": min(24000, duration),
                "payload": {"from_ms": 8700, "to_ms": 18100, "reason": "misconception_review"},
            },
        ]
    )
    return sorted(events, key=lambda item: item["t"])


def session_to_payload(session: dict[str, Any]) -> dict[str, Any]:
    persona = session["persona"]
    recording = session["ferbai"]["recording"]
    final_snapshot = (recording.get("snapshots") or [{}])[-1]
    elements = final_snapshot.get("elements") or []
    return {
        "session_id": session["sessionId"],
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "transcript": chat_transcript(session["ferbai"]["chatMessages"]),
        "events": interaction_events(session),
        "drawings": [elements_to_svg(elements)],
        "metadata": {
            "duration_ms": recording.get("durationMs"),
            "subject": persona.get("subject"),
            "student_id": persona.get("id"),
            "student_kind": persona.get("kind", "agent_student"),
            "agent_generated": session.get("generatedBy") == "agent_student_swarm",
            "human_student": session.get("humanStudent", False),
            "recording_id": session.get("recordingId"),
            "understanding_band": persona.get("understandingBand"),
            "confidence": persona.get("confidence"),
        },
    }


def post_payload(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"content-type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"POST failed: HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise SystemExit(f"POST failed: {exc.reason}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FerbAI agent sessions as FastAPI session-output payloads.")
    parser.add_argument("--input", default="ferbai_agent_swarm_sessions.json")
    parser.add_argument("--out", default="ferbai_session_outputs.json")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--post-url", default="")
    args = parser.parse_args()

    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    payloads = [session_to_payload(session) for session in source["sessions"][: args.limit]]
    Path(args.out).write_text(json.dumps(payloads, indent=2), encoding="utf-8")
    print(f"Wrote {len(payloads)} FerbAI session-output payload(s) to {args.out}")

    if args.post_url:
      for payload in payloads:
          result = post_payload(args.post_url, payload)
          print(json.dumps({"posted": payload["session_id"], "response": result}, indent=2))


if __name__ == "__main__":
    main()
