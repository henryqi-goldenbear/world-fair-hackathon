from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field


class ClaimFeature(BaseModel):
    claim: str
    sentence_idx: int
    verifiable: bool


class BehavioralSignals(BaseModel):
    rewatch_rate: float = Field(..., description="Rewatch events per minute of session time")
    hesitation_ms: float = Field(..., description="Average long pause before student responses")
    drawing_score: float = Field(..., description="0-1 drawing complexity score from strokes and coverage")


class FeatureBundle(BaseModel):
    claims: list[ClaimFeature]
    behavior: BehavioralSignals
    metadata: dict[str, Any] = Field(default_factory=dict)


OPINION_MARKERS = {
    "i think",
    "i feel",
    "maybe",
    "probably",
    "not sure",
    "kind of",
    "i guess",
    "i believe",
}
FACTUAL_MARKERS = {
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "causes",
    "produces",
    "uses",
    "absorbs",
    "releases",
    "captures",
    "equals",
    "means",
    "because",
    "therefore",
}
ENTITY_PATTERNS = [
    {"label": "SCIENCE_TERM", "pattern": "photosynthesis"},
    {"label": "SCIENCE_TERM", "pattern": "chlorophyll"},
    {"label": "SCIENCE_TERM", "pattern": "carbon dioxide"},
    {"label": "SCIENCE_TERM", "pattern": "oxygen"},
    {"label": "SCIENCE_TERM", "pattern": "glucose"},
    {"label": "SCIENCE_TERM", "pattern": "water"},
    {"label": "MATH_TERM", "pattern": "denominator"},
    {"label": "MATH_TERM", "pattern": "numerator"},
    {"label": "MATH_TERM", "pattern": "fraction"},
    {"label": "PHYSICS_TERM", "pattern": "force"},
    {"label": "PHYSICS_TERM", "pattern": "acceleration"},
    {"label": "PHYSICS_TERM", "pattern": "velocity"},
]


@lru_cache(maxsize=1)
def nlp_pipeline() -> Any:
    try:
        import spacy

        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns(ENTITY_PATTERNS)
        return nlp
    except Exception:
        return None


def dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def split_sentences(text: str) -> list[tuple[str, list[str]]]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    nlp = nlp_pipeline()
    if nlp is not None:
        doc = nlp(clean)
        return [(sent.text.strip(), [ent.text for ent in sent.ents]) for sent in doc.sents if sent.text.strip()]
    parts = re.split(r"(?<=[.!?])\s+", clean)
    return [(part.strip(), []) for part in parts if part.strip()]


def normalize_turn_text(turn: dict[str, Any]) -> str:
    speaker = str(turn.get("speaker", "")).strip()
    text = str(turn.get("text", "")).strip()
    return f"{speaker}: {text}" if speaker else text


def transcript_sentences(transcript: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    sentences: list[tuple[str, list[str]]] = []
    for turn in transcript:
        # Tutor and student claims both matter for evaluator routing.
        sentences.extend(split_sentences(normalize_turn_text(turn)))
    return sentences


def is_verifiable(sentence: str, entities: list[str]) -> bool:
    lowered = sentence.lower()
    words = re.findall(r"[a-zA-Z0-9]+", lowered)
    if len(words) < 5:
        return False
    if sentence.strip().endswith("?"):
        return False
    if any(marker in lowered for marker in OPINION_MARKERS):
        return False
    has_number = bool(re.search(r"\d", sentence))
    has_factual_marker = any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in FACTUAL_MARKERS)
    has_entity = bool(entities) or any(pattern["pattern"] in lowered for pattern in ENTITY_PATTERNS)
    return has_number or (has_factual_marker and has_entity)


def extract_claims(transcript: list[dict[str, Any]]) -> list[ClaimFeature]:
    claims: list[ClaimFeature] = []
    for idx, (sentence, entities) in enumerate(transcript_sentences(transcript)):
        claim = re.sub(r"^(Tutor|Student):\s*", "", sentence).strip()
        if not claim:
            continue
        claims.append(
            ClaimFeature(
                claim=claim,
                sentence_idx=idx,
                verifiable=is_verifiable(claim, entities),
            )
        )
    return claims


def event_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") or {}
    return payload if isinstance(payload, dict) else {}


def rewatch_rate(events: list[dict[str, Any]], duration_ms: float) -> float:
    rewatch_count = sum(1 for event in events if str(event.get("type", "")).lower() == "rewatch")
    minutes = max(duration_ms / 60000.0, 1 / 60)
    return round(rewatch_count / minutes, 3)


def hesitation_ms(transcript: list[dict[str, Any]], events: list[dict[str, Any]]) -> float:
    pauses = [
        float(event_payload(event).get("duration_ms", 0) or 0)
        for event in events
        if str(event.get("type", "")).lower() == "pause"
    ]
    speaker_pauses: list[float] = []
    last_tutor_t: float | None = None
    for turn in sorted(transcript, key=lambda item: float(item.get("t", 0) or 0)):
        speaker = str(turn.get("speaker", "")).lower()
        t = float(turn.get("t", 0) or 0)
        if speaker == "tutor":
            last_tutor_t = t
        elif speaker == "student" and last_tutor_t is not None:
            delta = max(0.0, t - last_tutor_t)
            if delta >= 2500:
                speaker_pauses.append(delta)
            last_tutor_t = None
    combined = pauses + speaker_pauses
    if not combined:
        return 0.0
    return round(sum(combined) / len(combined), 1)


def numbers_from_svg(svg: str) -> list[float]:
    return [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", svg)]


def drawing_complexity(events: list[dict[str, Any]], drawings: list[str]) -> float:
    stroke_events = sum(1 for event in events if str(event.get("type", "")).lower() in {"add", "addmany"})
    svg_strokes = 0
    coverage_values: list[float] = []
    for drawing in drawings:
        svg_strokes += len(re.findall(r"<(?:path|line|rect|ellipse|text)\b", drawing, flags=re.I))
        nums = numbers_from_svg(drawing)
        xs = nums[0::2]
        ys = nums[1::2]
        if xs and ys:
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            coverage_values.append(max(0.0, min(1.0, (width * height) / (960 * 640))))
    stroke_count = max(stroke_events, svg_strokes)
    coverage = max(coverage_values) if coverage_values else 0.0
    stroke_score = 1 - math.exp(-stroke_count / 12.0)
    coverage_score = 1 - math.exp(-coverage * 4.0)
    return round(max(0.0, min(1.0, 0.7 * stroke_score + 0.3 * coverage_score)), 3)


def extract_behavioral_signals(session: dict[str, Any], streamed_events: list[dict[str, Any]] | None = None) -> BehavioralSignals:
    metadata = session.get("metadata") or {}
    transcript = session.get("transcript") or []
    events = list(session.get("events") or [])
    if streamed_events:
        events.extend(streamed_events)
    drawings = session.get("drawings") or []
    duration_ms = float(metadata.get("duration_ms") or max([float(e.get("t", 0) or 0) for e in events] + [1.0]))
    return BehavioralSignals(
        rewatch_rate=rewatch_rate(events, duration_ms),
        hesitation_ms=hesitation_ms(transcript, events),
        drawing_score=drawing_complexity(events, drawings),
    )


def extract_feature_bundle(session: dict[str, Any], streamed_events: list[dict[str, Any]] | None = None) -> FeatureBundle:
    transcript = session.get("transcript") or []
    events = list(session.get("events") or [])
    if streamed_events:
        events.extend(streamed_events)
    claims = extract_claims(transcript)
    return FeatureBundle(
        claims=claims,
        behavior=extract_behavioral_signals(session, streamed_events),
        metadata={
            "claim_count": len(claims),
            "verifiable_claim_count": sum(1 for claim in claims if claim.verifiable),
            "event_count": len(events),
            "extractor": "spacy_entity_ruler_rules" if nlp_pipeline() is not None else "regex_rules",
        },
    )
