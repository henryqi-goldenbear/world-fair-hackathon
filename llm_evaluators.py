from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


EVALUATOR_TIMEOUT_SECONDS = float(os.getenv("EVALUATOR_TIMEOUT_SECONDS", "8"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY") or os.getenv("MINIMAX_ACCESS_TOKEN", "")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-Text-01")
MINIMAX_API_URL = os.getenv("MINIMAX_API_URL", "https://api.minimax.io/v1/text/chatcompletion_v2")


class EvaluatorResult(BaseModel):
    provider: str
    configured: bool
    ok: bool
    verdict: str | None = None
    score: float | None = None
    confidence: str | None = None
    flagged_claims: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_flags: list[str] = Field(default_factory=list)
    error: str | None = None
    latency_ms: int | None = None
    raw_text: str | None = None


class AgreementResult(BaseModel):
    mode: str
    confidence: str
    score_delta: float | None = None
    agreement: bool | None = None
    disagreement_reason: str | None = None
    final_verdict: str
    final_score: float
    evaluators: list[EvaluatorResult]


@dataclass(frozen=True)
class LocalVerdict:
    verdict: str
    score: float


def evaluator_prompt(session: dict[str, Any], local_report: dict[str, Any]) -> str:
    features = local_report.get("features", {})
    evidence = local_report.get("evidence", {})
    compact_transcript = [
        {
            "t": turn.get("t"),
            "speaker": turn.get("speaker"),
            "text": str(turn.get("text", ""))[:500],
        }
        for turn in session.get("transcript", [])[:16]
    ]
    payload = {
        "task": "Evaluate an agent-student tutoring session for learning understanding.",
        "requirements": {
            "verdict_values": ["on_track", "needs_targeted_support", "needs_review", "uncertain"],
            "score": "0.0 to 1.0",
            "confidence": "low, medium, or high",
            "flagged_claims": "Claims that are incorrect, unverifiable, or pedagogically risky.",
            "reasoning_flags": "Top 3 concise reasons for the verdict.",
        },
        "local_baseline": {
            "verdict": local_report.get("verdict"),
            "score": local_report.get("score"),
            "evidence": evidence,
        },
        "features": features,
        "transcript": compact_transcript,
        "drawings_count": len(session.get("drawings", [])),
    }
    return (
        "Return only JSON with keys: verdict, score, confidence, flagged_claims, reasoning_flags.\n"
        + json.dumps(payload, ensure_ascii=True)
    )


def parse_jsonish(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.strip("`")
        if clean.lower().startswith("json"):
            clean = clean[4:].strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            return json.loads(clean[start : end + 1])
        raise


def coerce_result(provider: str, configured: bool, data: dict[str, Any], latency_ms: int, raw_text: str) -> EvaluatorResult:
    score = data.get("score")
    try:
        score = round(max(0.0, min(1.0, float(score))), 3)
    except (TypeError, ValueError):
        score = None
    verdict = data.get("verdict")
    if verdict not in {"on_track", "needs_targeted_support", "needs_review", "uncertain"}:
        verdict = None
    confidence = data.get("confidence")
    if confidence not in {"low", "medium", "high"}:
        confidence = None
    return EvaluatorResult(
        provider=provider,
        configured=configured,
        ok=verdict is not None and score is not None,
        verdict=verdict,
        score=score,
        confidence=confidence,
        flagged_claims=data.get("flagged_claims") or [],
        reasoning_flags=data.get("reasoning_flags") or [],
        latency_ms=latency_ms,
        raw_text=raw_text[:1500],
    )


async def call_gemini(session: dict[str, Any], local_report: dict[str, Any]) -> EvaluatorResult:
    import httpx

    if not GEMINI_API_KEY:
        return EvaluatorResult(provider="gemini", configured=False, ok=False, error="GEMINI_API_KEY missing")
    prompt = evaluator_prompt(session, local_report)
    url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent"
    started = asyncio.get_running_loop().time()
    try:
        async with httpx.AsyncClient(timeout=EVALUATOR_TIMEOUT_SECONDS) as client:
            response = await client.post(
                url,
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "responseMimeType": "application/json",
                    },
                },
            )
            response.raise_for_status()
        elapsed = int((asyncio.get_running_loop().time() - started) * 1000)
        body = response.json()
        raw_text = body["candidates"][0]["content"]["parts"][0]["text"]
        return coerce_result("gemini", True, parse_jsonish(raw_text), elapsed, raw_text)
    except Exception as exc:
        elapsed = int((asyncio.get_running_loop().time() - started) * 1000)
        return EvaluatorResult(provider="gemini", configured=True, ok=False, error=str(exc), latency_ms=elapsed)


def minimax_payload(prompt: str) -> dict[str, Any]:
    return {
        "model": MINIMAX_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an independent tutoring-session evaluator. Return strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "stream": False,
    }


def extract_minimax_text(body: dict[str, Any]) -> str:
    if body.get("choices"):
        choice = body["choices"][0]
        message = choice.get("message") or {}
        return message.get("content") or choice.get("text") or ""
    if body.get("reply"):
        return str(body["reply"])
    if body.get("data") and isinstance(body["data"], dict):
        return str(body["data"].get("text") or body["data"].get("reply") or "")
    return json.dumps(body)


async def call_minimax(session: dict[str, Any], local_report: dict[str, Any]) -> EvaluatorResult:
    import httpx

    if not MINIMAX_API_KEY:
        return EvaluatorResult(provider="minimax", configured=False, ok=False, error="MINIMAX_API_KEY missing")
    prompt = evaluator_prompt(session, local_report)
    started = asyncio.get_running_loop().time()
    try:
        async with httpx.AsyncClient(timeout=EVALUATOR_TIMEOUT_SECONDS) as client:
            response = await client.post(
                MINIMAX_API_URL,
                headers={
                    "Authorization": f"Bearer {MINIMAX_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=minimax_payload(prompt),
            )
            response.raise_for_status()
        elapsed = int((asyncio.get_running_loop().time() - started) * 1000)
        raw_text = extract_minimax_text(response.json())
        return coerce_result("minimax", True, parse_jsonish(raw_text), elapsed, raw_text)
    except Exception as exc:
        elapsed = int((asyncio.get_running_loop().time() - started) * 1000)
        return EvaluatorResult(provider="minimax", configured=True, ok=False, error=str(exc), latency_ms=elapsed)


def merge_agreement(local: LocalVerdict, results: list[EvaluatorResult]) -> AgreementResult:
    usable = [result for result in results if result.ok and result.score is not None and result.verdict]
    if not usable:
        return AgreementResult(
            mode="local_fallback",
            confidence="medium",
            final_verdict=local.verdict,
            final_score=local.score,
            evaluators=results,
            disagreement_reason="No external evaluator returned a usable result.",
        )
    scores = [result.score for result in usable if result.score is not None]
    score_delta = round(max(scores) - min(scores), 3) if len(scores) > 1 else None
    verdicts = {str(result.verdict) for result in usable}
    agreement = len(verdicts) == 1 and (score_delta is None or score_delta <= 0.3)
    if agreement:
        final_score = round(sum(scores) / len(scores), 3)
        final_verdict = usable[0].verdict or local.verdict
        confidence = "high" if len(usable) > 1 else "medium"
        reason = None
    else:
        final_score = round((sum(scores) + local.score) / (len(scores) + 1), 3)
        final_verdict = "uncertain"
        confidence = "low"
        reason = "Evaluator verdicts diverged or score delta exceeded 0.3."
    return AgreementResult(
        mode="dual_external" if len(usable) > 1 else f"{usable[0].provider}_only",
        confidence=confidence,
        score_delta=score_delta,
        agreement=agreement,
        disagreement_reason=reason,
        final_verdict=final_verdict,
        final_score=final_score,
        evaluators=results,
    )


async def evaluate_with_external_llms(session: dict[str, Any], local_report: dict[str, Any]) -> AgreementResult:
    results = await asyncio.gather(
        call_gemini(session, local_report),
        call_minimax(session, local_report),
    )
    return merge_agreement(
        LocalVerdict(verdict=str(local_report["verdict"]), score=float(local_report["score"])),
        list(results),
    )
