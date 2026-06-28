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
    claim_disagreement: bool = False
    disagreement_reason: str | None = None
    final_verdict: str
    final_score: float
    flagged_claims: list[dict[str, Any]] = Field(default_factory=list)
    behavioral_notes: dict[str, Any] = Field(default_factory=dict)
    self_improvement_seed: dict[str, Any] | None = None
    evaluators: list[EvaluatorResult]


class VerdictDocument(BaseModel):
    session_id: str
    overall_score: float = Field(..., ge=0.0, le=1.0)
    verdict: str
    confidence: str
    flagged_claims: list[dict[str, Any]] = Field(default_factory=list)
    behavioral_notes: dict[str, Any] = Field(default_factory=dict)
    evaluator_mode: str
    agreement: bool | None = None
    score_delta: float | None = None
    claim_disagreement: bool = False
    self_improvement_seed: dict[str, Any] | None = None


@dataclass(frozen=True)
class LocalVerdict:
    verdict: str
    score: float


def evaluator_prompt(session: dict[str, Any], local_report: dict[str, Any]) -> str:
    features = local_report.get("features", {})
    evidence = local_report.get("evidence", {})
    self_improvement = local_report.get("self_improvement_context") or {}
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
        "self_improvement": self_improvement,
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
    flagged_claims = []
    for item in data.get("flagged_claims") or []:
        if isinstance(item, dict):
            flagged_claims.append(item)
        else:
            flagged_claims.append({"claim": str(item), "reason": "provider_flagged"})
    reasoning_flags = [str(item) for item in data.get("reasoning_flags") or []]
    return EvaluatorResult(
        provider=provider,
        configured=configured,
        ok=verdict is not None and score is not None,
        verdict=verdict,
        score=score,
        confidence=confidence,
        flagged_claims=flagged_claims,
        reasoning_flags=reasoning_flags,
        latency_ms=latency_ms,
        raw_text=raw_text[:1500],
    )


def normalize_claim_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("claim") or value.get("text") or value.get("statement") or ""
    return " ".join(str(value).lower().strip().split())


def claim_key(item: dict[str, Any]) -> str:
    return normalize_claim_text(item)


FALSE_FLAG_PHRASES = (
    "nice revision",
    "good revision",
    "great job",
    "good job",
    "thanks",
    "thank you",
    "confidence is about",
    "my confidence is about",
)


def should_ignore_flagged_claim(item: dict[str, Any], prompt_version: int) -> bool:
    if prompt_version < 2:
        return False
    key = claim_key(item)
    if any(phrase in key for phrase in FALSE_FLAG_PHRASES):
        return True
    words = key.split()
    if len(words) <= 3 and not any(word in key for word in ("light", "oxygen", "glucose", "water", "carbon")):
        return True
    return False


def prompt_version_from_report(local_report: dict[str, Any] | None) -> int:
    context = (local_report or {}).get("self_improvement_context") or {}
    try:
        return int(context.get("prompt_version") or 1)
    except (TypeError, ValueError):
        return 1


def collect_flagged_claims(results: list[EvaluatorResult], prompt_version: int = 1) -> list[dict[str, Any]]:
    claims: dict[str, dict[str, Any]] = {}
    for result in results:
        if not result.ok:
            continue
        for item in result.flagged_claims:
            if should_ignore_flagged_claim(item, prompt_version):
                continue
            key = claim_key(item)
            if not key:
                continue
            current = claims.setdefault(
                key,
                {
                    "claim": item.get("claim") or item.get("text") or key,
                    "reason": item.get("reason") or "provider_flagged",
                    "providers": [],
                },
            )
            current["providers"].append(result.provider)
    return list(claims.values())


def has_claim_disagreement(results: list[EvaluatorResult], prompt_version: int = 1) -> bool:
    usable = [result for result in results if result.ok]
    if len(usable) < 2:
        return False
    claim_sets = [
        {
            claim_key(item)
            for item in result.flagged_claims
            if claim_key(item) and not should_ignore_flagged_claim(item, prompt_version)
        }
        for result in usable
    ]
    first = claim_sets[0]
    return any(claims != first for claims in claim_sets[1:])


def behavioral_notes_from_report(local_report: dict[str, Any]) -> dict[str, Any]:
    evidence = local_report.get("evidence", {})
    return {
        "rewatch": {
            "rate": evidence.get("rewatch_rate", 0),
            "note": "high_rewatch" if float(evidence.get("rewatch_rate") or 0) >= 1.5 else "normal",
        },
        "hesitation": {
            "average_ms": evidence.get("hesitation_ms", 0),
            "note": "high_hesitation" if float(evidence.get("hesitation_ms") or 0) >= 3500 else "normal",
        },
        "drawing": {
            "score": evidence.get("drawing_score", 0),
            "note": "strong_board_work" if float(evidence.get("drawing_score") or 0) >= 0.6 else "limited_board_work",
        },
    }


def adjust_confidence_for_behavior(confidence: str, behavioral_notes: dict[str, Any], uncertain: bool) -> str:
    if uncertain:
        return "low"
    levels = ["low", "medium", "high"]
    idx = levels.index(confidence)
    if behavioral_notes.get("hesitation", {}).get("note") == "high_hesitation":
        idx = max(0, idx - 1)
    if behavioral_notes.get("rewatch", {}).get("note") == "high_rewatch":
        idx = max(0, idx - 1)
    if behavioral_notes.get("drawing", {}).get("note") == "strong_board_work" and idx < 2:
        idx += 1
    return levels[idx]


def make_self_improvement_seed(
    local: LocalVerdict,
    results: list[EvaluatorResult],
    reason: str | None,
    flagged_claims: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not reason:
        return None
    return {
        "seed_type": "evaluator_disagreement",
        "reason": reason,
        "local_baseline": {"verdict": local.verdict, "score": local.score},
        "evaluator_outputs": [
            {
                "provider": result.provider,
                "verdict": result.verdict,
                "score": result.score,
                "confidence": result.confidence,
                "flagged_claims": result.flagged_claims,
                "reasoning_flags": result.reasoning_flags,
            }
            for result in results
        ],
        "flagged_claims": flagged_claims,
        "next_action": "Use as prompt-tuning seed, label the disagreement, then re-evaluate the session.",
    }


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


def merge_agreement(
    local: LocalVerdict,
    results: list[EvaluatorResult],
    local_report: dict[str, Any] | None = None,
) -> AgreementResult:
    usable = [result for result in results if result.ok and result.score is not None and result.verdict]
    prompt_version = prompt_version_from_report(local_report)
    flagged_claims = collect_flagged_claims(results, prompt_version)
    claim_disagreement = has_claim_disagreement(results, prompt_version)
    behavioral_notes = behavioral_notes_from_report(local_report or {})
    if not usable:
        confidence = adjust_confidence_for_behavior("medium", behavioral_notes, uncertain=False)
        return AgreementResult(
            mode="local_fallback",
            confidence=confidence,
            final_verdict=local.verdict,
            final_score=local.score,
            evaluators=results,
            flagged_claims=flagged_claims,
            behavioral_notes=behavioral_notes,
            disagreement_reason="No external evaluator returned a usable result.",
        )
    scores = [result.score for result in usable if result.score is not None]
    score_delta = round(max(scores) - min(scores), 3) if len(scores) > 1 else None
    verdicts = {str(result.verdict) for result in usable}
    score_disagreement = score_delta is not None and score_delta > 0.3
    verdict_disagreement = len(verdicts) != 1
    agreement = not verdict_disagreement and not score_disagreement and not claim_disagreement
    if agreement:
        final_score = round((sum(scores) + local.score) / (len(scores) + 1), 3)
        final_verdict = usable[0].verdict or local.verdict
        confidence = adjust_confidence_for_behavior("high" if len(usable) > 1 else "medium", behavioral_notes, False)
        reason = None
    else:
        final_score = round((sum(scores) + local.score) / (len(scores) + 1), 3)
        final_verdict = "uncertain" if score_disagreement else "flagged_for_review"
        confidence = adjust_confidence_for_behavior("low", behavioral_notes, True)
        reasons = []
        if verdict_disagreement:
            reasons.append("Evaluator verdicts diverged.")
        if claim_disagreement:
            reasons.append("Evaluators disagreed on flagged claims.")
        if score_disagreement:
            reasons.append("Score delta exceeded 0.3.")
        reason = " ".join(reasons)
    self_improvement_seed = make_self_improvement_seed(local, results, reason, flagged_claims)
    return AgreementResult(
        mode="dual_external" if len(usable) > 1 else f"{usable[0].provider}_only",
        confidence=confidence,
        score_delta=score_delta,
        agreement=agreement,
        claim_disagreement=claim_disagreement,
        disagreement_reason=reason,
        final_verdict=final_verdict,
        final_score=final_score,
        flagged_claims=flagged_claims,
        behavioral_notes=behavioral_notes,
        self_improvement_seed=self_improvement_seed,
        evaluators=results,
    )


def build_verdict_document(session_id: str, agreement: AgreementResult) -> VerdictDocument:
    return VerdictDocument(
        session_id=session_id,
        overall_score=agreement.final_score,
        verdict=agreement.final_verdict,
        confidence=agreement.confidence,
        flagged_claims=agreement.flagged_claims,
        behavioral_notes=agreement.behavioral_notes,
        evaluator_mode=agreement.mode,
        agreement=agreement.agreement,
        score_delta=agreement.score_delta,
        claim_disagreement=agreement.claim_disagreement,
        self_improvement_seed=agreement.self_improvement_seed,
    )


async def evaluate_with_external_llms(session: dict[str, Any], local_report: dict[str, Any]) -> AgreementResult:
    results = await asyncio.gather(
        call_gemini(session, local_report),
        call_minimax(session, local_report),
    )
    return merge_agreement(
        LocalVerdict(verdict=str(local_report["verdict"]), score=float(local_report["score"])),
        list(results),
        local_report,
    )
