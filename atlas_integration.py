from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


EMBEDDING_DIMENSIONS = 64


def stable_embedding(text: str, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    vector = [0.0] * dimensions
    tokens = [token.strip(".,;:!?()[]{}\"'").lower() for token in text.split()]
    for token in tokens:
        if not token:
            continue
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


def flagged_claim_embeddings(report: dict[str, Any]) -> list[dict[str, Any]]:
    embeddings = []
    for claim in report.get("flagged_claims") or []:
        text = str(claim.get("claim") or "")
        if not text:
            continue
        embeddings.append(
            {
                "claim": text,
                "reason": claim.get("reason"),
                "providers": claim.get("providers", []),
                "embedding": stable_embedding(text),
            }
        )
    return embeddings


def prepare_session_document(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **record,
        "collection_role": "raw_ingest",
        "atlas_provider": "gcp",
    }


def prepare_verdict_document(report: dict[str, Any]) -> dict[str, Any]:
    return {
        **report,
        "collection_role": "final_evaluated_output",
        "atlas_provider": "gcp",
        "flagged_claim_embeddings": flagged_claim_embeddings(report),
    }


def prepare_disagreement_document(session_id: str, seed: dict[str, Any], created_at: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "created_at": created_at,
        "collection_role": "training_seed_pool",
        "atlas_provider": "gcp",
        **seed,
        "flagged_claim_embeddings": [
            {
                "claim": str(claim.get("claim") or ""),
                "reason": claim.get("reason"),
                "embedding": stable_embedding(str(claim.get("claim") or "")),
            }
            for claim in seed.get("flagged_claims", [])
            if claim.get("claim")
        ],
    }


def load_personas(path: Path = Path("synthetic_student_swarm.json")) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    personas: dict[str, dict[str, Any]] = {}
    for session in data.get("sessions", []):
        persona = session.get("persona") or {}
        persona_id = persona.get("persona_id")
        if not persona_id:
            continue
        personas[persona_id] = {
            **persona,
            "collection_role": "synthetic_student_profile",
            "atlas_provider": "gcp",
            "source_swarm_id": data.get("swarm_id"),
            "ground_truth": session.get("ground_truth", {}),
        }
    return list(personas.values())


async def ensure_atlas_indexes(db: Any) -> dict[str, Any]:
    created: dict[str, list[str]] = {
        "sessions": [],
        "events": [],
        "verdicts": [],
        "disagreements": [],
        "personas": [],
    }
    await db.sessions.create_index("session_id", unique=True)
    created["sessions"].append("session_id_unique")
    await db.sessions.create_index("metadata.student_id")
    created["sessions"].append("metadata_student_id")
    await db.sessions.create_index("timestamp")
    created["sessions"].append("timestamp")

    await db.events.create_index("session_id")
    created["events"].append("session_id")
    await db.events.create_index("received_at")
    created["events"].append("received_at")
    await db.events.create_index([("type", 1), ("t", 1)])
    created["events"].append("type_t")

    await db.verdicts.create_index("session_id", unique=True)
    created["verdicts"].append("session_id_unique")
    await db.verdicts.create_index("generated_at")
    created["verdicts"].append("generated_at")
    await db.verdicts.create_index([("verdict", 1), ("confidence", 1)])
    created["verdicts"].append("verdict_confidence")
    await db.verdicts.create_index([("flagged_claims.claim", "text")])
    created["verdicts"].append("flagged_claim_text")

    await db.disagreements.create_index("session_id", unique=True)
    created["disagreements"].append("session_id_unique")
    await db.disagreements.create_index("created_at")
    created["disagreements"].append("created_at")
    await db.disagreements.create_index([("flagged_claims.claim", "text")])
    created["disagreements"].append("flagged_claim_text")

    await db.personas.create_index("persona_id", unique=True)
    created["personas"].append("persona_id_unique")
    await db.personas.create_index([("subject", 1), ("understanding_band", 1)])
    created["personas"].append("subject_understanding_band")
    await db.personas.create_index("ability")
    created["personas"].append("ability")
    return created


async def seed_personas(db: Any, personas: list[dict[str, Any]]) -> int:
    count = 0
    for persona in personas:
        await db.personas.update_one(
            {"persona_id": persona["persona_id"]},
            {"$set": persona},
            upsert=True,
        )
        count += 1
    return count


async def verdict_drift(db: Any, limit: int = 50) -> list[dict[str, Any]]:
    cursor = db.verdicts.find(
        {},
        {
            "_id": False,
            "session_id": True,
            "generated_at": True,
            "verdict": True,
            "overall_score": True,
            "confidence": True,
            "evidence.llm_mode": True,
        },
    ).sort("generated_at", -1).limit(limit)
    return [doc async for doc in cursor]


async def swarm_analytics(db: Any) -> dict[str, Any]:
    verdict_counts = [
        doc
        async for doc in db.verdicts.aggregate(
            [
                {"$group": {"_id": "$verdict", "count": {"$sum": 1}, "avg_score": {"$avg": "$overall_score"}}},
                {"$sort": {"count": -1}},
            ]
        )
    ]
    confidence_counts = [
        doc
        async for doc in db.verdicts.aggregate(
            [
                {"$group": {"_id": "$confidence", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
            ]
        )
    ]
    disagreement_count = await db.disagreements.count_documents({})
    persona_count = await db.personas.count_documents({})
    session_count = await db.sessions.count_documents({})
    verdict_count = await db.verdicts.count_documents({})
    return {
        "session_count": session_count,
        "verdict_count": verdict_count,
        "disagreement_seed_count": disagreement_count,
        "persona_count": persona_count,
        "verdict_counts": verdict_counts,
        "confidence_counts": confidence_counts,
    }


def atlas_search_index_definitions() -> dict[str, Any]:
    return {
        "text_search": {
            "collection": "verdicts",
            "name": "flagged_claim_text_search",
            "definition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "flagged_claims": {
                            "type": "document",
                            "fields": {
                                "claim": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                        }
                    },
                }
            },
        },
        "vector_search": {
            "collection": "verdicts",
            "name": "flagged_claim_vector_index",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "flagged_claim_embeddings.embedding",
                        "numDimensions": EMBEDDING_DIMENSIONS,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "verdict"},
                    {"type": "filter", "path": "confidence"},
                ]
            },
        },
    }
