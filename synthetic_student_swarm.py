#!/usr/bin/env python3
"""
Generate a synthetic student swarm for FerbAI-style tutoring evaluations.

The output is JSON shaped like session payloads: each synthetic student has a
stable persona, varied subject understanding, a transcript, behavioral signals,
simple drawing strokes, and ground-truth labels for evaluator scoring.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SUBJECT_MODELS: dict[str, dict[str, list[str]]] = {
    "photosynthesis": {
        "concepts": [
            "plants use light energy",
            "carbon dioxide enters through leaves",
            "water is absorbed by roots",
            "glucose stores chemical energy",
            "oxygen is released as a byproduct",
            "chlorophyll captures light",
        ],
        "misconceptions": [
            "plants get most of their food from soil",
            "oxygen is the main input plants consume",
            "photosynthesis only happens in flowers",
            "sunlight turns directly into oxygen",
            "plants do not respire",
        ],
    },
    "fractions": {
        "concepts": [
            "the denominator names equal parts",
            "the numerator counts selected parts",
            "equivalent fractions can name the same value",
            "common denominators help with addition",
            "fractions can be placed on a number line",
            "simplifying preserves value",
        ],
        "misconceptions": [
            "larger denominators always mean larger fractions",
            "add numerators and denominators separately",
            "equivalent fractions are different amounts",
            "fractions cannot be greater than one",
            "simplifying changes the amount",
        ],
    },
    "newtonian mechanics": {
        "concepts": [
            "forces change motion",
            "net force determines acceleration",
            "mass resists acceleration",
            "velocity includes speed and direction",
            "gravity is a force",
            "friction opposes motion",
        ],
        "misconceptions": [
            "motion requires a continuous forward force",
            "heavier objects always fall faster",
            "velocity and acceleration are the same",
            "forces only exist when objects touch",
            "friction always stops motion immediately",
        ],
    },
}

FALLBACK_MODEL = {
    "concepts": [
        "define the central idea",
        "connect cause and effect",
        "use evidence for claims",
        "apply the idea to a new example",
        "compare related concepts",
        "explain limits or exceptions",
    ],
    "misconceptions": [
        "confuses the definition with an example",
        "uses memorized language without causal reasoning",
        "overgeneralizes one case to every case",
        "misses an important prerequisite idea",
        "treats related terms as interchangeable",
    ],
}

NAMES = [
    "Amina",
    "Ben",
    "Carla",
    "Dev",
    "Elena",
    "Felix",
    "Grace",
    "Hana",
    "Isaac",
    "Jules",
    "Kai",
    "Lina",
    "Mateo",
    "Nora",
    "Owen",
    "Priya",
    "Quinn",
    "Rafi",
    "Sofia",
    "Theo",
    "Uma",
    "Vera",
    "Will",
    "Yara",
    "Zane",
]

LEARNING_STYLES = [
    "visual",
    "verbal",
    "procedural",
    "example-driven",
    "reflective",
    "guess-and-check",
]

LANGUAGE_PROFILES = [
    "native English",
    "English language learner",
    "concise speaker",
    "verbose explainer",
    "uses informal language",
]

BEHAVIORS = [
    "asks clarifying questions",
    "answers quickly but revises",
    "hesitates before committing",
    "draws while thinking",
    "overconfident when uncertain",
    "checks work aloud",
]


@dataclass
class StudentPersona:
    persona_id: str
    name: str
    subject: str
    ability: float
    understanding_band: str
    confidence: float
    pace: str
    learning_style: str
    language_profile: str
    behavior: str
    mastered_concepts: list[str]
    misconceptions: list[str]
    support_need: str


def band_for_ability(ability: float) -> str:
    if ability < 0.3:
        return "novice"
    if ability < 0.55:
        return "developing"
    if ability < 0.8:
        return "proficient"
    return "advanced"


def support_need_for_band(band: str) -> str:
    return {
        "novice": "high scaffolding",
        "developing": "guided practice",
        "proficient": "targeted feedback",
        "advanced": "extension challenge",
    }[band]


def subject_model(subject: str) -> dict[str, list[str]]:
    return SUBJECT_MODELS.get(subject.lower(), FALLBACK_MODEL)


def generate_persona(index: int, subject: str, rng: random.Random) -> StudentPersona:
    model = subject_model(subject)
    ability = max(0.05, min(0.98, rng.betavariate(2.0, 2.0)))
    band = band_for_ability(ability)
    concept_count = {
        "novice": rng.randint(1, 2),
        "developing": rng.randint(2, 3),
        "proficient": rng.randint(3, 5),
        "advanced": rng.randint(5, len(model["concepts"])),
    }[band]
    misconception_count = {
        "novice": rng.randint(2, 3),
        "developing": rng.randint(1, 3),
        "proficient": rng.randint(0, 1),
        "advanced": 0 if rng.random() < 0.8 else 1,
    }[band]

    mastered = rng.sample(model["concepts"], k=min(concept_count, len(model["concepts"])))
    misconceptions = rng.sample(
        model["misconceptions"], k=min(misconception_count, len(model["misconceptions"]))
    )
    confidence = max(0.05, min(0.98, ability + rng.uniform(-0.25, 0.25)))
    if rng.random() < 0.18:
        confidence = max(0.1, min(0.95, 1.0 - ability + rng.uniform(-0.1, 0.1)))

    return StudentPersona(
        persona_id=f"stu_{index:03d}_{uuid.uuid4().hex[:8]}",
        name=f"{rng.choice(NAMES)} {index:02d}",
        subject=subject,
        ability=round(ability, 3),
        understanding_band=band,
        confidence=round(confidence, 3),
        pace=rng.choice(["slow", "steady", "fast"]),
        learning_style=rng.choice(LEARNING_STYLES),
        language_profile=rng.choice(LANGUAGE_PROFILES),
        behavior=rng.choice(BEHAVIORS),
        mastered_concepts=mastered,
        misconceptions=misconceptions,
        support_need=support_need_for_band(band),
    )


def timestamp(base: datetime, seconds: int) -> str:
    return (base + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def generate_transcript(persona: StudentPersona, rng: random.Random) -> list[dict[str, str]]:
    base = datetime.now(timezone.utc)
    mastered = persona.mastered_concepts
    weak_spot = persona.misconceptions[0] if persona.misconceptions else None
    opening = {
        "novice": "I kind of remember the words, but I am not sure how they fit together.",
        "developing": "I can explain part of it, but I get mixed up in the middle.",
        "proficient": "I think I can explain it if I go step by step.",
        "advanced": "I can explain it and test it with a different example.",
    }[persona.understanding_band]
    correct_claim = mastered[0] if mastered else "the main idea"
    second_claim = mastered[1] if len(mastered) > 1 else correct_claim

    student_error = (
        f"I think {weak_spot}."
        if weak_spot
        else f"The key is that {correct_claim}, and that connects to {second_claim}."
    )
    repair = (
        f"Wait, I should revise that: {correct_claim} matters, but my earlier idea was too broad."
        if weak_spot and persona.ability > 0.45
        else "I need another hint before I can fix that."
        if weak_spot
        else f"So the stronger explanation is that {correct_claim} because {second_claim}."
    )

    return [
        {
            "t": timestamp(base, 0),
            "speaker": "tutor",
            "text": f"Explain the most important idea in {persona.subject}.",
        },
        {"t": timestamp(base, rng.randint(3, 9)), "speaker": "student", "text": opening},
        {
            "t": timestamp(base, rng.randint(12, 20)),
            "speaker": "student",
            "text": student_error,
        },
        {
            "t": timestamp(base, rng.randint(24, 35)),
            "speaker": "tutor",
            "text": "What evidence or mechanism supports that answer?",
        },
        {"t": timestamp(base, rng.randint(38, 58)), "speaker": "student", "text": repair},
        {
            "t": timestamp(base, rng.randint(62, 82)),
            "speaker": "student",
            "text": f"My confidence is about {round(persona.confidence * 100)} percent.",
        },
    ]


def drawing_sequence(persona: StudentPersona, rng: random.Random) -> list[dict[str, Any]]:
    node_count = {
        "novice": 2,
        "developing": 3,
        "proficient": 4,
        "advanced": 5,
    }[persona.understanding_band]
    strokes = []
    for idx in range(node_count):
        x = 40 + idx * 80 + rng.randint(-10, 10)
        y = 55 + rng.randint(-20, 35)
        strokes.append(
            {
                "type": "node",
                "label": (persona.mastered_concepts + persona.misconceptions)[
                    idx % max(1, len(persona.mastered_concepts + persona.misconceptions))
                ],
                "svg_path": f"M{x},{y} m-24,0 a24,16 0 1,0 48,0 a24,16 0 1,0 -48,0",
            }
        )
        if idx:
            strokes.append(
                {
                    "type": "edge",
                    "label": "connects to" if persona.ability > 0.5 else "maybe",
                    "svg_path": f"M{x - 56},{y} L{x - 24},{y}",
                }
            )
    return strokes


def behavioral_signals(persona: StudentPersona, transcript: list[dict[str, str]]) -> dict[str, Any]:
    student_turns = [turn["text"] for turn in transcript if turn["speaker"] == "student"]
    avg_words = sum(len(turn.split()) for turn in student_turns) / max(1, len(student_turns))
    hesitation = {
        "slow": 0.72,
        "steady": 0.38,
        "fast": 0.16,
    }[persona.pace]
    return {
        "avg_student_turn_words": round(avg_words, 1),
        "hesitation_score": hesitation,
        "self_revision": any("revise" in turn.lower() or "wait" in turn.lower() for turn in student_turns),
        "confidence": persona.confidence,
        "pace": persona.pace,
        "behavior": persona.behavior,
    }


def ground_truth(persona: StudentPersona) -> dict[str, Any]:
    if persona.misconceptions:
        verdict = "needs_review" if persona.ability < 0.65 else "partially_correct"
    else:
        verdict = "correct" if persona.ability >= 0.55 else "fragile_correct"
    return {
        "mastery_score": persona.ability,
        "understanding_band": persona.understanding_band,
        "has_misconception": bool(persona.misconceptions),
        "misconceptions": persona.misconceptions,
        "support_need": persona.support_need,
        "expected_verdict": verdict,
    }


def generate_session(persona: StudentPersona, rng: random.Random) -> dict[str, Any]:
    transcript = generate_transcript(persona, rng)
    return {
        "session_id": f"sess_{uuid.uuid4().hex}",
        "student_id": persona.persona_id,
        "subject": persona.subject,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "persona": asdict(persona),
        "transcript": transcript,
        "behavioral_signals": behavioral_signals(persona, transcript),
        "drawing_sequence": drawing_sequence(persona, rng),
        "ground_truth": ground_truth(persona),
    }


def generate_swarm(count: int, subject: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    personas = [generate_persona(index + 1, subject, rng) for index in range(count)]
    sessions = [generate_session(persona, rng) for persona in personas]
    by_band: dict[str, int] = {"novice": 0, "developing": 0, "proficient": 0, "advanced": 0}
    for persona in personas:
        by_band[persona.understanding_band] += 1
    return {
        "swarm_id": f"swarm_{uuid.uuid4().hex[:12]}",
        "subject": subject,
        "count": count,
        "seed": seed,
        "ability_distribution": by_band,
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic FerbAI student swarm sessions.")
    parser.add_argument("--subject", default="photosynthesis", help="Subject to simulate.")
    parser.add_argument("--count", type=int, default=50, help="Number of synthetic students.")
    parser.add_argument("--seed", type=int, default=20260627, help="Deterministic random seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("synthetic_student_swarm.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be at least 1")

    swarm = generate_swarm(args.count, args.subject, args.seed)
    args.out.write_text(json.dumps(swarm, indent=2), encoding="utf-8")
    print(
        f"Wrote {args.count} synthetic {args.subject} student sessions to {args.out} "
        f"with distribution {swarm['ability_distribution']}"
    )


if __name__ == "__main__":
    main()
