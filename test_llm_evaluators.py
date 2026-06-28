import unittest

from llm_evaluators import EvaluatorResult, LocalVerdict, build_verdict_document, merge_agreement


class LlmEvaluatorTests(unittest.TestCase):
    def test_dual_agreement_keeps_confident_verdict(self) -> None:
        agreement = merge_agreement(
            LocalVerdict(verdict="needs_targeted_support", score=0.62),
            [
                EvaluatorResult(provider="gemini", configured=True, ok=True, verdict="on_track", score=0.8),
                EvaluatorResult(provider="minimax", configured=True, ok=True, verdict="on_track", score=0.74),
            ],
        )
        self.assertEqual(agreement.mode, "dual_external")
        self.assertTrue(agreement.agreement)
        self.assertEqual(agreement.confidence, "high")
        self.assertEqual(agreement.final_verdict, "on_track")
        self.assertEqual(agreement.final_score, 0.72)

    def test_disagreement_returns_uncertain(self) -> None:
        agreement = merge_agreement(
            LocalVerdict(verdict="needs_targeted_support", score=0.62),
            [
                EvaluatorResult(provider="gemini", configured=True, ok=True, verdict="on_track", score=0.85),
                EvaluatorResult(
                    provider="minimax",
                    configured=True,
                    ok=True,
                    verdict="needs_review",
                    score=0.4,
                ),
            ],
        )
        self.assertFalse(agreement.agreement)
        self.assertEqual(agreement.confidence, "low")
        self.assertEqual(agreement.final_verdict, "uncertain")
        self.assertGreater(agreement.score_delta or 0, 0.3)

    def test_missing_external_evaluators_falls_back_local(self) -> None:
        agreement = merge_agreement(
            LocalVerdict(verdict="needs_targeted_support", score=0.62),
            [
                EvaluatorResult(provider="gemini", configured=False, ok=False, error="missing"),
                EvaluatorResult(provider="minimax", configured=False, ok=False, error="missing"),
            ],
        )
        self.assertEqual(agreement.mode, "local_fallback")
        self.assertEqual(agreement.final_verdict, "needs_targeted_support")
        self.assertEqual(agreement.final_score, 0.62)

    def test_claim_disagreement_flags_for_review_and_creates_seed(self) -> None:
        agreement = merge_agreement(
            LocalVerdict(verdict="needs_targeted_support", score=0.62),
            [
                EvaluatorResult(
                    provider="gemini",
                    configured=True,
                    ok=True,
                    verdict="needs_targeted_support",
                    score=0.66,
                    flagged_claims=[{"claim": "Sunlight turns directly into glucose.", "reason": "incorrect"}],
                ),
                EvaluatorResult(
                    provider="minimax",
                    configured=True,
                    ok=True,
                    verdict="needs_targeted_support",
                    score=0.66,
                    flagged_claims=[{"claim": "Chlorophyll is the main mechanism.", "reason": "imprecise"}],
                ),
            ],
        )
        self.assertFalse(agreement.agreement)
        self.assertTrue(agreement.claim_disagreement)
        self.assertEqual(agreement.final_verdict, "flagged_for_review")
        self.assertIsNotNone(agreement.self_improvement_seed)

    def test_prompt_version_filters_false_claim_disagreement(self) -> None:
        results = [
            EvaluatorResult(
                provider="gemini",
                configured=True,
                ok=True,
                verdict="needs_targeted_support",
                score=0.66,
                flagged_claims=[{"claim": "Nice revision.", "reason": "not actually a content claim"}],
            ),
            EvaluatorResult(
                provider="minimax",
                configured=True,
                ok=True,
                verdict="needs_targeted_support",
                score=0.66,
                flagged_claims=[],
            ),
        ]
        v1 = merge_agreement(LocalVerdict(verdict="needs_targeted_support", score=0.5), results)
        v2 = merge_agreement(
            LocalVerdict(verdict="needs_targeted_support", score=0.5),
            results,
            {"self_improvement_context": {"prompt_version": 2}},
        )
        self.assertTrue(v1.claim_disagreement)
        self.assertEqual(v1.final_verdict, "flagged_for_review")
        self.assertFalse(v2.claim_disagreement)
        self.assertTrue(v2.agreement)
        self.assertEqual(v2.final_verdict, "needs_targeted_support")

    def test_behavioral_notes_dampen_confidence(self) -> None:
        agreement = merge_agreement(
            LocalVerdict(verdict="needs_targeted_support", score=0.62),
            [
                EvaluatorResult(provider="gemini", configured=True, ok=True, verdict="on_track", score=0.8),
                EvaluatorResult(provider="minimax", configured=True, ok=True, verdict="on_track", score=0.78),
            ],
            {
                "evidence": {
                    "rewatch_rate": 2.0,
                    "hesitation_ms": 4200,
                    "drawing_score": 0.2,
                }
            },
        )
        self.assertEqual(agreement.final_verdict, "on_track")
        self.assertEqual(agreement.confidence, "low")
        self.assertEqual(agreement.behavioral_notes["rewatch"]["note"], "high_rewatch")

    def test_verdict_document_contract(self) -> None:
        agreement = merge_agreement(
            LocalVerdict(verdict="needs_targeted_support", score=0.62),
            [
                EvaluatorResult(provider="gemini", configured=True, ok=True, verdict="on_track", score=0.8),
                EvaluatorResult(provider="minimax", configured=True, ok=True, verdict="on_track", score=0.78),
            ],
        )
        document = build_verdict_document("session_1", agreement)
        self.assertEqual(document.session_id, "session_1")
        self.assertEqual(document.overall_score, agreement.final_score)
        self.assertIn(document.confidence, {"low", "medium", "high"})
        self.assertIn("rewatch", document.behavioral_notes)


if __name__ == "__main__":
    unittest.main()
