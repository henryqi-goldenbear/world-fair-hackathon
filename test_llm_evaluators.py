import unittest

from llm_evaluators import EvaluatorResult, LocalVerdict, merge_agreement


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


if __name__ == "__main__":
    unittest.main()
