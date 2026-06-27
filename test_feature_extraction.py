import unittest

from feature_extraction import extract_claims, extract_feature_bundle


class FeatureExtractionTests(unittest.TestCase):
    def sample_session(self):
        return {
            "session_id": "test_session",
            "transcript": [
                {"t": 0, "speaker": "Tutor", "text": "Photosynthesis uses carbon dioxide and water."},
                {"t": 5200, "speaker": "Student", "text": "I think oxygen is the main input."},
                {"t": 9000, "speaker": "Tutor", "text": "Chlorophyll captures light energy."},
                {"t": 15000, "speaker": "Student", "text": "Glucose stores chemical energy because light is captured."},
            ],
            "events": [
                {"type": "pause", "t": 5000, "payload": {"duration_ms": 3200}},
                {"type": "rewatch", "t": 12000, "payload": {"from_ms": 0, "to_ms": 9000}},
                {"type": "add", "t": 13000, "payload": {"element": {"type": "text"}}},
            ],
            "drawings": [
                '<svg width="960" height="640"><text x="72" y="80">CO2</text><line x1="80" y1="90" x2="300" y2="220"/></svg>'
            ],
            "metadata": {"duration_ms": 30000, "subject": "photosynthesis", "student_id": "agent_1"},
        }

    def test_claim_extraction_marks_factual_claims(self):
        claims = extract_claims(self.sample_session()["transcript"])
        self.assertGreaterEqual(len(claims), 4)
        verifiable = [claim.claim for claim in claims if claim.verifiable]
        self.assertTrue(any("Photosynthesis uses carbon dioxide" in claim for claim in verifiable))
        self.assertFalse(next(claim for claim in claims if "oxygen is the main input" in claim.claim).verifiable)

    def test_behavioral_signals(self):
        bundle = extract_feature_bundle(self.sample_session())
        self.assertGreater(bundle.behavior.rewatch_rate, 0)
        self.assertGreaterEqual(bundle.behavior.hesitation_ms, 3000)
        self.assertGreater(bundle.behavior.drawing_score, 0)

    def test_bundle_contract(self):
        bundle = extract_feature_bundle(self.sample_session())
        payload = bundle.model_dump() if hasattr(bundle, "model_dump") else bundle.dict()
        self.assertIn("claims", payload)
        self.assertIn("behavior", payload)
        self.assertIn("verifiable_claim_count", payload["metadata"])


if __name__ == "__main__":
    unittest.main()
