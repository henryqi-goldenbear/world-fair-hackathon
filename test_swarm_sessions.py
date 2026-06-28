import unittest

from swarm_sessions import clone_swarm_payload, load_swarm_payloads


class SwarmSessionTests(unittest.TestCase):
    def test_swarm_payloads_are_agent_students_with_varied_abilities(self) -> None:
        payloads = load_swarm_payloads()
        self.assertGreaterEqual(len(payloads), 10)
        abilities = {
            round(float(payload["metadata"]["ability"]), 2)
            for payload in payloads
            if payload["metadata"].get("ability") is not None
        }
        bands = {payload["metadata"].get("understanding_band") for payload in payloads}
        self.assertGreaterEqual(len(abilities), 8)
        self.assertIn("novice", bands)
        self.assertIn("advanced", bands)
        self.assertTrue(all(payload["metadata"]["student_kind"] == "agent_student" for payload in payloads))
        self.assertTrue(all(payload["metadata"]["human_student"] is False for payload in payloads))

    def test_clone_swarm_payload_round_robins_without_mutating_cache(self) -> None:
        first = clone_swarm_payload(0)
        second = clone_swarm_payload(1)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        first["session_id"] = "mutated"
        self.assertNotEqual(clone_swarm_payload(0)["session_id"], "mutated")
        self.assertNotEqual(first["metadata"]["ability"], second["metadata"]["ability"])


if __name__ == "__main__":
    unittest.main()
