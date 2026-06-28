import unittest

from atlas_integration import (
    EMBEDDING_DIMENSIONS,
    atlas_search_index_definitions,
    flagged_claim_embeddings,
    prepare_verdict_document,
    stable_embedding,
)


class AtlasIntegrationTests(unittest.TestCase):
    def test_stable_embedding_shape(self) -> None:
        first = stable_embedding("sunlight turns directly into oxygen")
        second = stable_embedding("sunlight turns directly into oxygen")
        self.assertEqual(first, second)
        self.assertEqual(len(first), EMBEDDING_DIMENSIONS)

    def test_verdict_document_adds_flagged_claim_embeddings(self) -> None:
        report = {
            "session_id": "session_1",
            "flagged_claims": [
                {
                    "claim": "Sunlight turns directly into oxygen.",
                    "reason": "incorrect",
                    "providers": ["gemini", "minimax"],
                }
            ],
        }
        prepared = prepare_verdict_document(report)
        self.assertEqual(prepared["collection_role"], "final_evaluated_output")
        self.assertEqual(len(prepared["flagged_claim_embeddings"]), 1)
        self.assertEqual(len(prepared["flagged_claim_embeddings"][0]["embedding"]), EMBEDDING_DIMENSIONS)

    def test_empty_flagged_claims_have_no_embeddings(self) -> None:
        self.assertEqual(flagged_claim_embeddings({"flagged_claims": []}), [])

    def test_search_index_definitions_include_text_and_vector(self) -> None:
        definitions = atlas_search_index_definitions()
        self.assertIn("text_search", definitions)
        self.assertIn("vector_search", definitions)
        self.assertEqual(
            definitions["vector_search"]["definition"]["fields"][0]["numDimensions"],
            EMBEDDING_DIMENSIONS,
        )


if __name__ == "__main__":
    unittest.main()
