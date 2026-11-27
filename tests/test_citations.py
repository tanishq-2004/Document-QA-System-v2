"""Unit tests for citations (pure stdlib, no ML models).

Run:  python -m pytest tests/test_citations.py -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from citations import (
    assign_citations,
    content_tokens,
    overlap_coefficient,
    split_sentences,
)


CHUNK_891 = {
    "source_file": "FR_part1.pdf",
    "chunk_id": 891,
    "content": (
        "Fryatt's trial balance at 31 May 20X7 shows revenue of "
        "$630,000 and cost of sales of $420,000."
    ),
}

CHUNK_907 = {
    "source_file": "FR_part1.pdf",
    "chunk_id": 907,
    "content": (
        "The share capital at acquisition was $500,000 and retained "
        "earnings were $120,000."
    ),
}


class TestSplitSentences(unittest.TestCase):
    def test_splits_on_period(self):
        self.assertEqual(
            split_sentences("First sentence. Second sentence."),
            ["First sentence.", "Second sentence."],
        )

    def test_single_sentence_unchanged(self):
        self.assertEqual(
            split_sentences("Only one."),
            ["Only one."],
        )


class TestOverlap(unittest.TestCase):
    def test_identical_tokens_score_one(self):
        t = content_tokens("Fryatt revenue 630000")
        self.assertEqual(overlap_coefficient(t, t), 1.0)

    def test_empty_side_scores_zero(self):
        self.assertEqual(overlap_coefficient(set(), {"a"}), 0.0)

    def test_stopwords_do_not_count(self):
        self.assertEqual(content_tokens("the of and"), set())


class TestAssignCitations(unittest.TestCase):
    def test_paraphrased_sentence_gets_right_chunk(self):
        answer = "Fryatt's revenue was $630,000."
        cited, citations = assign_citations(answer, [CHUNK_891, CHUNK_907])
        self.assertIn("[FR_part1.pdf #891]", cited)
        self.assertEqual(
            citations, [{"source_file": "FR_part1.pdf", "chunk_id": 891}]
        )

    def test_each_sentence_cited_to_its_own_chunk(self):
        answer = (
            "Fryatt's revenue was $630,000. "
            "The share capital at acquisition was $500,000."
        )
        cited, citations = assign_citations(answer, [CHUNK_891, CHUNK_907])
        self.assertIn("[FR_part1.pdf #891]", cited)
        self.assertIn("[FR_part1.pdf #907]", cited)
        self.assertEqual(len(citations), 2)

    def test_unrelated_sentence_gets_no_marker(self):
        answer = "Penguins migrate across the southern ocean."
        cited, citations = assign_citations(answer, [CHUNK_891, CHUNK_907])
        self.assertEqual(cited, answer)
        self.assertEqual(citations, [])

    def test_mixed_answer_only_matching_sentences_cited(self):
        answer = (
            "Fryatt's revenue was $630,000. "
            "Penguins migrate across the southern ocean."
        )
        cited, citations = assign_citations(answer, [CHUNK_891])
        self.assertIn("[FR_part1.pdf #891]", cited)
        self.assertIn("Penguins migrate", cited)
        self.assertNotIn("Penguins migrate across the southern ocean. [", cited)
        self.assertEqual(len(citations), 1)

    def test_no_sources_returns_answer_unchanged(self):
        answer = "Fryatt's revenue was $630,000."
        cited, citations = assign_citations(answer, [])
        self.assertEqual(cited, answer)
        self.assertEqual(citations, [])

    def test_fewer_than_three_shared_words_gets_no_marker(self):
        # Two shared words at high overlap is not enough: short
        # generic sentences must not earn a citation from one or two
        # shared words.
        answer = "Based on the documents, revenue rose."
        chunk = {
            "source_file": "doc.txt",
            "chunk_id": 0,
            "content": "The documents describe revenue and sales.",
        }
        cited, citations = assign_citations(answer, [chunk])
        self.assertEqual(cited, answer)
        self.assertEqual(citations, [])

    # NOTE: the retrieval-only boilerplate sentence ("Retrieval-only
    # mode (no Groq API key provided).") shares "groq api key" with
    # the rogue memo chunk and was once wrongly cited to it. The
    # matcher alone cannot tell boilerplate from a real sentence, so
    # the fix is in app.py: citations only run for real LLM
    # generation (groq_api_key present), never in retrieval-only mode.


if __name__ == "__main__":
    unittest.main()
