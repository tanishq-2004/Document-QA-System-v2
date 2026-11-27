"""Unit tests for hybrid retrieval (retrieval.py).

No ML models: BM25, RRF fusion, diversity and the rerank fallback
are all pure functions, tested here with plain dict chunks and a
stub reranker.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import (  # noqa: E402
    BM25Index,
    diversify,
    rerank_with_fallback,
    rrf_fuse,
    token_overlap,
    tokenize,
)


def chunk(text, source, chunk_id):
    return {"text": text, "source": source, "chunk_id": chunk_id}


DOCS = [
    chunk("Kestrel K-7 uses a 5200 mAh lithium battery for flight",
          "kestrel.txt", 0),
    chunk("Aurora Solar grants 24 days of paid annual leave per year",
          "aurora.txt", 0),
    chunk("Harbourside Cafe earned 18.4 lakh rupees in Q2 2026",
          "harbourside.txt", 0),
    chunk("Kestrel K-7 uses a 5200 mAh lithium battery for flight",
          "kestrel.txt", 1),  # exact duplicate of chunk 0, new id
]


class TestTokenize(unittest.TestCase):
    def test_lowercases_and_keeps_numbers(self):
        self.assertIn("5200", tokenize("5200 mAh battery"))
        self.assertIn("kestrel", tokenize("Kestrel K-7"))

    def test_stopwords_removed(self):
        self.assertNotIn("the", tokenize("the battery"))
        self.assertEqual(tokenize(""), [])


class TestBM25(unittest.TestCase):
    def test_rare_term_ranks_its_doc_first(self):
        idx = BM25Index(DOCS[:3])
        hits = idx.search("5200 mAh battery", k=3)
        self.assertEqual(hits[0][0], 0)
        self.assertGreater(hits[0][1], 0)

    def test_unknown_term_scores_nothing(self):
        idx = BM25Index(DOCS[:3])
        self.assertEqual(idx.search("quasar telescope", k=3), [])

    def test_empty_index(self):
        idx = BM25Index([])
        self.assertEqual(idx.search("anything", k=3), [])
        self.assertEqual(idx.scores("anything"), [])


class TestRRF(unittest.TestCase):
    def test_top_in_both_lists_wins(self):
        fused = rrf_fuse([[("a", 0), ("b", 0)], [("a", 0), ("c", 0)]])
        self.assertEqual(fused[0][0], ("a", 0))

    def test_rank_one_in_one_list_still_scores(self):
        fused = rrf_fuse([[("a", 0)], [("b", 0)]])
        keys = [k for k, _ in fused]
        self.assertIn(("a", 0), keys)
        self.assertIn(("b", 0), keys)

    def test_rrf_score_decays_with_rank(self):
        fused = dict(rrf_fuse([[("a", 0), ("b", 0), ("c", 0)]]))
        self.assertGreater(fused[("a", 0)], fused[("b", 0)])
        self.assertGreater(fused[("b", 0)], fused[("c", 0)])


class TestDiversify(unittest.TestCase):
    def test_exact_duplicate_skipped(self):
        ranked = [(3.0, DOCS[0]), (2.0, DOCS[3]), (1.0, DOCS[1])]
        out = diversify(ranked, top_n=2)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][1]["chunk_id"], 0)
        # The duplicate (chunk 1, same text) must not be selected.
        self.assertNotEqual(out[1][1]["chunk_id"], 1)

    def test_q6_adjacent_chunks_both_kept(self):
        # Adjacent 500-char chunks share only the 50-char overlap, so
        # their token overlap is low and both survive diversity. Use
        # realistic vocab: ~70 distinct words each, ~8 shared.
        a = " ".join(f"alpha{i}" for i in range(70))
        a += " shared phrase altitude limit"
        b = "shared phrase altitude limit " + " ".join(
            f"beta{i}" for i in range(70))
        self.assertLess(token_overlap(a, b), 0.6)
        ranked = [(3.0, chunk(a, "d.txt", 1)), (2.0, chunk(b, "d.txt", 2))]
        self.assertEqual(len(diversify(ranked, top_n=2)), 2)

    def test_fills_from_skipped_if_needed(self):
        ranked = [(3.0, DOCS[0]), (2.0, DOCS[3])]
        out = diversify(ranked, top_n=2)
        self.assertEqual(len(out), 2)


class StubReranker:
    def __init__(self, scores=None, fail=False):
        self.scores = scores
        self.fail = fail

    def predict(self, pairs):
        if self.fail:
            raise RuntimeError("reranker down")
        if self.scores is not None:
            return self.scores
        return [0.0] * len(pairs)


class TestRerankFallback(unittest.TestCase):
    def test_reranker_reorders(self):
        cands = [(1.0, DOCS[0]), (2.0, DOCS[1])]
        ranked, fallback = rerank_with_fallback(
            "leave", cands, StubReranker(scores=[0.1, 5.0]))
        self.assertFalse(fallback)
        self.assertEqual(ranked[0][1]["source"], "aurora.txt")

    def test_reranker_failure_falls_back_to_retrieval_order(self):
        cands = [(1.0, DOCS[0]), (2.0, DOCS[1])]
        ranked, fallback = rerank_with_fallback(
            "leave", cands, StubReranker(fail=True))
        self.assertTrue(fallback)
        self.assertEqual(ranked, cands)

    def test_empty_candidates(self):
        ranked, fallback = rerank_with_fallback(
            "leave", [], StubReranker())
        self.assertEqual(ranked, [])
        self.assertFalse(fallback)


if __name__ == "__main__":
    unittest.main()
