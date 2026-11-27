"""Unit tests for answer_cache (pure stdlib, no ML models).

Run:  python -m pytest tests/test_answer_cache.py -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from answer_cache import AnswerCache, cache_key, normalize_question


class TestNormalizeQuestion(unittest.TestCase):
    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(
            normalize_question("  What IS   the capital? "),
            normalize_question("what is the capital?"),
        )

    def test_empty_stays_empty(self):
        self.assertEqual(normalize_question("   "), "")


class TestCacheKey(unittest.TestCase):
    def test_same_question_same_key(self):
        self.assertEqual(
            cache_key("What is Paris?", True, "hybrid"),
            cache_key("  what is paris? ", True, "hybrid"),
        )

    def test_rerank_flag_changes_key(self):
        self.assertNotEqual(
            cache_key("What is Paris?", True, "hybrid"),
            cache_key("What is Paris?", False, "hybrid"),
        )

    def test_retrieval_method_changes_key(self):
        self.assertNotEqual(
            cache_key("What is Paris?", True, "hybrid"),
            cache_key("What is Paris?", True, "dense"),
        )


class TestAnswerCache(unittest.TestCase):
    def test_put_get_roundtrip(self):
        c = AnswerCache(max_entries=10)
        key = cache_key("What is Paris?", True, "hybrid")
        c.put(key, {"answer": "Paris"})
        self.assertEqual(c.get(key), {"answer": "Paris"})

    def test_miss_returns_none(self):
        c = AnswerCache(max_entries=10)
        self.assertIsNone(c.get(("nope", True, "hybrid")))

    def test_overwrite_same_key(self):
        c = AnswerCache(max_entries=10)
        key = cache_key("Q?", True, "hybrid")
        c.put(key, {"answer": "one"})
        c.put(key, {"answer": "two"})
        self.assertEqual(c.get(key), {"answer": "two"})
        self.assertEqual(len(c), 1)

    def test_fifo_eviction_at_cap(self):
        c = AnswerCache(max_entries=2)
        c.put(("q1", True, "hybrid"), {"answer": "1"})
        c.put(("q2", True, "hybrid"), {"answer": "2"})
        c.put(("q3", True, "hybrid"), {"answer": "3"})
        self.assertIsNone(c.get(("q1", True, "hybrid")))
        self.assertEqual(c.get(("q2", True, "hybrid")), {"answer": "2"})
        self.assertEqual(c.get(("q3", True, "hybrid")), {"answer": "3"})

    def test_clear_empties(self):
        c = AnswerCache(max_entries=10)
        c.put(("q1", True, "hybrid"), {"answer": "1"})
        c.clear()
        self.assertEqual(len(c), 0)
        self.assertIsNone(c.get(("q1", True, "hybrid")))

    def test_zero_max_disables(self):
        c = AnswerCache(max_entries=0)
        c.put(("q1", True, "hybrid"), {"answer": "1"})
        self.assertEqual(len(c), 0)
        self.assertIsNone(c.get(("q1", True, "hybrid")))


if __name__ == "__main__":
    unittest.main()
