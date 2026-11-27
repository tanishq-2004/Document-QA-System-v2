"""Fast unit tests for the Document QA text helpers.

These tests import only text_utils (pure text handling + the
langchain text splitter). They deliberately do NOT import app.py,
because importing app.py loads the embedding model and the
cross-encoder reranker at module level.

Run:  python -m unittest discover -s tests -v
"""

import hashlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_utils import clean_text, get_file_hash, split_text


class TestCleanText(unittest.TestCase):
    def test_collapses_newlines_and_whitespace(self):
        self.assertEqual(clean_text("hello\n\n  world \n test"), "hello world test")

    def test_empty_stays_empty(self):
        self.assertEqual(clean_text(""), "")

    def test_already_clean_unchanged(self):
        self.assertEqual(clean_text("one two three"), "one two three")


class TestSplitText(unittest.TestCase):
    def test_short_text_is_one_chunk_with_metadata(self):
        chunks = split_text("A short document about solar panels.", "doc.txt")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["source"], "doc.txt")
        self.assertEqual(chunks[0].metadata["chunk_id"], 0)

    def test_long_text_splits_within_chunk_size(self):
        text = " ".join(f"word{i}" for i in range(400))
        chunks = split_text(text, "long.txt")
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 500)

    def test_chunk_ids_are_sequential(self):
        text = " ".join(f"word{i}" for i in range(400))
        chunks = split_text(text, "long.txt")
        self.assertEqual(
            [c.metadata["chunk_id"] for c in chunks],
            list(range(len(chunks))),
        )
        self.assertTrue(all(c.metadata["source"] == "long.txt" for c in chunks))

    def test_overlap_shares_text_between_chunks(self):
        text = " ".join(f"word{i}" for i in range(400))
        chunks = split_text(text, "long.txt")
        # With 50 chars of overlap, the start of chunk 2 appears in chunk 1.
        self.assertIn(chunks[1].page_content[:20], chunks[0].page_content)


class TestGetFileHash(unittest.TestCase):
    def test_same_file_same_hash(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("hello world")
            path = f.name
        try:
            self.assertEqual(get_file_hash(path), get_file_hash(path))
            self.assertEqual(
                get_file_hash(path),
                hashlib.sha256(b"hello world").hexdigest(),
            )
            self.assertEqual(len(get_file_hash(path)), 64)
        finally:
            os.unlink(path)

    def test_different_content_different_hash(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f1:
            f1.write("aaa")
            p1 = f1.name
        with tempfile.NamedTemporaryFile("w", delete=False) as f2:
            f2.write("bbb")
            p2 = f2.name
        try:
            self.assertNotEqual(get_file_hash(p1), get_file_hash(p2))
        finally:
            os.unlink(p1)
            os.unlink(p2)


if __name__ == "__main__":
    unittest.main()
