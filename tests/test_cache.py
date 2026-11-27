"""Unit tests for the manifest cache (manifest.py).

No ML models: cache-hit, staleness and manifest persistence are
pure dict/file operations, tested here with a temp directory.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manifest import (  # noqa: E402
    is_cache_hit,
    is_stale,
    load_manifest,
    manifest_entry,
    save_manifest,
    sha256_bytes,
    stale_chunk_keys,
)


class TestSha256(unittest.TestCase):
    def test_stable_and_hex(self):
        h = sha256_bytes(b"hello")
        self.assertEqual(len(h), 64)
        self.assertEqual(h, sha256_bytes(b"hello"))
        self.assertNotEqual(h, sha256_bytes(b"HELLO"))


class TestManifestRoundTrip(unittest.TestCase):
    def test_save_load_preserves_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            m = {"a.txt": manifest_entry("abc", 3)}
            save_manifest(path, m)
            loaded = load_manifest(path)
            self.assertEqual(loaded["a.txt"]["hash"], "abc")
            self.assertEqual(loaded["a.txt"]["chunks"], 3)
            self.assertIn("indexed_at", loaded["a.txt"])

    def test_missing_file_is_empty(self):
        self.assertEqual(load_manifest("/nonexistent/manifest.json"), {})

    def test_corrupt_file_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text("{not json")
            self.assertEqual(load_manifest(path), {})


class TestCacheHit(unittest.TestCase):
    def setUp(self):
        self.manifest = {"a.txt": manifest_entry("hash-a", 2)}
        self.registry = {("a.txt", 0), ("a.txt", 1), ("b.txt", 0)}

    def test_hit_when_hash_and_chunks_present(self):
        self.assertTrue(
            is_cache_hit(self.manifest, "a.txt", "hash-a", self.registry))

    def test_miss_when_hash_changed(self):
        self.assertFalse(
            is_cache_hit(self.manifest, "a.txt", "hash-b", self.registry))

    def test_miss_when_chunks_missing_from_registry(self):
        # Global ntotal > 0 is NOT enough: a.txt's chunk 1 is gone.
        registry = {("a.txt", 0), ("b.txt", 0)}
        self.assertFalse(
            is_cache_hit(self.manifest, "a.txt", "hash-a", registry))

    def test_miss_for_unknown_file(self):
        self.assertFalse(
            is_cache_hit(self.manifest, "c.txt", "hash-c", self.registry))


class TestStale(unittest.TestCase):
    def setUp(self):
        self.manifest = {"a.txt": manifest_entry("hash-a", 2)}

    def test_same_hash_is_not_stale(self):
        self.assertFalse(is_stale(self.manifest, "a.txt", "hash-a"))

    def test_changed_hash_is_stale(self):
        self.assertTrue(is_stale(self.manifest, "a.txt", "hash-b"))

    def test_unknown_file_is_not_stale(self):
        self.assertFalse(is_stale(self.manifest, "c.txt", "hash-c"))

    def test_stale_chunk_keys_cover_all_old_chunks(self):
        self.assertEqual(
            stale_chunk_keys(self.manifest, "a.txt"),
            {("a.txt", 0), ("a.txt", 1)})

    def test_stale_chunk_keys_empty_for_unknown_file(self):
        self.assertEqual(stale_chunk_keys(self.manifest, "c.txt"), set())


if __name__ == "__main__":
    unittest.main()
