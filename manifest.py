"""Manifest-based cache for the Document QA System.

The old cache was a per-hash ``{"indexed": True}`` pickle in
vector_store/ plus a global ``ntotal > 0`` check. That could not tell
*which* file a hash belonged to, so uploading the same filename with
different content added the new chunks while the old chunks stayed
in the index (stale chunks), and a cache hit only proved *some*
document was indexed, not *this* one.

The manifest (``vector_store/manifest.json``) is the source of truth
for what is indexed:

    {filename: {"hash": sha256, "chunks": n, "indexed_at": iso8601}}

This module is pure (stdlib only, operates on plain dicts) so it can
be unit-tested without loading any ML model. app.py owns the actual
FAISS index and the in-memory chunk registry; the helpers here decide
cache hits, staleness, and manifest updates from the manifest dict
plus the set of chunk keys currently in the registry.
"""

import hashlib
import json
from datetime import datetime, timezone


def sha256_bytes(data):
    """SHA-256 hex digest of bytes (hash the upload bytes, before
    writing the file to disk, so the cache check needs no disk I/O)."""
    return hashlib.sha256(data).hexdigest()


def manifest_entry(file_hash, chunk_count, indexed_at=None):
    return {
        "hash": file_hash,
        "chunks": int(chunk_count),
        "indexed_at": indexed_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def load_manifest(path):
    """Read a manifest file; missing / corrupt / wrong-shaped files
    mean 'nothing indexed' rather than a crash."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for filename, entry in data.items():
        if (isinstance(entry, dict) and isinstance(entry.get("hash"), str)
                and isinstance(entry.get("chunks"), int)):
            out[filename] = entry
    return out


def save_manifest(path, manifest):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def is_cache_hit(manifest, filename, file_hash, registry_keys):
    """True only if THIS file's hash is in the manifest AND every
    chunk it claims is actually in the chunk registry.

    ``registry_keys`` is the set of (source, chunk_id) keys currently
    indexed. A global ntotal > 0 is not enough: after a reset, a
    rebuild, or a stale-file replacement, the index may be non-empty
    while this file's chunks are gone.
    """
    entry = manifest.get(filename)
    if not entry or entry.get("hash") != file_hash:
        return False
    expected = {(filename, i) for i in range(entry.get("chunks", 0))}
    return bool(expected) and expected.issubset(registry_keys)


def is_stale(manifest, filename, file_hash):
    """True if the manifest knows this filename under a different
    hash — i.e. the file was edited and its old chunks must be
    removed before the new ones are added."""
    entry = manifest.get(filename)
    return bool(entry) and entry.get("hash") != file_hash


def stale_chunk_keys(manifest, filename):
    """Chunk keys to remove when ``filename`` is being replaced."""
    entry = manifest.get(filename)
    if not entry:
        return set()
    return {(filename, i) for i in range(entry.get("chunks", 0))}
