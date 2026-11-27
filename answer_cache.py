"""In-memory cache for repeated questions (pure stdlib, no ML models).

A repeated question against an unchanged index gets the stored answer
with no retrieval, rerank or LLM call, so no Groq tokens are spent.
Only real generated answers are cached: refusals cost no LLM tokens
anyway, and a cached refusal could go stale.

Correctness rules, enforced by the caller (app.py):
- The key covers the normalised question text, the rerank flag and
  the retrieval method, so different arms never share an answer.
- The cache is cleared whenever the index changes (a new or edited
  document is indexed, or the index is reset), so a hit always
  matches the current documents.
- Size is capped (FIFO eviction); set ANSWER_CACHE_MAX_ENTRIES=0 to
  disable the cache entirely.
"""

import os
import re
from collections import OrderedDict

ANSWER_CACHE_MAX_ENTRIES = int(
    os.environ.get("ANSWER_CACHE_MAX_ENTRIES", "200")
)

_WS_RE = re.compile(r"\s+")


def normalize_question(question):
    """Lowercase, strip, collapse whitespace — 'What IS  the capital?' matches 'what is the capital?'."""
    return _WS_RE.sub(" ", str(question).strip().lower())


def cache_key(question, rerank, retrieval_method):
    """Cache key: normalised question + rerank flag + retrieval method."""
    return (
        normalize_question(question),
        bool(rerank),
        str(retrieval_method).lower(),
    )


class AnswerCache:
    """Tiny FIFO cache. ``max_entries <= 0`` disables it (puts are no-ops)."""

    def __init__(self, max_entries=ANSWER_CACHE_MAX_ENTRIES):
        self.max_entries = int(max_entries)
        self._store = OrderedDict()

    def get(self, key):
        return self._store.get(key)

    def put(self, key, value):
        if self.max_entries <= 0:
            return
        if key in self._store:
            self._store[key] = value
            return
        while len(self._store) >= self.max_entries:
            self._store.popitem(last=False)
        self._store[key] = value

    def clear(self):
        self._store.clear()

    def __len__(self):
        return len(self._store)


# Process-wide cache used by app.py.
cache = AnswerCache()
