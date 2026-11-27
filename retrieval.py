"""Hybrid retrieval for the Document QA System: BM25 + FAISS + RRF.

Pure Python, no new dependency (no rank_bm25 — keeping the HF Docker
build light). This module imports nothing from langchain, torch,
sentence-transformers or FastAPI, so it can be unit-tested without
loading any ML model. app.py imports from here.

Design
-------
* BM25 (Okapi, k1=1.5, b=0.75) over chunk text. The index is rebuilt
  from the in-memory chunk registry whenever the registry changes
  (upload / reset / startup) — at our scale (400 chunks) that is
  milliseconds, and it keeps BM25 and FAISS in sync by construction.
* Retrieval: dense FAISS top-10 + BM25 top-10, fused with Reciprocal
  Rank Fusion (RRF, k=60), take the top-10 fused, then the
  cross-encoder reranks to the top 3. RRF needs only rank order, so
  it fuses scores that live on incomparable scales (L2 distance vs
  BM25) without any normalisation hack.
* Rerank fallback + diversity live here too, so they are testable
  with stub rerankers: if ``reranker.predict`` throws, the caller
  gets the fused/dense order back with ``fallback=True`` instead of
  a 500; after ranking, ``diversify`` drops near-duplicate chunks
  (high token-overlap with an already-selected chunk) from the top 3.

Chunk identity: a chunk is identified by (source_file, chunk_id).
Functions accept either langchain Documents (duck-typed via
``page_content`` / ``metadata``) or plain dicts with ``text`` /
``source`` / ``chunk_id`` keys, so the unit tests need no langchain.
"""

import logging
import math
import re

logger = logging.getLogger(__name__)

# Words that carry no retrieval signal. Numbers and proper nouns are
# deliberately NOT stopwords — BM25's job here is exactly the
# keyword-heavy queries (exact amounts, model names, place names)
# where dense retrieval is weakest.
_STOPWORDS = frozenset(
    "a an the is are was were be been being do does did have has had "
    "of in on at to for with by from as and or not no if it its this "
    "that these those what when where which who whom how many much "
    "can could should would will shall may might".split()
)

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
# Overlap coefficient above which a lower-ranked chunk is treated as
# a near-duplicate of an already-selected one and skipped.
DEDUP_MAX_OVERLAP = 0.6


def tokenize(text):
    """Lowercase alphanumeric tokens, stopwords removed."""
    if not text:
        return []
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in _STOPWORDS]


def chunk_key(doc):
    """(source_file, chunk_id) for a Document or a plain dict."""
    if isinstance(doc, dict):
        return (doc.get("source", "unknown"), doc.get("chunk_id", "N/A"))
    meta = getattr(doc, "metadata", {}) or {}
    return (meta.get("source", "unknown"), meta.get("chunk_id", "N/A"))


def chunk_text(doc):
    """Text of a Document or a plain dict."""
    if isinstance(doc, dict):
        return doc.get("text", "") or doc.get("page_content", "") or ""
    return getattr(doc, "page_content", "") or ""


class BM25Index:
    """Okapi BM25 over a list of chunks. Empty index scores nothing."""

    def __init__(self, docs):
        self.docs = list(docs)
        self.doc_tokens = [tokenize(chunk_text(d)) for d in self.docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.n = len(self.docs)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        df = {}
        for tokens in self.doc_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        # Robertson idf, floored at 0 so a term in every doc is neutral,
        # never negative.
        self.idf = {
            t: max(0.0, math.log(1 + (self.n - d + 0.5) / (d + 0.5)))
            for t, d in df.items()
        }

    def scores(self, query):
        """BM25 score per doc index, in doc order."""
        if self.n == 0:
            return []
        q_tokens = tokenize(query)
        out = [0.0] * self.n
        for i, tokens in enumerate(self.doc_tokens):
            if not tokens or not self.avgdl:
                continue
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            dl = self.doc_len[i]
            s = 0.0
            for t in set(q_tokens):
                if t not in tf or t not in self.idf:
                    continue
                f = tf[t]
                denom = f + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avgdl)
                s += self.idf[t] * f * (BM25_K1 + 1) / denom
            out[i] = s
        return out

    def search(self, query, k=10):
        """Top-k (doc_index, score), highest first, zero scores dropped."""
        ranked = sorted(
            enumerate(self.scores(query)), key=lambda x: x[1], reverse=True
        )
        return [(i, s) for i, s in ranked[:k] if s > 0]


def rrf_fuse(ranked_lists, k=RRF_K):
    """Reciprocal Rank Fusion over rank-ordered key lists.

    ``ranked_lists`` is a list of lists of chunk keys, best first.
    Returns [(key, rrf_score)] sorted by fused score, highest first.
    A key ranked #1 in one list and absent in the other still scores
    (1/(k+1)), so fusion never drops a candidate entirely.
    """
    fused = {}
    for lst in ranked_lists:
        for rank, key in enumerate(lst):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def token_overlap(a, b):
    """Overlap coefficient of two token sets: |A∩B| / min(|A|, |B|).

    1.0 = one chunk's tokens are a subset of the other's (near
    duplicate, e.g. an exact re-upload under a different chunk id);
    ~0.1 = two adjacent 500-char chunks sharing only the 50-char
    overlap (the Q6 case: chunks 1 and 2 share one short phrase).
    """
    set_a, set_b = set(tokenize(a)), set(tokenize(b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def diversify(ranked_docs, top_n=3, max_overlap=DEDUP_MAX_OVERLAP):
    """Greedy diversity selection over (score, doc) pairs, best first.

    Takes the top chunk, then each next chunk only if its token
    overlap with every already-selected chunk is <= max_overlap.
    If dedup leaves fewer than top_n, the remaining slots are filled
    in rank order (a near-duplicate is better than no source at all).
    Returns a list of (score, doc) in selection order.
    """
    selected, skipped = [], []
    for score, doc in ranked_docs:
        text = chunk_text(doc)
        if any(token_overlap(text, chunk_text(d)) > max_overlap
               for _, d in selected):
            skipped.append((score, doc))
            continue
        selected.append((score, doc))
        if len(selected) >= top_n:
            break
    if len(selected) < top_n:
        selected.extend(skipped[: top_n - len(selected)])
    return selected[:top_n]


def rerank_with_fallback(question, candidates, reranker, top_n=3):
    """Cross-encoder rerank with a safe fallback.

    ``candidates`` is a list of (score, doc) in retrieval order (the
    score is the fused/dense score, used only as the fallback score).
    Returns (ranked_docs, fallback_bool): ranked_docs is the full
    ranked list, best first. On any reranker error the retrieval
    order is returned unchanged and fallback=True — the caller must
    never 500 on a reranker failure.
    """
    if not candidates:
        return [], False
    try:
        pairs = [(question, chunk_text(doc)) for _, doc in candidates]
        scores = reranker.predict(pairs)
    except Exception as e:
        logger.warning("reranker failed, falling back to retrieval order: %s", e)
        return list(candidates), True
    ranked = sorted(
        zip(scores, [doc for _, doc in candidates]),
        key=lambda x: x[0], reverse=True,
    )
    return ranked, False
