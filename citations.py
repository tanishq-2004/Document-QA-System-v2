"""Deterministic inline citations for generated answers (pure stdlib).

Every sentence of the answer is matched against the retrieved source
chunks by content-word overlap. A sentence that overlaps a chunk
strongly enough gets that chunk cited inline, e.g.:

    Fryatt's revenue was $630,000. [FR_part1.pdf #891]

A sentence with no chunk above the threshold gets no marker — an
uncited claim is honest, a guessed citation is not. Refusal answers
and injection-flagged answers are never cited (the caller skips them).

Why deterministic matching instead of asking the LLM to cite?
The LLM can cite the wrong chunk or invent a citation number; overlap
matching can only cite a chunk the sentence actually shares words
with, costs no tokens, and is unit-testable without a model.
"""

import re

# A sentence must share at least this much of its content words with
# a chunk (overlap coefficient = shared / shorter side) AND at least
# this many shared words to earn a citation. The word-count floor
# stops short generic sentences ("Based on the documents") earning a
# citation from one shared word — but it is capped at the sentence's
# own token count, so a bare-number answer ("630,000") can still be
# cited: it has 2 tokens ("630", "000") and must match both. That
# cap was missing at first, and the live demo answered "630,000"
# with no citation because 2 shared words < 3. 0.30 was checked
# against realistic paraphrased answers: "Fryatt's revenue was
# $630,000." vs its chunk scores ~0.5 with 4 shared words, while an
# unrelated sentence scores ~0.0.
MIN_OVERLAP = 0.30
MIN_SHARED_TOKENS = 3

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Words too common to prove a sentence came from a chunk. Kept small
# on purpose: dropping real content words would hide true matches.
_STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "are", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "and", "or", "but", "it", "its", "this", "that", "these", "those",
    "what", "which", "who", "when", "where", "how", "why", "was",
    "is", "did", "do", "does", "have", "has", "had", "not", "no",
    "per", "in", "about", "into", "over", "under", "between",
}


def split_sentences(text):
    """Split on sentence-ending punctuation, keeping it attached."""
    parts = _SENTENCE_RE.split(str(text).strip())
    return [p.strip() for p in parts if p.strip()]


def content_tokens(text):
    """Lowercase word tokens minus stopwords, as a set."""
    return {
        w for w in _WORD_RE.findall(str(text).lower())
        if w not in _STOPWORDS
    }


def overlap_coefficient(a_tokens, b_tokens):
    """Shared words / size of the smaller side (0.0 when either is empty)."""
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / min(len(a_tokens), len(b_tokens))


def assign_citations(
    answer, sources,
    min_overlap=MIN_OVERLAP,
    min_shared=MIN_SHARED_TOKENS,
):
    """Attach inline citations to an answer.

    ``sources`` is the response's source list: dicts with
    ``source_file``, ``chunk_id`` and ``content``. Returns
    ``(cited_answer, citations)`` where ``citations`` is the list of
    cited ``{"source_file", "chunk_id"}`` dicts in order of first use.
    """
    if not sources or not str(answer).strip():
        return answer, []
    chunk_tokens = [
        (s, content_tokens(s.get("content", "")))
        for s in sources
    ]
    cited_sentences = []
    citations = []
    seen = set()
    for sentence in split_sentences(answer):
        sent_tokens = content_tokens(sentence)
        best = None
        best_score = 0.0
        best_shared = 0
        for source, tokens in chunk_tokens:
            shared = len(sent_tokens & tokens)
            score = overlap_coefficient(sent_tokens, tokens)
            if score > best_score:
                best_score = score
                best = source
                best_shared = shared
        # A short sentence cannot share 3 words it does not have:
        # it must match all of its own content words instead.
        required_shared = min(min_shared, len(sent_tokens))
        if (
            best is not None
            and best_score >= min_overlap
            and best_shared >= required_shared
        ):
            key = (best.get("source_file"), best.get("chunk_id"))
            cited_sentences.append(
                f"{sentence} [{key[0]} #{key[1]}]"
            )
            if key not in seen:
                seen.add(key)
                citations.append(
                    {"source_file": key[0], "chunk_id": key[1]}
                )
        else:
            cited_sentences.append(sentence)
    return " ".join(cited_sentences), citations
