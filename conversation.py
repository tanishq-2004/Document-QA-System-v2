"""Multi-turn follow-up handling (pure helpers; the LLM rewrite call
lives in app.py).

A follow-up question ("What year was that?") cannot be answered
without the conversation, so it is rewritten into a standalone
question ("What year was Fryatt's revenue in the trial balance?")
before retrieval. Only the last 3 turns are sent, and the rewrite is
only attempted when the question actually looks like a follow-up —
a standalone question is never rewritten, so single-shot behaviour
is unchanged. If the rewrite fails or there is no API key, the
original question is used as-is.
"""

import re

MAX_HISTORY_TURNS = 3

_FOLLOWUP_PRONOUNS = {
    "it", "that", "those", "they", "he", "she", "this", "these",
    "its", "their", "there", "them", "him", "her", "itself",
}

_FOLLOWUP_STARTERS = (
    "and ", "what about", "how about", "also ", "then ",
    "and what", "and how", "and when", "and where", "and why",
)

_WORD_RE = re.compile(r"[a-z]+")


def truncate_history(history, max_turns=MAX_HISTORY_TURNS):
    """Keep the last ``max_turns`` turns as clean question/answer pairs."""
    turns = []
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        q = str(turn.get("question", "")).strip()
        a = str(turn.get("answer", "")).strip()
        if q and a:
            turns.append({"question": q, "answer": a})
    return turns[-max_turns:]


def looks_like_followup(question):
    """True when the question likely refers back to the conversation."""
    words = _WORD_RE.findall(str(question).lower())
    if not words:
        return False
    if len(words) <= 5:
        return True
    if any(w in _FOLLOWUP_PRONOUNS for w in words):
        return True
    lowered = str(question).strip().lower()
    return any(lowered.startswith(s) for s in _FOLLOWUP_STARTERS)


def build_rewrite_messages(history, question):
    """Chat messages for the rewrite call: history + the follow-up."""
    lines = []
    for turn in truncate_history(history):
        lines.append(f"Q: {turn['question']}")
        lines.append(f"A: {turn['answer']}")
    history_text = "\n".join(lines)
    return [
        {
            "role": "system",
            "content": (
                "Rewrite the user's latest question as a standalone "
                "question that can be understood without the "
                "conversation history. Use the history only to resolve "
                "pronouns and missing context (what 'it', 'that' or "
                "'there' refers to). If the question is already "
                "standalone, return it unchanged. Output ONLY the "
                "rewritten question, no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{history_text}\n\n"
                f"Latest question: {question}"
            ),
        },
    ]
