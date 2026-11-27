"""Unit tests for conversation helpers (pure stdlib, no ML models).

Run:  python -m pytest tests/test_conversation.py -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conversation import (
    build_rewrite_messages,
    looks_like_followup,
    truncate_history,
)

HISTORY = [
    {"question": "What was Fryatt's revenue?", "answer": "$630,000."},
    {"question": "What was Spearmint's share capital?", "answer": "$500,000."},
    {"question": "What is the capital of France?", "answer": "Paris."},
    {"question": "What is the capital of Spain?", "answer": "Madrid."},
]


class TestTruncateHistory(unittest.TestCase):
    def test_keeps_last_three_turns(self):
        kept = truncate_history(HISTORY)
        self.assertEqual(len(kept), 3)
        self.assertEqual(kept[0]["question"], "What was Spearmint's share capital?")
        self.assertEqual(kept[-1]["question"], "What is the capital of Spain?")

    def test_drops_empty_turns(self):
        kept = truncate_history([{"question": "", "answer": "x"}, HISTORY[0]])
        self.assertEqual(len(kept), 1)

    def test_non_dict_entries_ignored(self):
        self.assertEqual(truncate_history(["nope", HISTORY[0]]), [HISTORY[0]])

    def test_none_history_is_empty(self):
        self.assertEqual(truncate_history(None), [])


class TestLooksLikeFollowup(unittest.TestCase):
    def test_pronoun_question_is_followup(self):
        self.assertTrue(looks_like_followup("What year was that?"))

    def test_short_question_is_followup(self):
        self.assertTrue(looks_like_followup("And the other one?"))

    def test_what_about_is_followup(self):
        self.assertTrue(looks_like_followup("What about Spearmint?"))

    def test_standalone_question_is_not_followup(self):
        self.assertFalse(
            looks_like_followup(
                "What was Fryatt's revenue in the trial balance at 31 May 20X7?"
            )
        )


class TestBuildRewriteMessages(unittest.TestCase):
    def test_history_and_question_in_messages(self):
        msgs = build_rewrite_messages(HISTORY[:1], "What year was that?")
        self.assertEqual(len(msgs), 2)
        self.assertIn("standalone", msgs[0]["content"].lower())
        self.assertIn("Fryatt's revenue", msgs[1]["content"])
        self.assertIn("What year was that?", msgs[1]["content"])

    def test_output_only_instruction_present(self):
        msgs = build_rewrite_messages(HISTORY[:1], "What year was that?")
        self.assertIn("ONLY", msgs[0]["content"])


class TestRewriteBudget(unittest.TestCase):
    """The rewrite call's token budget must fit the model's reasoning
    (regression test — live failure, 27 Nov 2026).

    gpt-oss-20b spent 98 of a 100-token budget reasoning and returned
    empty content (finish=length), so "What year was that?" fell back
    to the original question and was refused. At 300 tokens the same
    call returns a proper standalone question. This test reads app.py
    instead of importing it, because importing app.py loads the ML
    models (see the note at the top of this file).
    """

    def test_rewrite_call_asks_for_500_tokens(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo, "app.py")) as f:
            source = f.read()
        start = source.index("def rewrite_followup_question")
        end = source.index("@app.get", start)
        body = source[start:end]
        self.assertIn("max_tokens=500", body)
        self.assertNotIn("max_tokens=100", body)


if __name__ == "__main__":
    unittest.main()
