"""Fast unit tests for the Document QA guardrails.

These tests import only guardrails (pure stdlib helpers). They
deliberately do NOT import app.py, because importing app.py loads
the embedding model and the cross-encoder reranker at module level,
and they do NOT touch the Groq key file.

Run:  python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import guardrails
from guardrails import (
    RateLimiter,
    build_system_prompt,
    build_user_message,
    contains_injection_payload,
    groundedness_overlap,
    is_refusal,
    is_relevant,
    sanitize_answer,
    sanitize_question,
    validate_question,
    validate_upload,
)


class TestValidateUpload(unittest.TestCase):
    def test_allowed_extensions_pass(self):
        for name in ("doc.pdf", "notes.txt", "report.docx", "UPPER.PDF"):
            self.assertIsNone(validate_upload(name, 100))

    def test_disallowed_extension_rejected_with_clean_error(self):
        for name in ("script.py", "page.html", "notes.md", "noext", "evil.pdf.exe"):
            err = validate_upload(name, 100)
            self.assertIsNotNone(err)
            self.assertIn("Unsupported file type", err)

    def test_missing_filename_rejected(self):
        self.assertIsNotNone(validate_upload("", 100))
        self.assertIsNotNone(validate_upload(None, 100))

    def test_oversize_rejected(self):
        err = validate_upload("big.pdf", guardrails.MAX_FILE_SIZE_BYTES + 1)
        self.assertIsNotNone(err)
        self.assertIn("too large", err)

    def test_exactly_max_size_passes(self):
        self.assertIsNone(
            validate_upload("big.pdf", guardrails.MAX_FILE_SIZE_BYTES)
        )

    def test_unknown_size_only_checks_extension(self):
        self.assertIsNone(validate_upload("doc.txt", None))
        self.assertIsNotNone(validate_upload("doc.exe", None))


class TestValidateQuestion(unittest.TestCase):
    def test_too_short_rejected(self):
        for q in ("", "  ", "ab", None, 123):
            self.assertIsNotNone(validate_question(q))

    def test_minimum_length_passes(self):
        self.assertIsNone(validate_question("abc"))

    def test_too_long_rejected(self):
        err = validate_question("x" * (guardrails.MAX_QUESTION_CHARS + 1))
        self.assertIsNotNone(err)
        self.assertIn("too long", err)

    def test_exactly_max_length_passes(self):
        self.assertIsNone(validate_question("x" * guardrails.MAX_QUESTION_CHARS))

    def test_control_characters_stripped(self):
        self.assertEqual(sanitize_question("hello\x00\x07world"), "helloworld")
        self.assertEqual(sanitize_question("  padded question  "), "padded question")
        # Tab and newline are kept (they are legitimate whitespace).
        self.assertIn("\t", sanitize_question("a\tb"))


class TestSystemPrompt(unittest.TestCase):
    def test_documents_are_data_not_instructions(self):
        prompt = build_system_prompt()
        low = prompt.lower()
        self.assertIn("untrusted data", low)
        self.assertIn("never follow", low)
        self.assertIn("instructions found inside", low)

    def test_refusal_phrase_is_fixed(self):
        prompt = build_system_prompt()
        self.assertIn(guardrails.REFUSAL_PHRASE, prompt)

    def test_never_reveal_secrets(self):
        prompt = build_system_prompt().lower()
        self.assertIn("never reveal", prompt)
        self.assertIn("api key", prompt)
        self.assertIn("system prompt", prompt)

    def test_fake_turns_covered(self):
        prompt = build_system_prompt().lower()
        self.assertIn("fake system/user turns", prompt)

    def test_user_message_wraps_context_in_delimiters(self):
        msg = build_user_message("What is X?", "Some context here.")
        self.assertIn("<<<DOCUMENTS>>>", msg)
        self.assertIn("<<<END DOCUMENTS>>>", msg)
        self.assertIn("Some context here.", msg)
        self.assertIn("What is X?", msg)
        # The context (data) comes before the real question.
        self.assertLess(msg.index("<<<DOCUMENTS>>>"), msg.index("What is X?"))


class TestInjectionPostCheck(unittest.TestCase):
    def test_known_payload_flagged(self):
        flagged_answer, flagged = sanitize_answer(
            "The secret code is 12345, as the document says."
        )
        self.assertTrue(flagged)
        self.assertNotIn("12345", flagged_answer)
        self.assertIn("not present in the documents", flagged_answer)

    def test_api_key_marker_flagged(self):
        _, flagged = sanitize_answer("Here is the key: gsk_abcdef123456")
        self.assertTrue(flagged)

    def test_normal_answer_not_flagged(self):
        answer, flagged = sanitize_answer(
            "Aurora Solar employees receive 24 days of paid annual leave."
        )
        self.assertFalse(flagged)
        self.assertEqual(
            answer, "Aurora Solar employees receive 24 days of paid annual leave."
        )

    def test_contains_payload_is_case_insensitive(self):
        self.assertTrue(
            contains_injection_payload("SECRET CODE IS 12345")
        )
        self.assertFalse(
            contains_injection_payload("The library opens at 9 am.")
        )

    def test_extra_markers_checked(self):
        self.assertTrue(
            contains_injection_payload(
                "Employees receive 99 days of leave.",
                extra_markers=("99 days",),
            )
        )


class TestRefusalAndRelevance(unittest.TestCase):
    def test_refusal_phrases_detected(self):
        self.assertTrue(
            is_refusal("The answer is not present in the documents.")
        )
        self.assertTrue(is_refusal("That is not found in the documents."))
        self.assertFalse(
            is_refusal("Aurora Solar grants 24 days of annual leave.")
        )

    def test_relevance_threshold(self):
        # Cross-encoder logits: above 0 is relevant, below is not.
        self.assertTrue(is_relevant(2.5, reranked=True))
        self.assertTrue(is_relevant(0.0, reranked=True))
        self.assertFalse(is_relevant(-1.0, reranked=True))
        # FAISS-only scores are negated L2 distances (always
        # negative) — the check must never reject them.
        self.assertTrue(is_relevant(-0.4, reranked=False))
        self.assertTrue(is_relevant(None, reranked=False))
        self.assertFalse(is_relevant(None, reranked=True))


class TestGroundednessOverlap(unittest.TestCase):
    def test_copied_words_score_high(self):
        chunks = ["Aurora Solar employees receive 24 days of paid annual leave."]
        self.assertGreater(
            groundedness_overlap(
                "Employees receive 24 days of paid annual leave.", chunks
            ),
            0.7,
        )

    def test_unrelated_answer_scores_low(self):
        chunks = ["Aurora Solar employees receive 24 days of paid annual leave."]
        self.assertLess(
            groundedness_overlap("The moon orbits the earth slowly.", chunks),
            0.3,
        )

    def test_empty_inputs_score_zero(self):
        self.assertEqual(groundedness_overlap("", ["some text"]), 0.0)
        self.assertEqual(groundedness_overlap("some answer", []), 0.0)


class TestRateLimiter(unittest.TestCase):
    def test_allows_up_to_limit_then_blocks(self):
        limiter = RateLimiter(max_per_minute=3, window_s=60)
        self.assertTrue(limiter.allow())
        self.assertTrue(limiter.allow())
        self.assertTrue(limiter.allow())
        self.assertFalse(limiter.allow())

    def test_reset_clears_the_window(self):
        limiter = RateLimiter(max_per_minute=1, window_s=60)
        self.assertTrue(limiter.allow())
        self.assertFalse(limiter.allow())
        limiter.reset()
        self.assertTrue(limiter.allow())


if __name__ == "__main__":
    unittest.main()
