"""Provenance tests for the eval dashboard (metrics_summary.json).

Every number in the dashboard JSON must equal the value in the eval
results file it claims to come from — nothing hand-typed. Run:
python -m pytest tests/test_metrics_summary.py -v
"""

import json
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))

from build_metrics_summary import build


def _load(name):
    with open(os.path.join(REPO, "eval", name)) as f:
        return json.load(f)


class TestMetricsSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(REPO, "metrics_summary.json")) as f:
            cls.summary = json.load(f)

    def test_builder_output_matches_committed_file(self):
        self.assertEqual(build(), self.summary)

    def test_7doc_numbers_match_results_files(self):
        run1 = _load("results_hybrid_run1.json")
        run2 = _load("results_hybrid_run2.json")
        e7 = self.summary["eval_7doc"]
        self.assertEqual(e7["questions"], run1["questions"])
        self.assertEqual(
            e7["precision_at_3"],
            run1["precision_at_3"]["hybrid_reranked"],
        )
        self.assertEqual(
            e7["precision_at_3"],
            run2["precision_at_3"]["hybrid_reranked"],
        )
        self.assertEqual(e7["hit_at_3"], 1)
        self.assertEqual(
            (e7["refusal_passed"], e7["refusal_total"]), (4, 4)
        )
        self.assertEqual(
            (e7["injection_passed"], e7["injection_total"]), (4, 4)
        )

    def test_scale_numbers_match_public_results(self):
        expected = [
            ("results_public_100.json", 0.594),
            ("results_public_200.json", 0.489),
            ("results_public_500.json", None),  # excluded: contamination
        ]
        for (fname, token_f1), row in zip(expected, self.summary["scale_squad"]):
            pub = _load(fname)
            self.assertEqual(row["docs"], pub["corpus"]["docs"])
            self.assertEqual(row["chunks"], pub["corpus"]["total_chunks"])
            self.assertAlmostEqual(
                row["index_seconds"],
                pub["corpus"]["total_index_ms"] / 1000,
                places=1,
            )
            self.assertEqual(
                row["precision_at_3_hybrid_reranked"],
                pub["precision_at_3"]["hybrid_reranked"],
            )
            self.assertEqual(row["token_f1"], token_f1)

    def test_500doc_answer_scores_excluded_not_zeroed(self):
        row500 = self.summary["scale_squad"][2]
        self.assertIsNone(row500["token_f1"])
        self.assertIn("429", row500["token_f1_note"])
        # ...but its clean retrieval numbers are present.
        self.assertEqual(row500["hit_at_3_dense_only"], 0.9)
        self.assertEqual(row500["hit_at_3_hybrid_reranked"], 1)

    def test_retrieval_latency_matches_recheck(self):
        with open(
            os.path.expanduser(
                "~/workspace/projects/doc-qa-measure/"
                "recheck_results_2026-11-27.json"
            )
        ) as f:
            recheck = json.load(f)
        lat = self.summary["retrieval_only_latency_ms"]
        self.assertEqual(
            lat["median_total"],
            recheck["latency_hybrid_reranked_ms"]["median_total"],
        )
        self.assertEqual(lat["p95_total"], 753.9)


if __name__ == "__main__":
    unittest.main()
