"""Build metrics_summary.json for the in-app eval dashboard.

Reads ONLY the eval results files (no hand-typed numbers) and writes
a small JSON the Streamlit frontend displays in its sidebar. The
500-doc answer scores are deliberately EXCLUDED: that run's
generations were contaminated by Groq 429 rate-limit failures, so its
token-F1 (0.069) is not a real measurement. Its retrieval numbers
(P@3, Hit@3) are clean and are included.

Unit-test count is the one literal here: it is verified by running
pytest, not by a results file. Update it when the suite grows.

Run:  python eval/build_metrics_summary.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Verified by running the suite on 2026-11-27 (96 passed).
UNIT_TESTS_PASSED = 96


def _load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def build():
    run1 = _load("results_hybrid_run1.json")
    run2 = _load("results_hybrid_run2.json")
    pub100 = _load("results_public_100.json")
    pub200 = _load("results_public_200.json")
    pub500 = _load("results_public_500.json")
    recheck_path = os.path.join(
        os.path.expanduser("~"),
        "workspace/projects/doc-qa-measure/"
        "recheck_results_2026-11-27.json",
    )
    with open(recheck_path) as f:
        recheck = json.load(f)

    assert run1["precision_at_3"]["hybrid_reranked"] == run2[
        "precision_at_3"
    ]["hybrid_reranked"], "7-doc runs disagree on P@3"
    assert run1["hit_at_3"]["hybrid_reranked"] == run2["hit_at_3"][
        "hybrid_reranked"
    ], "7-doc runs disagree on Hit@3"

    return {
        "generated_from": [
            "eval/results_hybrid_run1.json",
            "eval/results_hybrid_run2.json",
            "eval/results_public_100.json",
            "eval/results_public_200.json",
            "eval/results_public_500.json",
            "doc-qa-measure/recheck_results_2026-11-27.json",
        ],
        "generated_on": "2026-11-27",
        "notes": (
            "P@3 has a 0.333 ceiling with one right chunk per question, "
            "so Hit@3 is the real retrieval signal. 500-doc answer "
            "scores are excluded: Groq 429 failures contaminated them. "
            "End-to-end latency with live Groq generation varies a lot "
            "(0.79s vs 3.71s medians across runs), so only the "
            "retrieval-only median is shown as a stable latency."
        ),
        "eval_7doc": {
            "questions": run1["questions"],
            "runs": 2,
            "precision_at_3": run1["precision_at_3"]["hybrid_reranked"],
            "hit_at_3": run1["hit_at_3"]["hybrid_reranked"],
            "keyword_precision_at_3": run1["precision_at_3"][
                "keyword_questions"
            ]["hybrid_reranked"],
            "refusal_passed": run1["refusal"]["passed"],
            "refusal_total": run1["refusal"]["total"],
            "injection_passed": run1["injection"]["passed"],
            "injection_total": run1["injection"]["total"],
            "context_contains_expected_phrase": run1["groundedness"][
                "context_contains_expected_phrase"
            ],
        },
        "retrieval_only_latency_ms": {
            "median_total": recheck["latency_hybrid_reranked_ms"][
                "median_total"
            ],
            "p95_total": recheck["latency_hybrid_reranked_ms"]["p95_total"],
            "median_faiss": recheck["latency_hybrid_reranked_ms"][
                "median_faiss"
            ],
            "median_bm25": recheck["latency_hybrid_reranked_ms"][
                "median_bm25"
            ],
            "median_reranking": recheck["latency_hybrid_reranked_ms"][
                "median_reranking"
            ],
        },
        "scale_squad": [
            {
                "docs": pub["corpus"]["docs"],
                "chunks": pub["corpus"]["total_chunks"],
                "index_seconds": round(
                    pub["corpus"]["total_index_ms"] / 1000, 1
                ),
                "precision_at_3_dense_only": pub["precision_at_3"][
                    "dense_only"
                ],
                "precision_at_3_hybrid_reranked": pub["precision_at_3"][
                    "hybrid_reranked"
                ],
                "hit_at_3_dense_only": pub["hit_at_3"]["dense_only"],
                "hit_at_3_hybrid_reranked": pub["hit_at_3"][
                    "hybrid_reranked"
                ],
                "token_f1": pub["answer_correctness"][
                    "mean_token_f1_vs_reference"
                ]
                if pub is not pub500
                else None,
                "token_f1_note": (
                    "excluded: Groq 429 contamination"
                    if pub is pub500
                    else None
                ),
            }
            for pub in (pub100, pub200, pub500)
        ],
        "unit_tests_passed": UNIT_TESTS_PASSED,
    }


if __name__ == "__main__":
    summary = build()
    out = os.path.join(ROOT, "metrics_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")
