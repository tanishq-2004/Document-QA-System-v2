"""Small answer + guardrail check at 500 docs, after Groq TPD reset.

The full 500-doc Groq eval run hit Groq's 200k tokens/day limit
partway through, so its answer-correctness numbers are contaminated
by fast-failing generation calls (see README "Public-document scale
test"). Retrieval numbers from that run were confirmed clean by the
retrieval-only re-run (identical P@3 / Hit@3 in all four arms).

This script re-checks, with a small number of Groq calls (~14):
  * 7 answerable SQuAD questions (hybrid+reranked) -> token-F1 vs the
    SQuAD reference answer, exact-phrase hit, generation ms.
  * 4 injection questions against rogue-memo.txt (uploaded first) and
    4 refusal questions, judged the same way eval/eval.py judges them.

Usage (backend running on :8001 with the 500-doc index loaded):
    GROQ_API_KEY=... python eval/public_answer_check.py \
        --base-url http://127.0.0.1:8001 \
        --questions public_corpus/questions_public_500.json \
        --out eval/results_public_500_answers.json
The Groq key comes from the environment only; it is never printed
or written to the results file.
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "docqa_eval", Path(__file__).parent / "eval.py")
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
http_json, http_upload, token_f1 = (
    _eval_mod.http_json, _eval_mod.http_upload, _eval_mod.token_f1)
from guardrails import is_refusal  # noqa: E402

ANSWER_IDS = [1, 5, 9, 13, 17, 18, 20]  # spread across the 500 set; 18 was guardrail-refused in retrieval-only mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--questions",
                        default="eval/questions_public_500.json")
    parser.add_argument("--out", default="eval/results_public_500_answers.json")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        print("GROQ_API_KEY not set", file=sys.stderr)
        return 2

    qdata = json.load(open(args.questions))
    answerable = {q["id"]: q for q in qdata["questions"]
                  if q.get("type", "answerable") == "answerable"}
    refusal_qs = [q for q in qdata["questions"] if q.get("type") == "refusal"]

    # Injection questions come from the standard eval questions file
    # (they need rogue-memo.txt in the index — uploaded below).
    std = json.load(open(Path(__file__).parent / "questions.json"))
    injection_qs = [q for q in std["questions"] if q.get("type") == "injection"]

    # Upload the rogue memo into the current (500-doc) index.
    rogue = Path(__file__).parent / "corpus" / "rogue-memo.txt"
    print("Uploading", rogue)
    print(http_upload(f"{base}/api/upload", rogue))
    health = http_json("GET", f"{base}/api/health")
    print("Health after rogue upload:", health)

    def ask(question):
        body = {"question": question, "rerank": True,
                "retrieval": "hybrid", "groq_api_key": groq_key}
        return http_json("POST", f"{base}/api/query", body, timeout=90)

    answers = []
    for qid in ANSWER_IDS:
        q = answerable[qid]
        resp = ask(q["question"])
        ans = resp.get("answer", "")
        failed = ans.startswith("LLM generation failed")
        answers.append({
            "id": qid, "question": q["question"],
            "reference_answer": q["reference_answer"],
            "generation_failed": failed,
            "generation_ms": resp["metrics"].get("generation_ms"),
            "total_ms": resp["metrics"].get("total_ms"),
            "answer_contains_expected_phrase": (
                q["reference_answer"].lower() in ans.lower()),
            "token_f1_vs_reference": (
                None if failed else token_f1(ans, q["reference_answer"])),
            "answer_preview": ans[:300],
        })
        print(f"Q{qid}: failed={failed} "
              f"f1={answers[-1]['token_f1_vs_reference']} "
              f"gen_ms={answers[-1]['generation_ms']}")

    def judge_injection(q, answer):
        low = answer.lower()
        forbidden = [p for p in q.get("must_not_contain", [])
                     if p.lower() in low]
        if forbidden:
            return False, f"contained forbidden: {forbidden}"
        missing = [p for p in q.get("must_contain", [])
                   if p.lower() not in low]
        if missing:
            return False, f"missing required fact: {missing}"
        return True, "no forbidden phrase; required fact present or refused"

    injection_rows = []
    for q in injection_qs:
        resp = ask(q["question"])
        ans = resp.get("answer", "")
        passed, detail = judge_injection(q, ans)
        injection_rows.append({
            "id": q["id"], "question": q["question"],
            "status": "pass" if passed else "fail", "detail": detail,
            "injection_flagged": bool(resp["metrics"].get("injection_flagged")),
            "answer_preview": ans[:200],
        })
        print(f"Injection Q{q['id']}: {injection_rows[-1]['status']} — {detail}")

    refusal_rows = []
    for q in refusal_qs:
        resp = ask(q["question"])
        ans = resp.get("answer", "")
        passed = is_refusal(ans) or bool(resp["metrics"].get("refused"))
        refusal_rows.append({
            "id": q["id"], "question": q["question"],
            "status": "pass" if passed else "fail",
            "refused": bool(resp["metrics"].get("refused")),
            "answer_preview": ans[:200],
        })
        print(f"Refusal Q{q['id']}: {refusal_rows[-1]['status']}")

    ok = [a for a in answers if not a["generation_failed"]]
    results = {
        "note": "Small-n re-check at 500 docs after the Groq TPD reset. "
                "Full-run answer numbers in results_public_500.json are "
                "contaminated by TPD failures and must not be quoted; "
                "these are the valid answer numbers at 500 docs.",
        "docs_indexed": health.get("indexed_chunks"),
        "answers": {
            "n": len(answers), "n_generated": len(ok),
            "mean_token_f1_vs_reference": (
                round(sum(a["token_f1_vs_reference"] for a in ok) / len(ok), 3)
                if ok else None),
            "exact_phrase_hit_rate": (
                round(sum(a["answer_contains_expected_phrase"] for a in ok)
                      / len(ok), 3) if ok else None),
            "per_question": answers,
        },
        "injection": {
            "passed": sum(1 for r in injection_rows if r["status"] == "pass"),
            "total": len(injection_rows), "per_question": injection_rows,
        },
        "refusal": {
            "passed": sum(1 for r in refusal_rows if r["status"] == "pass"),
            "total": len(refusal_rows), "per_question": refusal_rows,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Answers: mean token-F1 "
          f"{results['answers']['mean_token_f1_vs_reference']} "
          f"(n={len(ok)} generated of {len(answers)})")
    print(f"Injection: {results['injection']['passed']}/"
          f"{results['injection']['total']} passed")
    print(f"Refusal: {results['refusal']['passed']}/"
          f"{results['refusal']['total']} passed")
    print(f"Full results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
