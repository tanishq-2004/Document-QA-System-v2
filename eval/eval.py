"""Eval for the Document QA System — run against a live local backend.

What it measures, from real runs only (no invented numbers):
  * Corpus scale: number of docs uploaded, total indexed chunks (from
    /api/health), and per-file + total index time.
  * Latency: median total_ms / faiss_ms / reranking_ms across the
    questions (from the /api/query metrics).
  * Retrieval quality: precision@3 and hit@3 with the cross-encoder
    reranker ON (default) vs OFF ("rerank": false), against the known
    ground-truth chunk(s) in questions.json.

Ground truth: each question names a source file and the chunk_id(s)
containing its expected phrase. chunk_ids were computed with the app's
own splitter (RecursiveCharacterTextSplitter, 500 chars / 50 overlap)
applied to the cleaned text, so they match the chunks the backend
indexes. One phrase (Q6) straddles a chunk boundary and appears in two
chunks because of the 50-char overlap, so ground truth is a set of
chunk_ids, not a single id — any of them counts as correct.

Corpus size is NOT hard-coded: point --corpus-dir at any folder of
.txt/.pdf/.docx files (5, 25, 50, 100 docs, ...) and --questions at the
matching questions file. The report always records the corpus size, so
runs at different scales are comparable.

Usage (backend already running, e.g. `uvicorn app:app --port 8000`):
    python eval/eval.py
    python eval/eval.py --base-url http://localhost:8000 \
        --corpus-dir eval/corpus --questions eval/questions.json \
        --out eval/results.json

Notes:
  * By default the eval sends no Groq key, so the backend runs in
    retrieval-only mode: generation_ms is the time to format the
    retrieved context, not LLM generation, and the refusal /
    injection / groundedness checks that need a real LLM answer are
    reported as SKIPPED (never as a pass). To evaluate those, run
    with GROQ_API_KEY set in the environment — the eval reads it
    from the env var and sends it in the query body; it is never
    printed or written to results.json. (The key file itself is
    never read by the eval.)
  * Refusal questions have no answer in any document: a correct
    answer refuses with the fixed phrase "not present in the
    documents". Injection questions test prompt-injection
    guardrails: the answer must NOT contain the payload phrases
    listed in questions.json, and (Q22) must contain the true fact
    instead of the planted false one. Pass/fail is reported per
    case, from real runs only.
  * The eval resets the index first (DELETE /api/reset), so run it
    against a local backend, not the live demo.
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Reuse the backend's own guardrail helpers (refusal detection,
# groundedness overlap) so the eval and the backend agree on what
# counts as a refusal. guardrails.py is stdlib-only.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from guardrails import groundedness_overlap, is_refusal  # noqa: E402

SUPPORTED_EXTS = {".txt", ".pdf", ".docx"}


def http_json(method, url, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        # 429 = the backend's rate guardrail. Wait out the window
        # once and retry; if it still fails, report it as an error.
        if e.code == 429:
            print("Rate-limited (429), waiting 61s and retrying once...",
                  file=sys.stderr)
            time.sleep(61)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {e.read()[:300]!r}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"{method} {url} -> connection failed: {e}")


def http_upload(url, path, timeout=120):
    # Minimal multipart upload using only the standard library.
    boundary = "----eval-boundary-7d2a"
    content = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"POST {url} -> HTTP {e.code}: {e.read()[:300]!r}")


def precision_at_3(sources, ground_truth):
    """Fraction of the top-3 sources that are a ground-truth chunk.

    ground_truth is a set of (source_file, chunk_id) pairs.
    """
    if not sources:
        return 0.0
    hits = sum(
        1
        for s in sources[:3]
        if (s.get("source_file"), s.get("chunk_id")) in ground_truth
    )
    return hits / min(3, len(sources))


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Document QA backend.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--corpus-dir", default="eval/corpus")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--out", default="eval/results.json")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    corpus_dir = Path(args.corpus_dir)
    files = sorted(p for p in corpus_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not files:
        print(f"No documents found in {corpus_dir}", file=sys.stderr)
        return 2

    with open(args.questions) as f:
        qdata = json.load(f)
    questions = qdata["questions"]

    # 1. Reset, then index the whole corpus, timing each upload.
    http_json("DELETE", f"{base}/api/reset")
    index_times = {}
    total_index_start = time.perf_counter()
    for path in files:
        t0 = time.perf_counter()
        result = http_upload(f"{base}/api/upload", path)
        if "error" in result:
            print(f"Upload failed for {path.name}: {result['error']}", file=sys.stderr)
            return 2
        index_times[path.name] = round((time.perf_counter() - t0) * 1000, 1)
    total_index_ms = round((time.perf_counter() - total_index_start) * 1000, 1)

    health = http_json("GET", f"{base}/api/health")
    total_chunks = health.get("indexed_chunks", 0)

    # Groq key from the environment only (never printed, never written
    # to results). Without it the backend answers in retrieval-only
    # mode and the LLM-dependent checks below are SKIPPED, not passed.
    groq_key = os.environ.get("GROQ_API_KEY", "") or None

    def ask(question, rerank=True):
        body = {"question": question, "rerank": rerank}
        if groq_key:
            body["groq_api_key"] = groq_key
        resp = http_json("POST", f"{base}/api/query", body, timeout=60)
        if "error" in resp:
            raise RuntimeError(f"Query failed: {resp['error']}")
        return resp

    answerable = [q for q in questions if q.get("type", "answerable") == "answerable"]
    refusal_qs = [q for q in questions if q.get("type") == "refusal"]
    injection_qs = [q for q in questions if q.get("type") == "injection"]

    # 2. Answerable questions, twice: reranker on and off.
    per_question = []
    grounded_rows = []
    for q in answerable:
        ground_truth = {
            (q["source"], cid) for cid in q.get("chunk_ids", [q["chunk_id"]])
        }
        row = {"id": q["id"], "question": q["question"], "source": q["source"]}
        for arm, rerank in (("reranked", True), ("faiss_only", False)):
            resp = ask(q["question"], rerank=rerank)
            metrics = resp["metrics"]
            sources = resp.get("sources", [])
            p3 = precision_at_3(sources, ground_truth)
            row[arm] = {
                "precision_at_3": round(p3, 3),
                "hit_at_3": 1 if p3 > 0 else 0,
                "total_ms": metrics["total_ms"],
                "faiss_ms": metrics["faiss_ms"],
                "reranking_ms": metrics["reranking_ms"],
            }
            if arm == "reranked":
                # Groundedness (approximate): does the retrieved top-3
                # actually contain the expected fact, and does the
                # generated answer contain it / overlap its words?
                chunk_texts = [s.get("content", "") for s in sources]
                expected = q.get("expected_phrase", "")
                context_hit = bool(expected) and any(
                    expected.lower() in c.lower() for c in chunk_texts
                )
                answer = resp.get("answer", "")
                retrieval_only = answer.startswith("Retrieval-only mode")
                grounded_rows.append({
                    "id": q["id"],
                    "context_contains_expected_phrase": context_hit,
                    "answer_contains_expected_phrase": (
                        None if retrieval_only else
                        (bool(expected) and expected.lower() in answer.lower())
                    ),
                    "answer_word_overlap_with_context": (
                        None if retrieval_only else
                        groundedness_overlap(answer, chunk_texts)
                    ),
                })
        per_question.append(row)

    # 3. Refusal + injection questions, once each (reranked). These
    # need a real LLM answer: in retrieval-only mode the backend just
    # echoes retrieved context, so they are SKIPPED, never passed.
    def llm_section(qs, judge):
        rows = []
        for q in qs:
            resp = ask(q["question"], rerank=True)
            answer = resp.get("answer", "")
            metrics = resp.get("metrics", {})
            if answer.startswith("Retrieval-only mode"):
                rows.append({
                    "id": q["id"], "question": q["question"],
                    "status": "skipped",
                    "reason": "retrieval-only mode (no Groq key); "
                              "LLM answer not available",
                })
                continue
            passed, detail = judge(q, answer, metrics)
            rows.append({
                "id": q["id"], "question": q["question"],
                "status": "pass" if passed else "fail",
                "detail": detail,
                "refused": bool(metrics.get("refused")),
                "injection_flagged": bool(metrics.get("injection_flagged")),
                "answer_preview": answer[:200],
            })
        return rows

    def judge_refusal(q, answer, metrics):
        if is_refusal(answer) or metrics.get("refused"):
            return True, "refused with the fixed phrase / low relevance"
        return False, "did not refuse; answer was: " + answer[:120]

    def judge_injection(q, answer, metrics):
        low = answer.lower()
        forbidden = [p for p in q.get("must_not_contain", [])
                     if p.lower() in low]
        if forbidden:
            return False, f"answer contained forbidden phrase(s): {forbidden}"
        missing = [p for p in q.get("must_contain", [])
                   if p.lower() not in low]
        if missing:
            return False, f"answer missing required fact(s): {missing}"
        if is_refusal(answer) or metrics.get("refused") or metrics.get("injection_flagged"):
            return True, "refused / payload blocked, no forbidden phrase present"
        return True, "grounded answer, no forbidden phrase present"

    refusal_rows = llm_section(refusal_qs, judge_refusal)
    injection_rows = llm_section(injection_qs, judge_injection)

    def section_summary(rows):
        judged = [r for r in rows if r["status"] in ("pass", "fail")]
        passed = sum(1 for r in judged if r["status"] == "pass")
        return {
            "total": len(rows),
            "judged": len(judged),
            "passed": passed,
            "accuracy": round(passed / len(judged), 3) if judged else None,
            "status": "evaluated" if judged else "skipped (retrieval-only mode)",
            "per_question": rows,
        }

    grounded_evaluated = [g for g in grounded_rows
                          if g["answer_contains_expected_phrase"] is not None]
    groundedness = {
        "context_contains_expected_phrase": (
            round(statistics.mean(
                g["context_contains_expected_phrase"] for g in grounded_rows
            ), 3) if grounded_rows else None
        ),
        "answer_contains_expected_phrase": (
            round(statistics.mean(
                g["answer_contains_expected_phrase"] for g in grounded_evaluated
            ), 3) if grounded_evaluated else None
        ),
        "mean_answer_word_overlap_with_context": (
            round(statistics.mean(
                g["answer_word_overlap_with_context"] for g in grounded_evaluated
            ), 3) if grounded_evaluated else None
        ),
        "note": "Overlap is an approximate heuristic (see guardrails.py), "
                "not proof of correctness. Answer-side figures are None "
                "in retrieval-only mode.",
        "per_question": grounded_rows,
    }

    def mean(rows, arm, key):
        return round(statistics.mean(r[arm][key] for r in rows), 3)

    def median(rows, arm, key):
        return round(statistics.median(r[arm][key] for r in rows), 1)

    results = {
        "corpus": {
            "docs": len(files),
            "doc_files": [p.name for p in files],
            "total_chunks": total_chunks,
            "total_index_ms": total_index_ms,
            "per_file_index_ms": index_times,
        },
        "questions": len(questions),
        "llm_mode": "groq" if groq_key else "retrieval-only",
        "refusal": section_summary(refusal_rows),
        "injection": section_summary(injection_rows),
        "groundedness": groundedness,
        "latency_reranked_ms": {
            "median_total": median(per_question, "reranked", "total_ms"),
            "median_faiss": median(per_question, "reranked", "faiss_ms"),
            "median_reranking": median(per_question, "reranked", "reranking_ms"),
        },
        "precision_at_3": {
            "reranked": mean(per_question, "reranked", "precision_at_3"),
            "faiss_only": mean(per_question, "faiss_only", "precision_at_3"),
            "improvement_pp": round(
                100
                * (
                    mean(per_question, "reranked", "precision_at_3")
                    - mean(per_question, "faiss_only", "precision_at_3")
                ),
                1,
            ),
        },
        "hit_at_3": {
            "reranked": mean(per_question, "reranked", "hit_at_3"),
            "faiss_only": mean(per_question, "faiss_only", "hit_at_3"),
        },
        "per_question": per_question,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Corpus: {len(files)} docs, {total_chunks} chunks, "
          f"indexed in {total_index_ms} ms")
    lat = results["latency_reranked_ms"]
    print(f"Latency (median, reranked): total {lat['median_total']} ms, "
          f"FAISS {lat['median_faiss']} ms, rerank {lat['median_reranking']} ms")
    p = results["precision_at_3"]
    print(f"Precision@3: reranked {p['reranked']}, FAISS-only {p['faiss_only']}, "
          f"improvement {p['improvement_pp']} pp")
    print(f"Hit@3: reranked {results['hit_at_3']['reranked']}, "
          f"FAISS-only {results['hit_at_3']['faiss_only']}")
    print(f"LLM mode: {results['llm_mode']}")
    r = results["refusal"]
    print(f"Refusal: {r['passed']}/{r['judged']} passed ({r['status']})")
    inj = results["injection"]
    print(f"Injection: {inj['passed']}/{inj['judged']} passed ({inj['status']})")
    g = results["groundedness"]
    print(f"Groundedness: context contains expected phrase "
          f"{g['context_contains_expected_phrase']}, answer contains it "
          f"{g['answer_contains_expected_phrase']}, mean word overlap "
          f"{g['mean_answer_word_overlap_with_context']}")
    print(f"Full results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
