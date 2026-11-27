"""Download the SQuAD v1.1 dev set and build nested public corpora.

Source (public, CC BY-SA 4.0):
    https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
    https://rajpurkar.github.io/SQuAD-explorer/  (dataset page)

Each unique SQuAD context becomes one .txt document (title + context).
Corpora are nested and deterministic (seed 42 shuffle, first 100 are a
subset of the first 200, which are a subset of the first 500), so index
times and retrieval scores at different scales are comparable.

Ground truth: for each sampled question the expected phrase is the
SQuAD answer span, and chunk_ids are computed with the app's own
splitter (text_utils.split_text, 500 chars / 50 overlap) on the exact
file content, so they match what the backend indexes. Only questions
whose answer span appears in at least one chunk are kept.

The corpus files themselves are NOT committed to git (see .gitignore:
eval/corpus_public/ and public_corpus/). Only this script, the
questions files and the eval results JSON are committed; anyone can
re-run this script to reproduce the corpus locally.

Usage:
    python eval/download_public_corpus.py --out-dir eval/corpus_public
    python eval/download_public_corpus.py --out-dir /path/to/public_corpus \
        --sizes 100 200 500 --questions-per-size 20
"""

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_utils import clean_text, split_text  # noqa: E402

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
SQUAD_PAGE = "https://rajpurkar.github.io/SQuAD-explorer/"
LICENCE = "CC BY-SA 4.0"
SEED = 42

# Refusal questions: facts about documents that are NOT in this corpus
# (reused from eval/questions.json — Aurora Solar is a fictional company
# from the original synthetic corpus, absent from SQuAD).
REFUSAL_QUESTIONS = [
    {"id": 901, "type": "refusal",
     "question": "How many days of paid annual leave do full-time employees at Aurora Solar receive?"},
    {"id": 902, "type": "refusal",
     "question": "What is Aurora Solar's policy on working from home on Fridays?"},
    {"id": 903, "type": "refusal",
     "question": "How much does Aurora Solar reimburse per day for travel outside Jaipur city limits?"},
    {"id": 904, "type": "refusal",
     "question": "What is the name of Aurora Solar's chief executive officer?"},
]


def download_squad(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Using existing download: {dest}")
        return dest
    print(f"Downloading {SQUAD_URL} ...")
    urllib.request.urlretrieve(SQUAD_URL, dest)
    print(f"Downloaded {dest} ({dest.stat().st_size} bytes)")
    return dest


def collect_contexts(squad_path: Path):
    """Return a deterministically shuffled list of unique contexts.

    Each item: {"title": ..., "context": ..., "qas": [...]}.
    Deduplicated by SHA-256 of the context text.
    """
    with open(squad_path) as f:
        data = json.load(f)
    seen = set()
    contexts = []
    for article in data["data"]:
        title = article["title"]
        for para in article["paragraphs"]:
            ctx = para["context"]
            h = hashlib.sha256(ctx.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            contexts.append({"title": title, "context": ctx, "qas": para["qas"]})
    rng = random.Random(SEED)
    rng.shuffle(contexts)
    return contexts, data.get("version", "1.1")


def chunk_ids_for_answer(file_text: str, filename: str, answer: str):
    """Chunk ids (app splitter) whose text contains the answer span."""
    cleaned_answer = clean_text(answer)
    chunks = split_text(clean_text(file_text), filename)
    ids = [i for i, c in enumerate(chunks)
           if cleaned_answer.lower() in c.page_content.lower()]
    return ids, len(chunks)


def build_size(contexts, size, n_questions, out_root: Path,
               min_bytes: int = 0):
    corpus_dir = out_root / f"corpus_{size}"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    if min_bytes:
        contexts = [c for c in contexts
                    if len((c["title"] + "\n\n" + c["context"])
                           .encode("utf-8")) >= min_bytes]
    subset = contexts[:size]

    # Write one .txt per context: title line + context.
    for i, ctx in enumerate(subset, start=1):
        fname = f"squad-{i:04d}.txt"
        (corpus_dir / fname).write_text(
            f"{ctx['title']}\n\n{ctx['context']}", encoding="utf-8")

    # Sample questions: evenly spaced contexts, one QA each, verified
    # against the app's splitter on the exact file content.
    questions = []
    step = max(1, size // n_questions)
    picked = 0
    idx = 0
    qid = 1
    while picked < n_questions and idx < size:
        ctx = subset[idx]
        fname = f"squad-{idx + 1:04d}.txt"
        file_text = (corpus_dir / fname).read_text(encoding="utf-8")
        for qa in ctx["qas"]:
            answers = qa.get("answers", [])
            if not answers:
                continue
            answer = answers[0]["text"]
            if len(answer.strip()) < 2:
                continue
            ids, _n = chunk_ids_for_answer(file_text, fname, answer)
            if not ids:
                continue
            # Keyword question if the answer is a number or a short
            # capitalised name — the cases BM25 should help most with.
            keyword = bool(any(ch.isdigit() for ch in answer)
                           or (len(answer.split()) <= 3
                               and answer[:1].isupper()))
            questions.append({
                "id": qid,
                "question": qa["question"],
                "source": fname,
                "chunk_id": ids[0],
                "chunk_ids": ids,
                "expected_phrase": answer,
                "reference_answer": answer,
                "keyword": keyword,
                "type": "answerable",
            })
            qid += 1
            picked += 1
            break
        idx += step

    questions.extend(REFUSAL_QUESTIONS)
    qpath = out_root / f"questions_public_{size}.json"
    with open(qpath, "w") as f:
        json.dump({
            "source": SQUAD_URL,
            "licence": LICENCE,
            "corpus_size": size,
            "method": "SQuAD v1.1 dev contexts as .txt docs (title + context)"
                      + (f"; only contexts whose .txt is >= {min_bytes} bytes"
                         if min_bytes else "")
                      + "; one sampled question per context, answer span verified "
                      "to sit in the app-splitter chunks of that file; "
                      "4 refusal questions about Aurora Solar, absent from SQuAD.",
            "questions": questions,
        }, f, indent=2)
    print(f"size {size}: {len(subset)} docs in {corpus_dir}, "
          f"{len(questions)} questions ({picked} answerable + 4 refusal) "
          f"-> {qpath}")
    return len(subset), len(questions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="eval/corpus_public")
    parser.add_argument("--squad-file", default=None,
                        help="Existing dev-v1.1.json (default: download it)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 500])
    parser.add_argument("--questions-per-size", type=int, default=20)
    parser.add_argument("--min-bytes", type=int, default=0,
                        help="Only keep contexts whose .txt file is at "
                             "least this many bytes (e.g. 1000 for the "
                             "bigger-documents scale test)")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    squad_path = (Path(args.squad_file) if args.squad_file
                  else out_root / "dev-v1.1.json")
    if not squad_path.exists():
        download_squad(squad_path)
    squad_sha = hashlib.sha256(squad_path.read_bytes()).hexdigest()

    contexts, version = collect_contexts(squad_path)
    print(f"SQuAD v{version}: {len(contexts)} unique contexts "
          f"(dev file SHA-256 {squad_sha[:16]}...)")

    if args.min_bytes:
        kept = [c for c in contexts
                if len((c["title"] + "\n\n" + c["context"])
                       .encode("utf-8")) >= args.min_bytes]
        print(f"min-bytes filter {args.min_bytes}: {len(kept)} of "
              f"{len(contexts)} contexts qualify")
        contexts = kept

    summary = {}
    for size in args.sizes:
        if size > len(contexts):
            print(f"Skipping size {size}: only {len(contexts)} unique contexts")
            continue
        n_docs, n_q = build_size(contexts, size, args.questions_per_size,
                                 out_root, args.min_bytes)
        summary[size] = {"docs": n_docs, "questions": n_q}

    note = {
        "source_url": SQUAD_URL,
        "dataset_page": SQUAD_PAGE,
        "licence": LICENCE,
        "version": version,
        "dev_file_sha256": squad_sha,
        "unique_contexts_available": len(contexts),
        "min_bytes": args.min_bytes,
        "seed": SEED,
        "sizes_built": summary,
        "note": "Corpus .txt files are NOT committed to git. Re-run this "
                "script to reproduce them locally.",
    }
    with open(out_root / "SOURCE.md", "w") as f:
        f.write("# Public corpus — SQuAD v1.1 dev\n\n")
        for k, v in note.items():
            f.write(f"- **{k}**: {v}\n")
    print(f"Wrote {out_root / 'SOURCE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
