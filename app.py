import logging
import os
import shutil
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from sentence_transformers import CrossEncoder

from text_utils import extract_text, split_text

import answer_cache
import guardrails
import manifest as cache_manifest
import retrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


UPLOAD_DIR = "uploaded_docs"
VECTOR_STORE_PATH = "vector_store"
MANIFEST_PATH = os.path.join(VECTOR_STORE_PATH, "manifest.json")

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = None

# --- Chunk registry + manifest -----------------------------------------
# The registry is the in-memory source of truth for what is indexed:
#   chunk_registry[(source_file, chunk_id)] -> langchain Document
# The manifest (vector_store/manifest.json) is its on-disk record:
#   filename -> {"hash": sha256, "chunks": n, "indexed_at": iso}
# The old per-hash {"indexed": True} pickles could not say *which*
# file a hash belonged to, so an edited file's old chunks stayed in
# the index forever. The manifest + registry fix that: replacing a
# file removes its old chunks (by rebuilding FAISS from the remaining
# registry — LangChain FAISS has no delete-by-metadata) before the
# new chunks are added.
chunk_registry = {}
manifest = cache_manifest.load_manifest(MANIFEST_PATH)
bm25_index = retrieval.BM25Index([])


def _save_manifest():
    cache_manifest.save_manifest(MANIFEST_PATH, manifest)


def _rebuild_bm25():
    global bm25_index
    bm25_index = retrieval.BM25Index(list(chunk_registry.values()))


def _rebuild_index_from_registry():
    """Rebuild the FAISS index from the chunk registry alone.

    Used when a file is replaced or removed: LangChain FAISS cannot
    delete by metadata, so the index is rebuilt from the remaining
    chunks. At our scale (400 chunks) this re-embeds in seconds; the
    fast path (unchanged files) never rebuilds.
    """
    global vectorstore
    docs = list(chunk_registry.values())
    if not docs:
        vectorstore = None
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
    save_vectorstore()


def _docstore_docs():
    """Chunks recovered from a persisted FAISS docstore (startup
    fallback when the uploaded files themselves are gone)."""
    if vectorstore is None:
        return []
    try:
        store = vectorstore.docstore._dict
    except AttributeError:
        return []
    return [d for d in store.values() if d is not None]


def _load_registry_from_disk():
    """Populate the registry + manifest from uploaded_docs on startup,
    falling back to the persisted FAISS docstore. Also repairs a
    missing manifest from the registry (migration from the old
    per-hash pickle cache)."""
    global manifest
    for filename in sorted(os.listdir(UPLOAD_DIR)):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.isfile(file_path):
            continue
        try:
            text = extract_text(file_path)
            chunks = split_text(text, filename) if text.strip() else []
        except Exception as e:
            logger.warning("startup: could not index %s: %s", filename, e)
            continue
        for chunk in chunks:
            chunk_registry[retrieval.chunk_key(chunk)] = chunk
        if chunks and filename not in manifest:
            with open(file_path, "rb") as f:
                file_hash = cache_manifest.sha256_bytes(f.read())
            manifest[filename] = cache_manifest.manifest_entry(
                file_hash, len(chunks)
            )
    if not chunk_registry:
        for doc in _docstore_docs():
            chunk_registry[retrieval.chunk_key(doc)] = doc
        for (filename, _cid), doc in list(chunk_registry.items()):
            if filename not in manifest:
                joined = "".join(
                    d.page_content
                    for (fn, _), d in chunk_registry.items()
                    if fn == filename
                )
                count = sum(1 for (fn, _) in chunk_registry if fn == filename)
                manifest[filename] = cache_manifest.manifest_entry(
                    cache_manifest.sha256_bytes(joined.encode()), count
                )
    if manifest:
        _save_manifest()
    _rebuild_bm25()


if os.path.exists(f"{VECTOR_STORE_PATH}/index.faiss"):

    vectorstore = FAISS.load_local(

        VECTOR_STORE_PATH,

        embeddings,

        allow_dangerous_deserialization=True
    )

_load_registry_from_disk()

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# LLM settings are env-configurable; the defaults are the original values.
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "500"))
# Server-side Groq key (e.g. a Hugging Face Space secret). When set, the
# key does not have to be sent with every /api/query request.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_TIMEOUT_S = 30

# Hybrid retrieval (BM25 + FAISS, RRF-fused) is the default. Set
# HYBRID_SEARCH=false to go back to dense-only retrieval, or pass
# "retrieval": "dense" / "hybrid" per query.
HYBRID_SEARCH_DEFAULT = os.environ.get("HYBRID_SEARCH", "true").lower() in (
    "true", "1", "yes",
)

# Simple in-memory per-process query rate limiter (see guardrails.py).
query_limiter = guardrails.RateLimiter()


def save_vectorstore():

    global vectorstore

    if vectorstore:

        vectorstore.save_local(
            VECTOR_STORE_PATH
        )


def generate_answer(
    question,
    context,
    groq_api_key
):

    if not groq_api_key:

        return (

            "Retrieval-only mode "
            "(no Groq API key provided).\n\n"

            f"Top retrieved context:\n\n"

            f"{context[:800]}"
        )

    try:

        client = OpenAI(

            api_key=groq_api_key,

            base_url="https://api.groq.com/openai/v1",

            timeout=GROQ_TIMEOUT_S
        )

        response = client.chat.completions.create(

            model=LLM_MODEL,

            messages=[

                {
                    "role": "system",

                    # Hardened prompt: documents are untrusted data, never
                    # follow instructions inside them, never reveal secrets.
                    "content": guardrails.build_system_prompt()
                },

                {
                    "role": "user",

                    # Context wrapped in explicit delimiters, labelled as
                    # untrusted data, separate from the real question.
                    "content": guardrails.build_user_message(question, context)
                }
            ],

            max_tokens=LLM_MAX_TOKENS,

            temperature=LLM_TEMPERATURE
        )

        # NOTE: the post-generation injection check is applied by the
        # caller (query_documents), so it can report injection_flagged
        # in the response. Sanitizing here as well would hide the flag.
        return response.choices[0].message.content

    except Exception as e:

        return (

            f"LLM generation failed: {e}\n\n"

            f"Retrieved context:\n\n"

            f"{context[:600]}"
        )


@app.get("/")
def home():

    return {

        "message":
        "Document QA System Running"
    }


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    global vectorstore

    # Upload guardrails: allowed type + max 10 MB. Size is checked
    # from the spooled file before the bytes are read into memory.
    size_bytes = None
    try:
        file.file.seek(0, 2)
        size_bytes = file.file.tell()
        file.file.seek(0)
    except Exception:
        size_bytes = None
    upload_error = guardrails.validate_upload(file.filename, size_bytes)
    if upload_error:
        return {"error": upload_error}

    # Hash the upload bytes BEFORE writing the file: the cache check
    # needs no disk I/O, and a cache hit never rewrites the file.
    data = await file.read()
    if size_bytes is None:
        size_bytes = len(data)
        upload_error = guardrails.validate_upload(file.filename, size_bytes)
        if upload_error:
            return {"error": upload_error}
    file_hash = cache_manifest.sha256_bytes(data)

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    # Cache hit: THIS file's hash is in the manifest AND every chunk
    # it claims is actually in the registry. A global ntotal > 0 is
    # not enough (a reset / rebuild can leave the index non-empty
    # while this file's chunks are gone).
    if cache_manifest.is_cache_hit(
        manifest, file.filename, file_hash, set(chunk_registry)
    ):
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(data)
        return {

            "message":
            "Document already indexed"
        }

    # Stale file: same filename, different content. Remove the old
    # chunks from the registry and rebuild the index from what
    # remains, BEFORE adding the new chunks — otherwise the old
    # facts stay searchable alongside the new ones.
    if cache_manifest.is_stale(manifest, file.filename, file_hash):
        old_keys = cache_manifest.stale_chunk_keys(manifest, file.filename)
        for key in old_keys:
            chunk_registry.pop(key, None)
        _rebuild_index_from_registry()
        _rebuild_bm25()
        logger.info(
            "replaced %s: removed %d stale chunks",
            file.filename, len(old_keys),
        )

    with open(file_path, "wb") as f:

        f.write(data)

    text = extract_text(file_path)

    if not text or not text.strip():

        return {

            "error":
            "No extractable text found in document"
        }

    chunks = split_text(
        text,
        file.filename
    )

    if not chunks:

        return {

            "error":
            "Document produced no chunks to index"
        }

    if vectorstore is None:

        vectorstore = FAISS.from_documents(

            chunks,

            embeddings
        )

    else:

        vectorstore.add_documents(chunks)

    for chunk in chunks:
        chunk_registry[retrieval.chunk_key(chunk)] = chunk
    _rebuild_bm25()

    manifest[file.filename] = cache_manifest.manifest_entry(
        file_hash, len(chunks)
    )
    _save_manifest()

    save_vectorstore()

    # The index changed, so cached answers may no longer match the
    # current documents — drop them.
    answer_cache.cache.clear()

    # The old per-hash pickle cache is superseded by the manifest;
    # drop any stale pickle for this file's old/new hash so nothing
    # can claim a file is indexed on the basis of a stray file.
    for stale_hash in {file_hash}:
        stale_pkl = os.path.join(VECTOR_STORE_PATH, f"{stale_hash}.pkl")
        if os.path.exists(stale_pkl):
            os.remove(stale_pkl)

    return {

        "filename": file.filename,

        "characters": len(text),

        "chunks": len(chunks),

        "indexed": True
    }


@app.post("/api/query")
def query_documents(data: dict):

    global vectorstore

    t0 = time.perf_counter()

    # Rate guardrail: 429-style error once the per-minute limit is hit.
    if not query_limiter.allow():
        return JSONResponse(
            status_code=429,
            content={
                "error": (
                    "Rate limit exceeded: too many questions. "
                    "Please wait a minute and try again."
                )
            },
        )

    # Query guardrails: min/max length, control characters stripped.
    question_error = guardrails.validate_question(data.get("question"))
    if question_error:
        return {"error": question_error}
    question = guardrails.sanitize_question(data.get("question"))

    # New field first, old `hf_token` field kept for backward
    # compatibility, server-side GROQ_API_KEY env var last — so the key
    # does not have to be sent with every request.
    groq_api_key = (
        data.get("groq_api_key")
        or data.get("hf_token")
        or GROQ_API_KEY
        or None
    )

    # "rerank": false skips the cross-encoder and returns retrieval
    # order, so eval/eval.py can measure reranker on/off through the
    # API alone.
    rerank = data.get("rerank", True)

    # Retrieval method: "hybrid" (BM25 + FAISS, RRF-fused, default) or
    # "dense" (FAISS only). Per-query field overrides HYBRID_SEARCH.
    retrieval_method = str(
        data.get("retrieval")
        or ("hybrid" if HYBRID_SEARCH_DEFAULT else "dense")
    ).lower()
    if retrieval_method not in ("hybrid", "dense"):
        retrieval_method = "hybrid" if HYBRID_SEARCH_DEFAULT else "dense"
    use_hybrid = retrieval_method == "hybrid"

    if vectorstore is None:

        return {

            "error":
            "No documents indexed"
        }

    # Answer cache: a repeated question (same text, rerank flag and
    # retrieval method) against an unchanged index gets the stored
    # answer with no retrieval, rerank or LLM call, so no Groq tokens
    # are spent. The cache is cleared whenever the index changes
    # (upload of new/edited content, reset), so a hit always matches
    # the current documents.
    cache_key = answer_cache.cache_key(question, rerank, retrieval_method)
    cached = answer_cache.cache.get(cache_key)
    if cached is not None:
        return {

            "question": question,

            "answer": cached["answer"],

            "sources": cached["sources"],

            "metrics": {

                "faiss_ms": 0.0,

                "bm25_ms": 0.0,

                "reranking_ms": 0.0,

                "generation_ms": 0.0,

                "total_ms": round(

                    (time.perf_counter() - t0) * 1000,

                    1
                ),

                "stage1_candidates": cached["stage1_candidates"],

                "final_sources": cached["final_sources"],

                "retrieval_method": cached["retrieval_method"],

                "reranked": cached["reranked"],

                "rerank_fallback": cached["rerank_fallback"],

                "refused": cached["refused"],

                "refusal_reason": cached["refusal_reason"],

                "injection_flagged": cached["injection_flagged"],

                "cached": True,
            },
        }

    # --- Stage 1: retrieval -------------------------------------------
    scored_candidates = vectorstore.similarity_search_with_score(

        question,

        k=10
    )

    # FAISS-only order. Scores are L2 distances (lower = closer),
    # negated so that, as with reranker scores, higher still means
    # more relevant.
    dense_ranked = [
        (float(-score), doc)
        for doc, score in scored_candidates
    ]
    dense_keys = [retrieval.chunk_key(doc) for _, doc in dense_ranked]
    t_dense = time.perf_counter()

    bm25_ms = 0.0
    if use_hybrid and bm25_index.n > 0:
        bm25_hits = bm25_index.search(question, k=10)
        bm25_keys = [
            retrieval.chunk_key(bm25_index.docs[i]) for i, _ in bm25_hits
        ]
        bm25_ms = (time.perf_counter() - t_dense) * 1000
        fused = retrieval.rrf_fuse([dense_keys, bm25_keys])
        candidates = []
        for key, rrf_score in fused[:10]:
            doc = chunk_registry.get(key)
            if doc is not None:
                candidates.append((rrf_score, doc))
        # If the registry lost a fused key, fall back to dense order
        # rather than returning fewer candidates than asked for.
        if len(candidates) < len(dense_ranked):
            seen = {retrieval.chunk_key(d) for _, d in candidates}
            candidates.extend(
                (s, d) for s, d in dense_ranked
                if retrieval.chunk_key(d) not in seen
            )
            candidates = candidates[:10]
    else:
        candidates = dense_ranked

    t1 = time.perf_counter()

    # --- Stage 2: rerank (with fallback + diversity) ------------------
    rerank_fallback = False
    if rerank:
        ranked_docs, rerank_fallback = retrieval.rerank_with_fallback(
            question, candidates, reranker
        )
        if rerank_fallback:
            logger.warning(
                "reranker failed for question %r; using retrieval order",
                question[:80],
            )
    else:
        ranked_docs = list(candidates)

    # Diversity: drop near-duplicate chunks from the top 3 (e.g. the
    # Q6 case, where adjacent chunks share a phrase). Falls back to
    # rank order if dedup would leave fewer than 3 sources.
    diversified = retrieval.diversify(ranked_docs, top_n=3)

    top_docs = [

        doc

        for score, doc in diversified
    ]

    t2 = time.perf_counter()

    context = "\n\n".join([

        doc.page_content

        for doc in top_docs
    ])

    # Groundedness guardrail: if the best reranked chunk scores below
    # the relevance threshold, nothing retrieved is actually about the
    # question — refuse instead of letting the LLM guess. Only applies
    # when the reranker actually ran (not on fallback scores, which
    # are RRF/L2 values on a different scale).
    refused = False
    refusal_reason = None
    if rerank and not rerank_fallback and ranked_docs and not guardrails.is_relevant(
        ranked_docs[0][0], reranked=True
    ):
        refused = True
        refusal_reason = "low_relevance"
        answer = f"The answer is {guardrails.REFUSAL_PHRASE}."
    else:
        answer = generate_answer(

            question,

            context,

            groq_api_key
        )

    # The post-generation check lives inside generate_answer; re-check
    # here so the response can report whether an answer was flagged.
    _cleaned, injection_flagged = guardrails.sanitize_answer(answer)
    if injection_flagged:
        answer = _cleaned

    t3 = time.perf_counter()

    sources = []

    for score, doc in diversified:

        sources.append({

            "source_file": doc.metadata.get(
                "source",
                "unknown"
            ),

            "chunk_id": doc.metadata.get(
                "chunk_id",
                "N/A"
            ),

            "score": round(
                float(score),
                2
            ),

            "content": (

                doc.page_content

                .replace("\n", " ")

                .strip()[:350]

                + (

                    "..."

                    if len(doc.page_content) > 350

                    else ""
                )
            )
        })

    response = {

        "question": question,

        "answer": answer,

        "sources": sources,

        "metrics": {

            "faiss_ms": round(

                (t_dense - t0) * 1000,

                1
            ),

            "bm25_ms": round(bm25_ms, 1),

            "reranking_ms": round(

                (t2 - t1) * 1000,

                1
            ),

            "generation_ms": round(

                (t3 - t2) * 1000,

                1
            ),

            "total_ms": round(

                (t3 - t0) * 1000,

                1
            ),

            "stage1_candidates": len(
                candidates
            ),

            "final_sources": len(
                top_docs
            ),

            "retrieval_method": retrieval_method,

            "reranked": bool(rerank),

            "rerank_fallback": bool(rerank_fallback),

            # Guardrail outcomes: refused=True means the answer is a
            # refusal (low relevance — nothing retrieved was actually
            # about the question). injection_flagged=True means the
            # generated answer repeated a known injection payload and
            # was replaced with a refusal by guardrails.sanitize_answer.
            "refused": bool(refused),

            "refusal_reason": refusal_reason,

            "injection_flagged": bool(injection_flagged),

            "cached": False,
        }
    }

    # Store only real generated answers. Refusals cost no LLM tokens
    # (generation 0.0 ms), and a flagged answer is a refusal, so there
    # is nothing to save by caching them.
    if not refused and not injection_flagged:
        answer_cache.cache.put(cache_key, {
            "answer": answer,
            "sources": response["sources"],
            "refused": bool(refused),
            "refusal_reason": refusal_reason,
            "injection_flagged": bool(injection_flagged),
            "retrieval_method": retrieval_method,
            "reranked": bool(rerank),
            "rerank_fallback": bool(rerank_fallback),
            "stage1_candidates": len(candidates),
            "final_sources": len(top_docs),
        })

    return response


@app.get("/api/health")
def health():

    total_chunks = 0

    if vectorstore:

        total_chunks = vectorstore.index.ntotal

    return {

        "status": "healthy",

        "indexed_chunks": total_chunks,

        "indexed_files": len(manifest),

        "retrieval_method": (
            "hybrid" if HYBRID_SEARCH_DEFAULT else "dense"
        ),
    }


@app.get("/api/documents")
def list_documents():

    files = []

    for file in os.listdir(UPLOAD_DIR):

        files.append(file)

    total_chunks = 0

    if vectorstore:

        total_chunks = vectorstore.index.ntotal

    return {

        "indexed_files": files,

        "total_files": len(files),

        "total_chunks": total_chunks,

        "per_file_chunks": {
            fn: entry.get("chunks", 0) for fn, entry in manifest.items()
        },
    }


@app.delete("/api/reset")
def reset_index():

    global vectorstore

    vectorstore = None

    # A reset is also a fresh start for the rate limiter (useful for
    # the eval, which resets before every run).
    query_limiter.reset()

    # The index is gone, so no cached answer can still be valid.
    answer_cache.cache.clear()

    # Clear the registry, manifest and BM25 index too — otherwise a
    # reset index would still claim files are cached.
    chunk_registry.clear()
    manifest.clear()
    _rebuild_bm25()

    if os.path.exists(VECTOR_STORE_PATH):

        shutil.rmtree(VECTOR_STORE_PATH)

    os.makedirs(
        VECTOR_STORE_PATH,
        exist_ok=True
    )

    for file in os.listdir(UPLOAD_DIR):

        file_path = os.path.join(
            UPLOAD_DIR,
            file
        )

        if os.path.isfile(file_path):

            os.remove(file_path)

    return {

        "message":
        "Index and uploaded files cleared successfully."
    }
