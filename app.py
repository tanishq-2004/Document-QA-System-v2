import os
import shutil
import pickle
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from sentence_transformers import CrossEncoder

from text_utils import extract_text, get_file_hash, split_text

import guardrails


UPLOAD_DIR = "uploaded_docs"
VECTOR_STORE_PATH = "vector_store"

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = None

if os.path.exists(f"{VECTOR_STORE_PATH}/index.faiss"):

    vectorstore = FAISS.load_local(

        VECTOR_STORE_PATH,

        embeddings,

        allow_dangerous_deserialization=True
    )

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# LLM settings are env-configurable; the defaults are the original values.
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "300"))
# Server-side Groq key (e.g. a Hugging Face Space secret). When set, the
# key does not have to be sent with every /api/query request.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_TIMEOUT_S = 30

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

    # Upload guardrails: allowed type + max 10 MB, checked BEFORE the
    # file is written, with a clean error (not a 500 from extract_text).
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

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as f:

        shutil.copyfileobj(
            file.file,
            f
        )

    file_hash = get_file_hash(file_path)

    cache_path = (
        f"{VECTOR_STORE_PATH}/{file_hash}.pkl"
    )

    # Cache hit only counts if the index is actually loaded and non-empty.
    # Otherwise a stale hash file would claim a document is indexed when
    # nothing is searchable (e.g. after a reset or a failed first index).
    if (
        os.path.exists(cache_path)
        and vectorstore is not None
        and getattr(vectorstore.index, "ntotal", 0) > 0
    ):

        return {

            "message":
            "Document already indexed"
        }

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

    save_vectorstore()

    with open(cache_path, "wb") as f:

        pickle.dump(

            {"indexed": True},

            f
        )

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

    # "rerank": false skips the cross-encoder and returns FAISS order, so
    # eval/eval.py can measure reranker on/off through the API alone.
    rerank = data.get("rerank", True)

    if vectorstore is None:

        return {

            "error":
            "No documents indexed"
        }

    scored_candidates = vectorstore.similarity_search_with_score(

        question,

        k=10
    )

    candidates = [doc for doc, _ in scored_candidates]

    t1 = time.perf_counter()

    if rerank:

        pairs = [

            (question, doc.page_content)

            for doc in candidates
        ]

        scores = reranker.predict(pairs)

        ranked_docs = sorted(

            zip(scores, candidates),

            key=lambda x: x[0],

            reverse=True
        )

    else:

        # FAISS-only order. Scores are L2 distances (lower = closer),
        # negated so that, as with reranker scores, higher still means
        # more relevant.
        ranked_docs = [
            (float(-score), doc)
            for doc, score in scored_candidates
        ]

    top_docs = [

        doc

        for score, doc in ranked_docs[:3]
    ]

    t2 = time.perf_counter()

    context = "\n\n".join([

        doc.page_content

        for doc in top_docs
    ])

    # Groundedness guardrail: if the best reranked chunk scores below
    # the relevance threshold, nothing retrieved is actually about the
    # question — refuse instead of letting the LLM guess. Only applies
    # to the reranked arm; FAISS-only scores are negated L2 distances
    # and are always negative (see guardrails.is_relevant).
    refused = False
    refusal_reason = None
    if rerank and ranked_docs and not guardrails.is_relevant(
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

    for score, doc in ranked_docs[:3]:

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

    return {

        "question": question,

        "answer": answer,

        "sources": sources,

        "metrics": {

            "faiss_ms": round(

                (t1 - t0) * 1000,

                1
            ),

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

            "reranked": bool(rerank),

            # Guardrail outcomes: refused=True means the answer is a
            # refusal (low relevance — nothing retrieved was actually
            # about the question). injection_flagged=True means the
            # generated answer repeated a known injection payload and
            # was replaced with a refusal by guardrails.sanitize_answer.
            "refused": bool(refused),

            "refusal_reason": refusal_reason,

            "injection_flagged": bool(injection_flagged)
        }
    }


@app.get("/api/health")
def health():

    total_chunks = 0

    if vectorstore:

        total_chunks = vectorstore.index.ntotal

    return {

        "status": "healthy",

        "indexed_chunks": total_chunks
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

        "total_chunks": total_chunks
    }


@app.delete("/api/reset")
def reset_index():

    global vectorstore

    vectorstore = None

    # A reset is also a fresh start for the rate limiter (useful for
    # the eval, which resets before every run).
    query_limiter.reset()

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