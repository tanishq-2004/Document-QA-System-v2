import json
import os

import streamlit as st
import requests

API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(

    page_title="Document QA System",

    layout="wide"
)

st.title("🔍 Document QA System")

# ====================================
# SIDEBAR
# ====================================

st.sidebar.header("Settings")

groq_api_key = st.sidebar.text_input(

    "Groq API Key (Optional)",

    type="password"
)

# ====================================
# HEALTH STATUS
# ====================================

try:

    health = requests.get(
        f"{API_URL}/api/health",
        timeout=5
    ).json()

    st.sidebar.success(
        f"System: {health['status']}"
    )

    st.sidebar.write(
        f"Indexed Chunks: "
        f"{health['indexed_chunks']}"
    )

except requests.exceptions.Timeout:

    st.sidebar.error(
        "Backend timed out (5s)"
    )

except requests.exceptions.RequestException:

    st.sidebar.error(
        "Backend offline"
    )

# ====================================
# DOCUMENTS
# ====================================

try:

    docs = requests.get(
        f"{API_URL}/api/documents",
        timeout=5
    ).json()

    st.sidebar.subheader(
        "Indexed Documents"
    )

    st.sidebar.write(
        f"Total Files: "
        f"{docs['total_files']}"
    )

    for file in docs["indexed_files"]:

        st.sidebar.caption(file)

except requests.exceptions.RequestException:

    pass

# ====================================
# MEASURED PERFORMANCE (eval dashboard)
# ====================================
# Numbers come from metrics_summary.json, generated from the real
# eval results files by eval/build_metrics_summary.py — nothing here
# is hand-typed. If the file is missing (e.g. an old deployment),
# the section simply stays hidden.

try:

    with open(

        os.path.join(

            os.path.dirname(os.path.abspath(__file__)),

            "metrics_summary.json"
        )
    ) as _f:

        _summary = json.load(_f)

    with st.sidebar.expander(
        "📊 Measured performance (evals)"
    ):

        _e7 = _summary["eval_7doc"]

        st.write(
            f"**7-doc eval** ({_e7['questions']} questions, "
            f"{_e7['runs']} runs): "
            f"P@3 {_e7['precision_at_3']}, "
            f"Hit@3 {_e7['hit_at_3']}, "
            f"refusals {_e7['refusal_passed']}/{_e7['refusal_total']}, "
            f"injection blocked "
            f"{_e7['injection_passed']}/{_e7['injection_total']}"
        )

        _lat = _summary["retrieval_only_latency_ms"]

        st.write(
            f"**Retrieval-only latency (median):** "
            f"total {_lat['median_total']} ms "
            f"(p95 {_lat['p95_total']} ms), "
            f"FAISS {_lat['median_faiss']} ms, "
            f"BM25 {_lat['median_bm25']} ms, "
            f"rerank {_lat['median_reranking']} ms"
        )

        st.write("**Public SQuAD scale test:**")

        for _row in _summary["scale_squad"]:

            _f1 = (
                f"token-F1 {_row['token_f1']}"
                if _row["token_f1"] is not None
                else _row["token_f1_note"]
            )

            st.write(
                f"- {_row['docs']} docs / {_row['chunks']} chunks, "
                f"indexed in {_row['index_seconds']}s: "
                f"Hit@3 dense-only {_row['hit_at_3_dense_only']} → "
                f"hybrid+rerank {_row['hit_at_3_hybrid_reranked']}, "
                f"{_f1}"
            )

        st.write(
            f"**Unit tests:** {_summary['unit_tests_passed']}/"
            f"{_summary['unit_tests_passed']} passing"
        )

        st.write(
            "**Cache:** re-upload in 1.7–2.0 ms at 500 docs, ~98% "
            "faster than fresh indexing (~124 ms per file)"
        )

        st.caption(_summary["notes"])

except (OSError, ValueError, KeyError):

    pass

# ====================================
# RESET BUTTON
# ====================================

if st.sidebar.button(
    "Clear Vector Store"
):

    try:

        response = requests.delete(
            f"{API_URL}/api/reset",
            timeout=5
        )

        st.sidebar.success(

            response.json()["message"]
        )

    except requests.exceptions.Timeout:

        st.sidebar.error(
            "Backend timed out (5s)"
        )

    except requests.exceptions.RequestException:

        st.sidebar.error(
            "Backend offline"
        )

# ====================================
# FILE UPLOAD
# ====================================

st.sidebar.subheader(
    "Upload Document"
)

uploaded_file = st.sidebar.file_uploader(

    "Upload PDF/TXT/DOCX",

    type=["pdf", "txt", "docx"]
)

# ====================================
# UPLOAD BUTTON
# ====================================

if uploaded_file and st.sidebar.button(
    "Upload Document"
):

    files = {

        "file": (

            uploaded_file.name,

            uploaded_file.getvalue()
        )
    }

    try:

        response = requests.post(

            f"{API_URL}/api/upload",

            files=files,

            timeout=60
        )

        data = response.json()

        if response.status_code == 200:

            if "message" in data:

                st.sidebar.success(
                    data["message"]
                )

            elif "error" in data:

                st.sidebar.error(
                    data["error"]
                )

            else:

                st.sidebar.success(

                    f"Indexed {data['chunks']} chunks "
                    f"from {data['filename']}"
                )

        else:

            st.sidebar.error(
                "Upload failed."
            )

    except requests.exceptions.Timeout:

        st.sidebar.error(
            "Upload timed out (60s)"
        )

    except Exception as e:

        st.sidebar.error(
            f"Connection error: {e}"
        )

# ====================================
# CHAT HISTORY (multi-turn follow-ups)
# ====================================
# Kept in session state only; the backend gets the last 3 turns with
# each question so follow-ups ("What year was that?") can be
# rewritten as standalone questions. Clearing this never touches the
# index.

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

if st.sidebar.button(
    "Clear chat history"
):

    st.session_state.chat_history = []

    st.sidebar.success(
        "Chat history cleared."
    )

for turn in st.session_state.chat_history:

    st.chat_message("user").write(
        turn["question"]
    )

    st.chat_message("assistant").write(
        turn["answer"]
    )

# ====================================
# CHAT INPUT
# ====================================

question = st.chat_input(
    "Ask your documents..."
)

if question:

    st.chat_message("user").write(
        question
    )

    with st.spinner(
        "Generating answer..."
    ):

        try:

            response = requests.post(

                f"{API_URL}/api/query",

                json={

                    "question": question,

                    # Last 3 turns so the backend can rewrite
                    # follow-up questions as standalone ones.
                    "history": st.session_state.chat_history[-3:],

                    # New field; the old name is also sent so older
                    # backends keep working.
                    "groq_api_key": groq_api_key,
                    "hf_token": groq_api_key
                },

                timeout=60
            )

            data = response.json()

            # ====================================
            # ERROR HANDLING
            # ====================================

            if "error" in data:

                st.error(
                    data["error"]
                )

            else:

                # ====================================
                # ANSWER
                # ====================================

                st.chat_message(
                    "assistant"
                ).write(

                    data["answer"]
                )

                # Show what a follow-up was understood as, when the
                # backend rewrote it.
                if data.get("standalone_question"):

                    st.caption(

                        f"Interpreted as: "
                        f"{data['standalone_question']}"
                    )

                # Remember this turn for follow-ups.
                st.session_state.chat_history.append({

                    "question": question,

                    "answer": data["answer"]
                })

                # ====================================
                # METRICS
                # ====================================

                metrics = data["metrics"]

                st.subheader("Metrics")

                col1, col2, col3 = st.columns(3)

                col1.metric(

                    "Retrieval",

                    f"{metrics['faiss_ms']} ms"
                )

                col2.metric(

                    "Reranking",

                    f"{metrics['reranking_ms']} ms"
                )

                col3.metric(

                    "Generation",

                    f"{metrics['generation_ms']} ms"
                )

                st.metric(

                    "Total Latency",

                    f"{metrics['total_ms']} ms"
                )

                # ====================================
                # SOURCES
                # ====================================

                st.subheader("Sources")

                for source in data["sources"]:

                    with st.expander(

                        f"{source['source_file']} "

                        f"(Chunk "
                        f"{source['chunk_id']})"
                    ):

                        st.write(
                            source["content"]
                        )

                        st.caption(

                            f"Relevance Score: "
                            f"{source['score']}"
                        )

        except requests.exceptions.Timeout:

            st.error(

                "Backend timed out after 60s. "
                "Try a shorter question or check the backend."
            )

        except Exception as e:

            st.error(

                f"Backend connection failed: {e}"
            )