"""Pure text helpers for the Document QA System.

Kept in a separate module (with no embedding / reranker / LLM imports)
so they can be unit-tested without loading any ML model. app.py imports
from here.
"""

import hashlib

import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text):

    text = text.replace("\n", " ")

    text = " ".join(text.split())

    return text


def extract_text(file_path):

    if file_path.endswith(".pdf"):

        # Lazy import: PyPDFLoader pulls in langchain_community, which is
        # only needed when a PDF is actually processed.
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file_path)

        docs = loader.load()

        text = "\n".join(

            [doc.page_content for doc in docs]
        )

        return clean_text(text)

    elif file_path.endswith(".txt"):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            return clean_text(f.read())

    elif (
        file_path.endswith(".docx")
        or file_path.endswith(".doc")
    ):

        doc = docx.Document(file_path)

        text = "\n".join(

            [para.text for para in doc.paragraphs]
        )

        return clean_text(text)

    else:

        raise ValueError(
            "Unsupported file type."
        )


def get_file_hash(file_path):

    with open(file_path, "rb") as f:

        return hashlib.sha256(
            f.read()
        ).hexdigest()


def split_text(text, filename):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=50
    )

    chunks = splitter.create_documents([text])

    for i, chunk in enumerate(chunks):

        chunk.metadata = {

            "source": filename,

            "chunk_id": i
        }

    return chunks
