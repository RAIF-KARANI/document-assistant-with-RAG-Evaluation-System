"""
Ingestion pipeline: load documents from data/, split them into chunks,
embed the chunks, and persist them into a local Chroma vector store.

Run this whenever you add/change documents in data/, or change chunking
settings in config.py:

    python -m src.ingest
"""

import re
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

LOADERS_BY_SUFFIX = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
}

# A table-of-contents / list-of-figures / list-of-tables page is dense with
# lines ending in a bare page number ("9.2 Limitations ... 43") and carries
# no informational content of its own - just navigation. Indexing it is
# actively harmful, not just wasteful: a section title repeated in a TOC
# line can lexically out-rank that section's own real content for a
# question about the same topic, because the TOC line matches the
# question's wording more closely than the substantive (but differently
# worded) content does. Confirmed happening in this project - see
# CONCEPTS.md ("9.2 Limitations" vs. the adjacent "9.3" TOC fragment).
_TOC_LINE_PATTERN = re.compile(r"\d{1,4}\s*$")


def is_toc_like(text: str, threshold: float = 0.5, min_lines: int = 4) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < min_lines:
        return False
    numbered = sum(1 for line in lines if _TOC_LINE_PATTERN.search(line))
    return (numbered / len(lines)) >= threshold


# --- Stage 1: Document loading ---
# Reads raw .pdf/.txt files off disk into LangChain Document objects
# (page_content = text, metadata = source file path). No chunking yet.
# Filters out TOC-like pages here, before they're ever chunked or embedded -
# cheaper and more effective than trying to filter chunks after the fact,
# since a TOC page's harmful lexical density survives chunking intact.
def load_documents(data_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        loader_cls = LOADERS_BY_SUFFIX.get(path.suffix.lower())
        if loader_cls is None:
            continue
        loader = loader_cls(str(path))
        for doc in loader.load():
            if is_toc_like(doc.page_content):
                continue
            documents.append(doc)
    return documents


# --- Stage 2: Chunking ---
# Splits each document into smaller pieces (CHUNK_SIZE chars, CHUNK_OVERLAP
# overlap) so each chunk is small/focused enough to embed and retrieve well.
# Params default to config.py so the main pipeline (src.ingest main()) is
# unaffected; overrides exist so experiments/run_experiment.py (Phase 4) can
# try other chunk sizes without touching this file's defaults.
def chunk_documents(
    documents: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


# --- Stage 3: Embedding + indexing ---
# Turns each chunk's text into a vector (via the local sentence-transformers
# model) and writes vector + text + metadata into the Chroma vector store.
# collection_name/persist_directory default to the main assistant's store;
# experiments pass their own isolated values so they never overwrite it.
def build_vector_store(
    chunks: list[Document],
    embedding_model: str = config.EMBEDDING_MODEL,
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path = config.CHROMA_DIR,
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )


# Drops a previously indexed collection, if one exists, so a re-ingest
# replaces it cleanly instead of silently duplicating every chunk on top
# of the old ones (Chroma.from_documents adds to an existing collection of
# the same name rather than replacing it). Uses Chroma's own collection API
# rather than deleting files on disk, since a client from earlier in the
# same process - e.g. src.retrieve's cached _vector_store - may already
# hold an open connection to that persist_directory.
def drop_existing_collection(
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path = config.CHROMA_DIR,
) -> None:
    if not persist_directory.exists():
        return
    client = chromadb.PersistentClient(path=str(persist_directory))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # collection didn't exist yet under this name - nothing to drop


# Full rebuild used by both `python -m src.ingest` and the Streamlit
# sidebar's "add a document" flow (app.py) - drop whatever's indexed now,
# then load + chunk + embed everything currently in data_dir from scratch.
# Returns (num_documents, num_chunks) so callers can report progress.
def rebuild_index(
    data_dir: Path = config.DATA_DIR,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    embedding_model: str = config.EMBEDDING_MODEL,
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: Path = config.CHROMA_DIR,
) -> tuple[int, int]:
    drop_existing_collection(collection_name, persist_directory)
    documents = load_documents(data_dir)
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    if chunks:
        build_vector_store(chunks, embedding_model, collection_name, persist_directory)
    return len(documents), len(chunks)


def main() -> None:
    print(f"Loading documents from {config.DATA_DIR} ...")
    print(f"Rebuilding index at {config.CHROMA_DIR} ...")
    num_documents, num_chunks = rebuild_index()
    if not num_documents:
        print(
            "No .pdf or .txt files found in data/. "
            "Add some documents (see data/README.md) and re-run."
        )
        return
    print(
        f"Loaded {num_documents} document(s), split into {num_chunks} chunk(s) "
        f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})."
    )
    print("Done. Vector store is ready — run `python -m src.ask \"your question\"`.")


if __name__ == "__main__":
    main()
