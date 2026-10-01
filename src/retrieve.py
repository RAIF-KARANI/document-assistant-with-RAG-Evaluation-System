"""
Retrieval: load the persisted Chroma vector store and fetch the top-k
chunks most relevant to a query.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src import config

_vector_store: Chroma | None = None


# Opens the Chroma vector store built by src/ingest.py. Must use the same
# embedding model here as at ingest time, or query vectors and chunk vectors
# won't live in the same space and similarity search becomes meaningless.
def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        _vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(config.CHROMA_DIR),
        )
    return _vector_store


# After src.ingest.rebuild_index() rewrites chroma_db/ on disk (e.g. from
# the Streamlit "add a document" flow), the cached _vector_store above
# would otherwise keep pointing at the old, now-replaced collection for
# the rest of the process's lifetime. Call this right after a rebuild so
# the next retrieve() reopens a fresh client against the new data.
def reset_vector_store() -> None:
    global _vector_store
    _vector_store = None


# Distinct 'source' metadata values currently indexed - used to build the
# document-scope picker in app.py. Reads real indexed values rather than
# re-deriving paths from the filesystem, so a filter built from this list
# always matches exactly.
def list_sources() -> list[str]:
    try:
        store = get_vector_store()
        result = store.get(include=["metadatas"])
    except Exception:
        return []
    return sorted({m.get("source", "unknown") for m in result["metadatas"]})


# --- Retrieval (semantic search) ---
# Embeds the raw query (no chunking - a question is already one short unit
# of meaning) and returns the k chunks whose vectors are closest to it.
# `source`, when given, scopes the search to chunks from just that one
# document (an exact match against list_sources()' values) - fixes queries
# like "tell me about this project" being ambiguous across a multi-document
# corpus with no other way to say which document "this" refers to.
def retrieve(query: str, k: int = config.TOP_K, source: str | None = None) -> list[Document]:
    store = get_vector_store()
    if source:
        return store.similarity_search(query, k=k, filter={"source": source})
    return store.similarity_search(query, k=k)


# Uncached variant for experiments/run_experiment.py (Phase 4): points at an
# isolated collection/persist_directory instead of the main assistant's
# cached _vector_store, so trying a different chunk size or embedding model
# never touches the real data the main pipeline uses.
def retrieve_from(
    query: str,
    k: int,
    collection_name: str,
    persist_directory,
    embedding_model: str = config.EMBEDDING_MODEL,
) -> list[Document]:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    return store.similarity_search(query, k=k)


# Parent-child variant of retrieve_from (experiments/run_experiment.py
# --parent-child): the index holds small *child* chunks (see
# src/ingest.py's chunk_documents_parent_child), so similarity search
# matches precisely - but this returns each match's *parent* chunk (full
# text carried in the child's "parent_content" metadata) instead of the
# child itself, so the LLM gets more complete surrounding context than
# whatever small piece the query happened to match. Dedupes by parent_id
# since two matching children can share the same parent; k child matches
# can therefore return fewer than k parent documents.
def retrieve_parent_child(
    query: str,
    k: int,
    collection_name: str,
    persist_directory,
    embedding_model: str = config.EMBEDDING_MODEL,
) -> list[Document]:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    child_hits = store.similarity_search(query, k=k)

    seen_parent_ids = set()
    parent_docs: list[Document] = []
    for child in child_hits:
        parent_id = child.metadata.get("parent_id")
        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)
        parent_docs.append(
            Document(
                page_content=child.metadata.get("parent_content", child.page_content),
                metadata={"source": child.metadata.get("source", "unknown")},
            )
        )
    return parent_docs
