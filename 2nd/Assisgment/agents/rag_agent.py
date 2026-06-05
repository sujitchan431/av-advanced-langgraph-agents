import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from state import AgentState

_BASE_DIR = Path(__file__).resolve().parent.parent
_INDEX_PATH = str(_BASE_DIR / "data" / "faiss_index")
_KB_PATH = str(_BASE_DIR / "data" / "knowledge_base.json")

_vector_store = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _build_vector_store() -> FAISS:
    print("[RAG] Building vector index from knowledge base…")
    embeddings = _get_embeddings()

    with open(_KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    docs = [
        Document(
            page_content=item["content"],
            metadata={"title": item["title"], "category": item["category"]},
        )
        for item in kb
    ]

    vs = FAISS.from_documents(docs, embeddings)
    os.makedirs(os.path.dirname(_INDEX_PATH), exist_ok=True)
    vs.save_local(_INDEX_PATH)
    print(f"[RAG] Index saved ({len(docs)} documents)")
    return vs


def _load_vector_store() -> FAISS:
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    embeddings = _get_embeddings()
    if os.path.exists(_INDEX_PATH):
        print("[RAG] Loading existing FAISS index…")
        _vector_store = FAISS.load_local(
            _INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        _vector_store = _build_vector_store()

    return _vector_store


def rag_node(state: AgentState) -> AgentState:
    query = state["query"]
    print(f"\n[RAG] Retrieving for: '{query[:60]}'")

    try:
        vs = _load_vector_store()
        docs = vs.similarity_search(query, k=3)

        formatted = "\n\n---\n\n".join(
            f"[{doc.metadata.get('title', 'Unknown')} | {doc.metadata.get('category', '')}]\n{doc.page_content}"
            for doc in docs
        )
        sources = [doc.metadata.get("title", "Unknown") for doc in docs]

        print(f"[RAG] Retrieved {len(docs)} documents")
        return {**state, "rag_results": formatted, "sources": sources}

    except Exception as exc:
        print(f"[RAG] ERROR: {exc}")
        return {**state, "rag_results": f"RAG retrieval failed: {exc}", "sources": []}
