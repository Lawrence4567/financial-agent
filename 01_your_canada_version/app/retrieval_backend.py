from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from rag_pipeline import retrieve_reference_context


class RetrievedChunk(BaseModel):
    id: str
    title: str
    section: str
    source_file: str
    score: float
    text: str
    snippet: str


class RetrievalResult(BaseModel):
    backend: str
    chunks: List[RetrievedChunk] = Field(default_factory=list)
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    query_used: str
    retrieval_mode: Optional[str] = None
    status: str = "ready"
    index_status: Optional[Dict[str, Any]] = None


class RetrieverBackend(ABC):
    backend_name: str

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4, filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        raise NotImplementedError


class LocalIndexRetriever(RetrieverBackend):
    backend_name = "local_index"

    def __init__(self, fallback_note: Optional[str] = None) -> None:
        self.fallback_note = fallback_note

    def retrieve(self, query: str, top_k: int = 4, filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        raw_result = retrieve_reference_context(query, top_k=top_k)
        applied_filters = dict(filters or {})
        if self.fallback_note:
            applied_filters["_backend_fallback"] = self.fallback_note
        return RetrievalResult(
            backend=self.backend_name,
            chunks=[RetrievedChunk(**chunk) for chunk in raw_result.get("chunks", [])],
            filters_applied=applied_filters,
            query_used=query,
            retrieval_mode=raw_result.get("retrieval_mode"),
            status=raw_result.get("status", "ready"),
            index_status=raw_result.get("index_status"),
        )


class VectorStoreRetriever(RetrieverBackend):
    backend_name = "vector_store"

    def retrieve(self, query: str, top_k: int = 4, filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        raise NotImplementedError("VectorStoreRetriever is a placeholder in this phase.")


def get_retriever_backend() -> RetrieverBackend:
    backend = os.getenv("RETRIEVAL_BACKEND", "local_index").strip().lower()
    if backend == "local_index":
        return LocalIndexRetriever()
    if backend == "vector_store":
        return LocalIndexRetriever(fallback_note="vector_store_placeholder_not_implemented")
    return LocalIndexRetriever(fallback_note=f"unknown_backend_{backend}_fell_back_to_local_index")
