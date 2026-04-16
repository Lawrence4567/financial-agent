from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency in beginner mode
    OpenAI = None

from data_sources import (
    PROJECT_DIR,
    REFERENCE_DIR,
    load_market_commentary,
    load_market_context,
    load_reference_knowledge,
)


INDEX_DIR = PROJECT_DIR / "data" / "artifacts_canada"
INDEX_FILE = INDEX_DIR / "reference_rag_index.json"
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split())


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _lexical_score(query: str, text: str) -> float:
    query_tokens = _tokenize(query)
    text_tokens = _tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0

    text_token_set = set(text_tokens)
    overlap = sum(1 for token in query_tokens if token in text_token_set)
    if overlap == 0:
        return 0.0

    return overlap / math.sqrt(len(query_tokens) * len(text_token_set))


def _entity_bonus(query: str, chunk: dict[str, Any]) -> float:
    tracked_terms = [
        "fhsa",
        "tfsa",
        "rrsp",
        "gic",
        "high-interest savings",
        "high interest savings",
        "etf",
    ]
    query_lower = query.lower()
    text = f"{chunk.get('title', '')} {chunk.get('text', '')}".lower()
    bonus = 0.0

    for term in tracked_terms:
        if term in query_lower and term in text:
            bonus += 0.035

    if any(term in query_lower for term in ["difference", "what is", "how does", "explain"]) and chunk.get("section") == "account_knowledge":
        bonus += 0.08
    if "difference" in query_lower and chunk.get("section") == "goal_priority_rules":
        bonus -= 0.05

    return bonus


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return 0.0

    dot_product = sum(left * right for left, right in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _build_account_chunks(reference_knowledge: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    accounts = reference_knowledge.get("account_knowledge", {}).get("accounts", [])

    for account in accounts:
        category = account["category"]
        text = _normalize_text(
            "\n".join(
                [
                    f"Account: {category}",
                    f"Purpose: {account.get('purpose', '')}",
                    "Strengths:",
                    *[f"- {item}" for item in account.get("strengths", [])],
                    "Watchouts:",
                    *[f"- {item}" for item in account.get("watchouts", [])],
                ]
            )
        )
        chunks.append(
            {
                "id": f"account::{_slugify(category)}",
                "title": f"{category} account overview",
                "section": "account_knowledge",
                "source_file": str(REFERENCE_DIR / "account_knowledge.json"),
                "text": text,
            }
        )

    return chunks


def _build_planning_chunks(reference_knowledge: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    planning_guidance = reference_knowledge.get("planning_guidance", {})

    for rule in planning_guidance.get("goal_priority_rules", []):
        rule_id = rule["rule_id"]
        text = _normalize_text(
            "\n".join(
                [
                    f"Rule: {rule_id}",
                    f"Matched goals: {', '.join(rule.get('match_keywords', []))}",
                    f"Priority order: {', '.join(rule.get('priority_order', []))}",
                    f"Reason: {rule.get('reason', '')}",
                ]
            )
        )
        chunks.append(
            {
                "id": f"planning_rule::{_slugify(rule_id)}",
                "title": f"Planning rule: {rule_id}",
                "section": "goal_priority_rules",
                "source_file": str(REFERENCE_DIR / "planning_guidance.json"),
                "text": text,
            }
        )

    for index, guideline in enumerate(planning_guidance.get("budgeting_guidelines", []), start=1):
        chunks.append(
            {
                "id": f"budgeting_guideline::{index}",
                "title": f"Budgeting guideline {index}",
                "section": "budgeting_guidelines",
                "source_file": str(REFERENCE_DIR / "planning_guidance.json"),
                "text": _normalize_text(f"Budgeting guideline: {guideline}"),
            }
        )

    for index, note in enumerate(planning_guidance.get("advisor_safety_notes", []), start=1):
        chunks.append(
            {
                "id": f"safety_note::{index}",
                "title": f"Advisor safety note {index}",
                "section": "advisor_safety_notes",
                "source_file": str(REFERENCE_DIR / "planning_guidance.json"),
                "text": _normalize_text(f"Safety note: {note}"),
            }
        )

    return chunks


def _build_official_rule_chunks(reference_knowledge: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    official_rules = reference_knowledge.get("official_account_rules", {}).get("rules", [])

    for rule in official_rules:
        text = _normalize_text(
            "\n".join(
                [
                    f"Rule title: {rule.get('title', '')}",
                    f"Category: {rule.get('category', '')}",
                    f"Source: {rule.get('source', '')}",
                    f"Summary: {rule.get('summary', '')}",
                    "Details:",
                    *[f"- {item}" for item in rule.get("details", [])],
                    f"Source URL: {rule.get('source_url', '')}",
                ]
            )
        )
        chunks.append(
            {
                "id": f"official_rule::{_slugify(rule['id'])}",
                "title": rule["title"],
                "section": "official_account_rules",
                "source_file": str(REFERENCE_DIR / "official_account_rules.json"),
                "text": text,
            }
        )

    return chunks


def _build_market_context_chunks() -> list[dict[str, Any]]:
    market_context = load_market_context()
    market_commentary = load_market_commentary()
    chunks: list[dict[str, Any]] = [
        {
            "id": "market_context::overview",
            "title": "Market context overview",
            "section": "market_context",
            "source_file": str(REFERENCE_DIR / "market_context.json"),
            "text": _normalize_text(
                "\n".join(
                    [
                        f"Status: {market_context.get('status', '')}",
                        f"Summary: {market_context.get('summary', '')}",
                        "Future sources:",
                        *[f"- {item}" for item in market_context.get("future_external_sources", [])],
                    ]
                )
            ),
        }
    ]

    for item in market_context.get("watchlist", market_context.get("starter_watchlist", [])):
        symbol = item.get("symbol", "market")
        chunks.append(
            {
                "id": f"market_watchlist::{_slugify(symbol)}",
                "title": f"Market watchlist {symbol}",
                "section": "market_watchlist",
                "source_file": str(REFERENCE_DIR / "market_context.json"),
                "text": _normalize_text(
                    "\n".join(
                        [
                            f"Symbol: {item.get('symbol', '')}",
                            f"Label: {item.get('label', '')}",
                            f"Why it matters: {item.get('why_it_matters', '')}",
                        ]
                    )
                ),
            }
        )

    current_market_story = market_commentary.get("current_market_story", {})
    chunks.append(
        {
            "id": "market_commentary::current_story",
            "title": "Current market story",
            "section": "market_commentary",
            "source_file": str(REFERENCE_DIR / "market_commentary.json"),
            "text": _normalize_text(
                "\n".join(
                    [
                        f"Headline: {current_market_story.get('headline', '')}",
                        f"Summary: {current_market_story.get('summary', '')}",
                        f"Advisor takeaway: {current_market_story.get('advisor_takeaway', '')}",
                    ]
                )
            ),
        }
    )

    for index, item in enumerate(market_commentary.get("current_themes", []), start=1):
        chunks.append(
            {
                "id": f"market_theme::{index}",
                "title": f"Market theme {index}: {item.get('theme', '')}",
                "section": "market_themes",
                "source_file": str(REFERENCE_DIR / "market_commentary.json"),
                "text": _normalize_text(
                    f"Theme: {item.get('theme', '')}\nImplication: {item.get('implication', '')}"
                ),
            }
        )

    for item in market_commentary.get("month_explanations", []):
        month = item.get("month", "month")
        chunks.append(
            {
                "id": f"market_month::{_slugify(month)}",
                "title": f"Market explanation {month}",
                "section": "market_month_explanations",
                "source_file": str(REFERENCE_DIR / "market_commentary.json"),
                "text": _normalize_text(
                    "\n".join(
                        [
                            f"Month: {month}",
                            f"Market context: {item.get('market_context', '')}",
                            f"Portfolio effect: {item.get('portfolio_effect', '')}",
                            f"Client message: {item.get('client_message', '')}",
                        ]
                    )
                ),
            }
        )

    return chunks


def build_reference_chunks() -> list[dict[str, Any]]:
    reference_knowledge = load_reference_knowledge()
    chunks = (
        _build_account_chunks(reference_knowledge)
        + _build_planning_chunks(reference_knowledge)
        + _build_official_rule_chunks(reference_knowledge)
        + _build_market_context_chunks()
    )

    for chunk in chunks:
        content_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
        chunk["content_hash"] = content_hash

    return chunks


def _get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _embed_texts(texts: list[str], model: str) -> list[list[float]]:
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is missing, so embeddings cannot be generated.")

    vectors: list[list[float]] = []
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def build_reference_rag_index(force_rebuild: bool = False) -> dict[str, Any]:
    if INDEX_FILE.exists() and not force_rebuild:
        return load_reference_rag_index()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = build_reference_chunks()
    embedding_model = DEFAULT_EMBEDDING_MODEL
    provider = "openai" if _get_openai_client() is not None else "lexical_fallback"

    if provider == "openai":
        vectors = _embed_texts([chunk["text"] for chunk in chunks], embedding_model)
        for chunk, vector in zip(chunks, vectors):
            chunk["vector"] = vector
    else:
        for chunk in chunks:
            chunk["vector"] = None

    index_payload = {
        "index_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_provider": provider,
        "embedding_model": embedding_model if provider == "openai" else None,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    INDEX_FILE.write_text(json.dumps(index_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return index_payload


def load_reference_rag_index() -> dict[str, Any]:
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def ensure_reference_rag_index() -> dict[str, Any]:
    if INDEX_FILE.exists():
        index_payload = load_reference_rag_index()
        if index_payload.get("index_version", 1) >= 2:
            return index_payload
    return build_reference_rag_index(force_rebuild=True)


def get_rag_index_status() -> dict[str, Any]:
    if INDEX_FILE.exists():
        index_payload = load_reference_rag_index()
        return {
            "status": "ready",
            "index_file": str(INDEX_FILE),
            "chunk_count": index_payload.get("chunk_count", 0),
            "embedding_provider": index_payload.get("embedding_provider"),
            "embedding_model": index_payload.get("embedding_model"),
            "created_at": index_payload.get("created_at"),
        }

    return {
        "status": "not_built",
        "index_file": str(INDEX_FILE),
        "chunk_count": 0,
        "embedding_provider": "openai" if _get_openai_client() is not None else "lexical_fallback",
        "embedding_model": DEFAULT_EMBEDDING_MODEL if _get_openai_client() is not None else None,
        "created_at": None,
    }


def retrieve_reference_context(query: str, top_k: int = 4) -> dict[str, Any]:
    index_payload = ensure_reference_rag_index()
    chunks = index_payload.get("chunks", [])
    if not chunks:
        return {
            "status": "empty",
            "retrieval_mode": "none",
            "chunks": [],
            "index_status": get_rag_index_status(),
        }

    retrieval_mode = index_payload.get("embedding_provider", "lexical_fallback")
    scored_chunks: list[dict[str, Any]] = []
    can_embed = retrieval_mode == "openai" and _get_openai_client() is not None

    if can_embed:
        query_vector = _embed_texts([query], index_payload.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)[0]
        for chunk in chunks:
            semantic_score = _cosine_similarity(query_vector, chunk.get("vector") or [])
            lexical_score = _lexical_score(query, chunk.get("text", ""))
            score = (semantic_score * 0.82) + (lexical_score * 0.15) + _entity_bonus(query, chunk)
            scored_chunks.append({**chunk, "score": round(float(score), 4)})
    else:
        retrieval_mode = "lexical_fallback"
        for chunk in chunks:
            score = _lexical_score(query, chunk.get("text", ""))
            scored_chunks.append({**chunk, "score": round(float(score), 4)})

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    top_chunks = []
    for chunk in scored_chunks[:top_k]:
        top_chunks.append(
            {
                "id": chunk["id"],
                "title": chunk["title"],
                "section": chunk["section"],
                "source_file": chunk["source_file"],
                "score": chunk["score"],
                "text": chunk["text"],
                "snippet": chunk["text"][:260] + ("..." if len(chunk["text"]) > 260 else ""),
            }
        )

    return {
        "status": "ready",
        "retrieval_mode": retrieval_mode,
        "chunks": top_chunks,
        "index_status": get_rag_index_status(),
    }


if __name__ == "__main__":
    index_payload = build_reference_rag_index(force_rebuild=True)
    print(f"Built reference RAG index with {index_payload['chunk_count']} chunks.")
    print(f"Provider: {index_payload['embedding_provider']}")
    print(f"Index file: {INDEX_FILE}")
