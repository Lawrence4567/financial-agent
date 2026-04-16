from __future__ import annotations

import math
import os
import re
from typing import Any

from conversation_memory import format_chat_history_as_text, get_last_assistant_route, looks_like_follow_up_query
from query_understanding import normalize_query_text

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency in beginner mode
    OpenAI = None


DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
ROUTE_LABEL_TO_NAME = {
    "Architecture rules": "architecture_rules",
    "Spending rules": "spending_rules",
    "Live market snapshot": "market_snapshot_rules",
    "Market explanation": "market_explanation",
    "Account summary": "account_summary_rules",
    "Portfolio explanation": "portfolio_explanation",
    "Performance explanation": "performance_explanation",
    "Risk profile": "risk_profile_rules",
    "Recommendation rules": "recommendation_rules",
    "RAG knowledge": "rag_knowledge",
    "Hybrid spending + recommendation": "hybrid_spending_recommendation",
    "Hybrid market + recommendation": "hybrid_market_advice",
    "Hybrid recommendation + rules": "hybrid_rule_advice",
    "Hybrid profile + RAG + market": "hybrid_profile_rag_market",
}
ROUTE_INTENT_EXAMPLES = {
    "architecture_rules": [
        "How does this app work?",
        "What data sources does this app use?",
        "Explain the architecture of the finance copilot.",
        "Where does the data come from in this project?",
    ],
    "spending_rules": [
        "How much money did I spend on food?",
        "Show my monthly spending trend.",
        "What are my spending patterns?",
        "How much did I pay for rent?",
        "What did I spend on groceries and dining?",
        "Help me understand my cash habits.",
    ],
    "market_snapshot_rules": [
        "Show me an ETF market snapshot.",
        "What are the latest ETF prices?",
        "Give me a quick market quote for ETFs.",
        "How are the tracked ETFs performing?",
    ],
    "market_explanation": [
        "Explain market changes.",
        "Why did the market move this month?",
        "What changed in the market and why does it matter?",
        "Explain the current market backdrop for this portfolio.",
        "Does the U.S. market impact my investments?",
    ],
    "account_summary_rules": [
        "Show my household account summary.",
        "What is my net worth right now?",
        "Show my balances and liquid assets.",
        "Give me an account overview.",
        "How many accounts do I have?",
        "How much cash do I currently have?",
        "How much do I have in each account?",
    ],
    "portfolio_explanation": [
        "Explain my current portfolio allocation.",
        "Walk me through my holdings and asset mix.",
        "How is this portfolio positioned?",
        "Generate a portfolio explanation for this client.",
        "Where is my money invested?",
    ],
    "performance_explanation": [
        "Why did my returns change this month?",
        "Explain this month's portfolio performance.",
        "Why was my portfolio return lower than last month?",
        "What drove the change in performance?",
        "Did I make or lose money this month?",
    ],
    "risk_profile_rules": [
        "What is a risk profile?",
        "What is my risk tolerance?",
        "Show my suitability snapshot.",
        "Explain this client's risk profile.",
    ],
    "recommendation_rules": [
        "Which account should I focus on first?",
        "Based on my profile, should I choose FHSA or TFSA?",
        "What product fits my goals best?",
        "What is the best first step for my goals?",
        "What product makes the most sense for me?",
        "What should I choose for my situation?",
    ],
    "rag_knowledge": [
        "What is the difference between FHSA and TFSA?",
        "Explain RRSP contribution rules.",
        "How do TFSA withdrawals work in Canada?",
        "What are the eligibility rules for an FHSA?",
    ],
    "hybrid_spending_recommendation": [
        "Given my spending habits, what should I prioritize?",
        "Based on my budget and cash flow, what account should I focus on?",
        "What should I do next using my spending pattern and goals?",
    ],
    "hybrid_market_advice": [
        "Which ETF fits me and what does the market snapshot show?",
        "Based on my profile, which ETF should I consider right now?",
        "Combine market performance with a recommendation for me.",
    ],
    "hybrid_rule_advice": [
        "Based on my profile and the rules, should I focus on FHSA, TFSA, or RRSP?",
        "Which account should I prioritize given my goals and the tax rules?",
        "Use my profile and the official rules to recommend an account.",
    ],
    "hybrid_profile_rag_market": [
        "Using my profile, the account rules, and the market snapshot, what should I do?",
        "Combine my goals, Canadian account rules, and ETF market context for a recommendation.",
        "What should I focus on after considering my profile, finance rules, and the market?",
    ],
}
_ROUTE_VECTOR_CACHE: dict[str, dict[str, list[float]]] = {}


def _get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _embed_texts(texts: list[str], model: str) -> list[list[float]]:
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is missing, so route embeddings cannot be generated.")
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _lexical_similarity(text_a: str, text_b: str) -> float:
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    set_b = set(tokens_b)
    overlap = sum(1 for token in tokens_a if token in set_b)
    if overlap == 0:
        return 0.0
    return overlap / math.sqrt(len(tokens_a) * len(set_b))


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return 0.0
    dot_product = sum(left * right for left, right in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _ensure_route_vectors(model: str) -> dict[str, list[float]]:
    if model in _ROUTE_VECTOR_CACHE:
        return _ROUTE_VECTOR_CACHE[model]

    flattened_examples: list[str] = []
    keys: list[str] = []
    for route_name, examples in ROUTE_INTENT_EXAMPLES.items():
        for index, example in enumerate(examples):
            keys.append(f"{route_name}::{index}")
            flattened_examples.append(example)

    vectors = _embed_texts(flattened_examples, model)
    cache = {
        key: vector
        for key, vector in zip(keys, vectors)
    }
    _ROUTE_VECTOR_CACHE[model] = cache
    return cache


def _build_routing_query(query: str, chat_history: list[dict[str, Any]] | None = None) -> str:
    query_text = normalize_query_text(query)
    history_text = format_chat_history_as_text(chat_history, max_messages=4, include_route=True, max_chars=800)
    if not history_text:
        return query_text
    return f"Recent conversation:\n{history_text}\n\nCurrent user question:\n{query_text}"


def semantic_route_query(query: str, chat_history: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    routing_query = _build_routing_query(query, chat_history=chat_history)
    client = _get_openai_client()
    provider = "openai" if client is not None else "lexical_fallback"
    best_match: dict[str, Any] | None = None

    if provider == "openai":
        model = DEFAULT_EMBEDDING_MODEL
        query_vector = _embed_texts([routing_query], model)[0]
        route_vectors = _ensure_route_vectors(model)
        for route_name, examples in ROUTE_INTENT_EXAMPLES.items():
            route_best_score = 0.0
            route_best_example = None
            for index, example in enumerate(examples):
                vector = route_vectors.get(f"{route_name}::{index}", [])
                score = _cosine_similarity(query_vector, vector)
                if score > route_best_score:
                    route_best_score = score
                    route_best_example = example
            candidate = {
                "route": route_name,
                "score": round(float(route_best_score), 4),
                "provider": provider,
                "matched_example": route_best_example,
            }
            if best_match is None or candidate["score"] > best_match["score"]:
                best_match = candidate
    else:
        normalized_query = normalize_query_text(routing_query)
        for route_name, examples in ROUTE_INTENT_EXAMPLES.items():
            route_best_score = 0.0
            route_best_example = None
            for example in examples:
                score = _lexical_similarity(normalized_query, normalize_query_text(example))
                if score > route_best_score:
                    route_best_score = score
                    route_best_example = example
            candidate = {
                "route": route_name,
                "score": round(float(route_best_score), 4),
                "provider": provider,
                "matched_example": route_best_example,
            }
            if best_match is None or candidate["score"] > best_match["score"]:
                best_match = candidate

    if best_match is None:
        return None

    last_route_label = get_last_assistant_route(chat_history)
    last_route = ROUTE_LABEL_TO_NAME.get(last_route_label or "", None)
    if last_route and last_route == best_match["route"] and looks_like_follow_up_query(query):
        best_match["score"] = round(float(best_match["score"] + 0.06), 4)
        best_match["history_boosted"] = True

    return best_match


def semantic_route_threshold(provider: str) -> float:
    return 0.43 if provider == "openai" else 0.22
