from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from query_understanding import is_semantic_spending_query, normalize_query_text
from semantic_router import semantic_route_query, semantic_route_threshold


@dataclass(frozen=True)
class QueryRouteDecision:
    route: str
    label: str
    reason: str
    uses_rag: bool = False
    prefers_llm: bool = False


SEMANTIC_ROUTE_METADATA = {
    "architecture_rules": {
        "label": "Architecture rules",
        "reason": "The semantic router matched the question to app-architecture examples.",
        "uses_rag": False,
    },
    "spending_rules": {
        "label": "Spending rules",
        "reason": "The semantic router matched the question to spending-analysis examples.",
        "uses_rag": False,
    },
    "market_snapshot_rules": {
        "label": "Live market snapshot",
        "reason": "The semantic router matched the question to market-snapshot examples.",
        "uses_rag": False,
    },
    "market_explanation": {
        "label": "Market explanation",
        "reason": "The semantic router matched the question to market-explanation examples.",
        "uses_rag": False,
    },
    "account_summary_rules": {
        "label": "Account summary",
        "reason": "The semantic router matched the question to account-summary examples.",
        "uses_rag": False,
    },
    "portfolio_explanation": {
        "label": "Portfolio explanation",
        "reason": "The semantic router matched the question to portfolio-positioning examples.",
        "uses_rag": False,
    },
    "performance_explanation": {
        "label": "Performance explanation",
        "reason": "The semantic router matched the question to monthly performance-analysis examples.",
        "uses_rag": True,
    },
    "risk_profile_rules": {
        "label": "Risk profile",
        "reason": "The semantic router matched the question to risk-profile and suitability examples.",
        "uses_rag": False,
    },
    "recommendation_rules": {
        "label": "Recommendation rules",
        "reason": "The semantic router matched the question to product-priority examples.",
        "uses_rag": False,
    },
    "rag_knowledge": {
        "label": "RAG knowledge",
        "reason": "The semantic router matched the question to Canadian account-rule and explanation examples.",
        "uses_rag": True,
    },
    "hybrid_spending_recommendation": {
        "label": "Hybrid spending + recommendation",
        "reason": "The semantic router matched the question to examples that combine spending with product guidance.",
        "uses_rag": False,
    },
    "hybrid_market_advice": {
        "label": "Hybrid market + recommendation",
        "reason": "The semantic router matched the question to examples that combine market context with product guidance.",
        "uses_rag": False,
    },
    "hybrid_rule_advice": {
        "label": "Hybrid recommendation + rules",
        "reason": "The semantic router matched the question to examples that combine profile guidance with account rules.",
        "uses_rag": True,
    },
    "hybrid_profile_rag_market": {
        "label": "Hybrid profile + RAG + market",
        "reason": "The semantic router matched the question to examples that combine profile, rules, and market context.",
        "uses_rag": True,
    },
}


def detect_safety_compliance_issue(query: str) -> dict[str, str] | None:
    query_lower = normalize_query_text(query)
    prompt_injection_terms = [
        "ignore previous instructions",
        "ignore prior instructions",
        "disregard previous instructions",
        "forget previous instructions",
        "override your instructions",
        "prompt injection",
        "reveal system prompt",
        "show system prompt",
        "developer instructions",
    ]
    insider_terms = [
        "insider tip",
        "insider tips",
        "inside information",
        "insider information",
        "material non public",
        "non public information",
        "mnpi",
    ]
    guaranteed_return_terms = [
        "guaranteed stock",
        "guaranteed return",
        "guaranteed profit",
        "risk free stock",
        "risk-free stock",
        "sure win stock",
        "cannot lose",
        "can't lose",
        "no risk return",
        "no-risk return",
    ]

    has_prompt_injection = any(term in query_lower for term in prompt_injection_terms)
    has_insider_request = any(term in query_lower for term in insider_terms)
    has_guaranteed_investment_claim = any(term in query_lower for term in guaranteed_return_terms) or (
        "stock" in query_lower and "guaranteed" in query_lower
    )

    if has_prompt_injection or has_insider_request:
        return {
            "kind": "prompt_injection_or_insider",
            "reason": "The question includes a prompt-override attempt or a request for insider information, so the app should refuse and switch to a safe explanation.",
        }

    if has_guaranteed_investment_claim:
        return {
            "kind": "guaranteed_investment_claim",
            "reason": "The question asks for a guaranteed investment outcome, so the app should avoid a misleading recommendation and explain safer alternatives instead.",
        }

    return None


def is_architecture_query(query: str) -> bool:
    query_lower = query.lower()
    architecture_keywords = [
        "architecture",
        "app structure",
        "how does this app work",
        "data source",
        "data sources",
        "data layer",
        "source layer",
        "market-data layer",
        "market data layer",
        "reference knowledge",
        "reference data",
        "rag",
        "retrieval",
        "vector",
        "product data",
        "user data",
        "where does the data come from",
        "what does the market-data layer do",
        "what does the market data layer do",
    ]
    return any(keyword in query_lower for keyword in architecture_keywords)


def is_recommendation_query(query: str) -> bool:
    query_lower = query.lower()
    recommendation_keywords = [
        "fit my goals",
        "focus on",
        "which should i choose",
        "which should i prioritize",
        "which is better for me",
        "best first step",
        "recommend",
        "recommendation",
        "avoid",
        "not recommend",
        "do not recommend",
        "don't recommend",
        "should i avoid",
        "should i not choose",
        "should i choose",
        "should i prioritize",
        "should i focus on",
    ]
    return any(keyword in query_lower for keyword in recommendation_keywords)


def is_market_snapshot_query(query: str) -> bool:
    query_lower = query.lower()
    if "yahoo finance" in query_lower:
        return True

    market_terms = ["market snapshot", "price", "prices", "quote", "quotes", "ticker", "tickers", "trend", "performance"]
    return "etf" in query_lower and any(term in query_lower for term in market_terms)


def is_market_explanation_query(query: str) -> bool:
    query_lower = normalize_query_text(query)
    explanation_terms = [
        "explain market",
        "market change",
        "market changes",
        "why market",
        "market moved",
        "market move",
        "what changed in the market",
        "market backdrop",
        "market context",
    ]
    impact_terms = [
        "impact my investments",
        "impact this portfolio",
        "affect my investments",
        "affect this portfolio",
        "mean for this portfolio",
        "matter for this portfolio",
    ]
    has_market_signal = any(term in query_lower for term in ["market", "u.s. market", "us market"])
    has_impact_signal = any(term in query_lower for term in impact_terms)
    return any(term in query_lower for term in explanation_terms) or (has_market_signal and has_impact_signal)


def is_spending_query(query: str) -> bool:
    query_lower = normalize_query_text(query)
    spending_keywords = [
        "monthly spending",
        "spending pattern",
        "spending patterns",
        "spending summary",
        "monthly trend",
        "monthly spending trend",
        "expense",
        "expenses",
        "spend",
        "spending",
    ]
    return any(keyword in query_lower for keyword in spending_keywords) or is_semantic_spending_query(query)


def is_account_summary_query(query: str) -> bool:
    query_lower = normalize_query_text(query)
    keywords = [
        "account summary",
        "account overview",
        "show my accounts",
        "show my balances",
        "net worth",
        "cash position",
        "household balance sheet",
        "how many accounts",
        "how much cash",
        "cash do i currently have",
        "available cash",
        "liquid assets",
        "how much do i have in each account",
        "each account",
    ]
    return any(keyword in query_lower for keyword in keywords) or (
        ("account" in query_lower or "accounts" in query_lower)
        and any(term in query_lower for term in ["how many", "count", "per account"])
    )


def is_portfolio_explanation_query(query: str) -> bool:
    query_lower = normalize_query_text(query)
    if is_market_explanation_query(query) or is_performance_explanation_query(query):
        return False
    keywords = [
        "portfolio",
        "holdings",
        "allocation",
        "asset mix",
        "portfolio explanation",
        "portfolio positioning",
        "where is my money invested",
        "where am i invested",
        "what am i invested in",
        "where is this money invested",
    ]
    return any(keyword in query_lower for keyword in keywords)


def is_performance_explanation_query(query: str) -> bool:
    if is_market_explanation_query(query):
        return False
    query_lower = normalize_query_text(query)
    direct_gain_loss_terms = [
        "make or lose money",
        "made or lost money",
        "made money",
        "lose money",
        "lost money",
        "gain or loss",
        "profit or loss",
        "up or down",
        "go down",
        "went down",
        "portfolio down",
        "decline",
        "declined",
    ]
    asks_direct_gain_loss = any(term in query_lower for term in direct_gain_loss_terms) and any(
        term in query_lower for term in ["month", "this month", "monthly"]
    )
    return asks_direct_gain_loss or (
        any(keyword in query_lower for keyword in ["return", "returns", "performance", "portfolio"])
        and any(keyword in query_lower for keyword in ["why", "change", "changed", "month", "moved", "move"])
    ) or (
        "portfolio" in query_lower
        and any(keyword in query_lower for keyword in ["down", "decline", "declined", "drop", "dropped"])
        and any(keyword in query_lower for keyword in ["month", "this month", "monthly"])
    )


def is_rag_knowledge_query(query: str) -> bool:
    query_lower = query.lower()
    topic_keywords = [
        "fhsa",
        "tfsa",
        "rrsp",
        "gic",
        "high-interest savings",
        "high interest savings",
        "etf",
        "registered account",
    ]
    knowledge_keywords = [
        "what is",
        "difference",
        "explain",
        "how does",
        "how do",
        "eligibility",
        "eligible",
        "contribution",
        "contributions",
        "withdrawal",
        "withdrawals",
        "tax",
        "deduction",
        "rule",
        "rules",
        "watchout",
        "purpose",
        "strength",
        "strengths",
    ]

    has_topic = any(keyword in query_lower for keyword in topic_keywords)
    has_knowledge_intent = any(keyword in query_lower for keyword in knowledge_keywords)
    return has_topic and has_knowledge_intent


def is_risk_profile_query(query: str) -> bool:
    query_lower = normalize_query_text(query)
    return any(
        term in query_lower
        for term in [
            "risk profile",
            "risk tolerance",
            "suitability",
            "what is a risk profile",
            "what is my risk profile",
        ]
    )


def is_fhsa_tfsa_comparison_query(query: str) -> bool:
    query_lower = query.lower()
    return "fhsa" in query_lower and "tfsa" in query_lower and any(
        term in query_lower for term in ["difference", "compare", "vs", "versus"]
    )


def is_hybrid_market_recommendation_query(query: str) -> bool:
    query_lower = query.lower()
    recommendation_terms = [
        "should i",
        "fit my goals",
        "focus on",
        "recommend",
        "priority",
        "for me",
        "my profile",
    ]
    market_terms = [
        "etf",
        "market",
        "snapshot",
        "price",
        "prices",
        "performance",
        "ticker",
    ]
    has_recommendation_signal = any(term in query_lower for term in recommendation_terms) or is_recommendation_query(query)
    has_market_signal = any(term in query_lower for term in market_terms)
    return has_recommendation_signal and has_market_signal


def is_hybrid_spending_recommendation_query(query: str) -> bool:
    query_lower = query.lower()
    spending_terms = [
        "spending",
        "expense",
        "expenses",
        "budget",
        "cash flow",
        "cashflow",
        "monthly",
        "spend",
    ]
    recommendation_terms = [
        "should i",
        "focus on",
        "prioritize",
        "priority",
        "recommend",
        "best for me",
        "fit my goals",
        "what should i do",
    ]
    has_spending_signal = any(term in query_lower for term in spending_terms)
    has_recommendation_signal = any(term in query_lower for term in recommendation_terms)
    return has_spending_signal and has_recommendation_signal


def is_hybrid_rule_recommendation_query(query: str) -> bool:
    query_lower = query.lower()
    profile_terms = [
        "my profile",
        "for me",
        "my goal",
        "my goals",
        "should i",
        "focus on",
        "prioritize",
        "priority",
        "best for me",
    ]
    rule_terms = [
        "rule",
        "rules",
        "eligibility",
        "eligible",
        "contribution",
        "contributions",
        "withdrawal",
        "withdrawals",
        "tax",
        "deduction",
        "qualify",
        "qualifying",
    ]
    account_terms = ["fhsa", "tfsa", "rrsp", "registered account"]

    has_profile_signal = any(term in query_lower for term in profile_terms) or is_recommendation_query(query)
    has_rule_signal = any(term in query_lower for term in rule_terms)
    has_account_signal = any(term in query_lower for term in account_terms)
    return has_profile_signal and has_rule_signal and has_account_signal


def is_hybrid_profile_rag_market_query(query: str) -> bool:
    query_lower = query.lower()
    profile_terms = [
        "my profile",
        "for me",
        "my goal",
        "my goals",
        "should i",
        "what should i do",
        "focus on",
        "prioritize",
    ]
    rag_terms = [
        "rule",
        "rules",
        "eligibility",
        "eligible",
        "contribution",
        "contributions",
        "withdrawal",
        "withdrawals",
        "tax",
        "difference",
        "compare",
    ]
    market_terms = [
        "market",
        "etf",
        "snapshot",
        "price",
        "prices",
        "performance",
        "ticker",
    ]
    has_profile_signal = any(term in query_lower for term in profile_terms) or is_recommendation_query(query)
    has_rag_signal = any(term in query_lower for term in rag_terms)
    has_market_signal = any(term in query_lower for term in market_terms)
    return has_profile_signal and has_rag_signal and has_market_signal


def _build_semantic_route_decision(route_name: str, use_llm: bool, semantic_match: dict[str, Any]) -> QueryRouteDecision:
    metadata = SEMANTIC_ROUTE_METADATA[route_name]
    matched_example = semantic_match.get("matched_example", "no example available")
    provider = semantic_match.get("provider", "semantic router")
    score = semantic_match.get("score")
    reason = (
        f"{metadata['reason']} Top matched example: \"{matched_example}\" "
        f"(provider: {provider}, score: {score})."
    )
    return QueryRouteDecision(
        route=route_name,
        label=metadata["label"],
        reason=reason,
        uses_rag=metadata.get("uses_rag", False),
        prefers_llm=use_llm,
    )


def route_query(query: str, use_llm: bool = True, chat_history: list[dict[str, Any]] | None = None) -> QueryRouteDecision:
    safety_issue = detect_safety_compliance_issue(query)
    if safety_issue is not None:
        return QueryRouteDecision(
            route="safety_compliance",
            label="Safety and compliance",
            reason=safety_issue["reason"],
        )

    if is_architecture_query(query):
        return QueryRouteDecision(
            route="architecture_rules",
            label="Architecture rules",
            reason="The question is about how the app is built, so it is safer to answer from local architecture rules.",
        )

    if is_hybrid_profile_rag_market_query(query):
        return QueryRouteDecision(
            route="hybrid_profile_rag_market",
            label="Hybrid profile + RAG + market",
            reason="The question mixes profile guidance, retrieved finance rules, and market context, so the app should combine all three lanes.",
            uses_rag=True,
            prefers_llm=use_llm,
        )

    if is_hybrid_spending_recommendation_query(query):
        return QueryRouteDecision(
            route="hybrid_spending_recommendation",
            label="Hybrid spending + recommendation",
            reason="The question mixes spending behavior with product guidance, so the app should combine spending context with deterministic recommendation logic.",
            prefers_llm=use_llm,
        )

    if is_spending_query(query):
        return QueryRouteDecision(
            route="spending_rules",
            label="Spending rules",
            reason="The question is mainly about structured transaction data and deterministic totals.",
        )

    if is_account_summary_query(query):
        return QueryRouteDecision(
            route="account_summary_rules",
            label="Account summary",
            reason="The question is asking for household balances, liquidity, or net-worth context.",
        )

    if is_performance_explanation_query(query):
        return QueryRouteDecision(
            route="performance_explanation",
            label="Performance explanation",
            reason="The question is asking why portfolio returns changed, so the app should use monthly performance data and market commentary.",
            uses_rag=True,
            prefers_llm=use_llm,
        )

    if is_risk_profile_query(query):
        return QueryRouteDecision(
            route="risk_profile_rules",
            label="Risk profile",
            reason="The question is asking about risk tolerance or suitability, so the app should answer from the client profile snapshot.",
        )

    if is_portfolio_explanation_query(query):
        return QueryRouteDecision(
            route="portfolio_explanation",
            label="Portfolio explanation",
            reason="The question is asking about holdings, allocation, or portfolio construction.",
        )

    if is_hybrid_market_recommendation_query(query):
        return QueryRouteDecision(
            route="hybrid_market_advice",
            label="Hybrid market + recommendation",
            reason="The question mixes profile-based guidance with ETF or market context, so the app should combine recommendation logic with live market data.",
            prefers_llm=use_llm,
        )

    if is_hybrid_rule_recommendation_query(query):
        return QueryRouteDecision(
            route="hybrid_rule_advice",
            label="Hybrid recommendation + rules",
            reason="The question mixes profile-based guidance with account rules, so the app should combine deterministic recommendations with retrieved policy context.",
            uses_rag=True,
            prefers_llm=use_llm,
        )

    if is_market_snapshot_query(query):
        return QueryRouteDecision(
            route="market_snapshot_rules",
            label="Live market snapshot",
            reason="The question is asking for live ETF data, so the app should fetch market data directly.",
        )

    if is_market_explanation_query(query):
        return QueryRouteDecision(
            route="market_explanation",
            label="Market explanation",
            reason="The question is asking for market interpretation rather than only quotes, so the app should combine watchlist context with portfolio-aware commentary.",
        )

    if is_recommendation_query(query):
        return QueryRouteDecision(
            route="recommendation_rules",
            label="Recommendation rules",
            reason="The question is asking for prioritization, so the deterministic recommendation engine should lead.",
        )

    if is_rag_knowledge_query(query):
        return QueryRouteDecision(
            route="rag_knowledge",
            label="RAG knowledge",
            reason="The question is asking for finance knowledge or rules, so the app should retrieve reference context first.",
            uses_rag=True,
            prefers_llm=use_llm and not is_fhsa_tfsa_comparison_query(query),
        )

    semantic_match = semantic_route_query(query, chat_history=chat_history)
    if semantic_match is not None and semantic_match["score"] >= semantic_route_threshold(semantic_match["provider"]):
        return _build_semantic_route_decision(
            semantic_match["route"],
            use_llm=use_llm,
            semantic_match=semantic_match,
        )

    if use_llm:
        return QueryRouteDecision(
            route="llm_general",
            label="General LLM answer",
            reason="No stronger rules-based route matched, so the app can use the general LLM path.",
            prefers_llm=True,
        )

    return QueryRouteDecision(
        route="rules_fallback",
        label="Rules fallback",
        reason="No stronger route matched and LLM use is disabled, so the app falls back to local rules.",
    )
