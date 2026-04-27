from __future__ import annotations

import os
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from conversation_memory import format_chat_history_as_text, looks_like_follow_up_query
from query_router import (
    QueryRouteDecision,
    detect_safety_compliance_issue,
    is_account_summary_query,
    is_architecture_query,
    is_fhsa_tfsa_comparison_query,
    is_market_explanation_query,
    is_market_snapshot_query,
    is_performance_explanation_query,
    is_portfolio_explanation_query,
    is_rag_knowledge_query,
    is_recommendation_query,
    is_risk_profile_query,
    is_spending_query,
    route_query,
)
from query_understanding import extract_spending_category_match, extract_spending_scope, normalize_query_text

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


DomainLiteral = Literal[
    "spending",
    "recommendation",
    "account",
    "portfolio",
    "performance",
    "market",
    "knowledge",
    "architecture",
    "safety",
]
OperatorLiteral = Literal[
    "summarize",
    "explain",
    "compare",
    "include",
    "exclude",
    "prioritize",
    "deprioritize",
    "validate",
    "general",
]
PolarityLiteral = Literal["positive", "negative", "neutral"]
ResponseStyleLiteral = Literal["brief", "explanatory", "tabular"]
ConfidenceLiteral = Literal["high", "medium", "low"]


class IntentSchema(BaseModel):
    domain: DomainLiteral
    operator: OperatorLiteral
    target_entities: List[str] = Field(default_factory=list)
    polarity: PolarityLiteral = "neutral"
    needs_recommendation: bool = False
    needs_rag: bool = False
    needs_market_data: bool = False
    needs_portfolio_tools: bool = False
    response_style: ResponseStyleLiteral = "explanatory"
    confidence: ConfidenceLiteral = "medium"
    fallback_reason: Optional[str] = None


class CapabilityPlan(BaseModel):
    tool_calls: List[str] = Field(default_factory=list)
    requires_generation: bool = True
    requires_compliance_review: bool = True
    ui_route_label: str
    route_reason: str
    legacy_route: str
    uses_rag: bool = False
    prefers_llm: bool = True


def _get_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _detect_polarity(query: str) -> PolarityLiteral:
    query_lower = normalize_query_text(query)
    if any(
        phrase in query_lower
        for phrase in [
            "not recommend",
            "do not recommend",
            "don't recommend",
            "avoid",
            "not choose",
            "not spend",
            "excluding",
            "except",
            "outside",
        ]
    ):
        return "negative"
    if any(
        phrase in query_lower
        for phrase in [
            "best",
            "focus on",
            "should i choose",
            "prioritize",
            "recommend",
        ]
    ):
        return "positive"
    return "neutral"


def _extract_target_entities(query: str) -> List[str]:
    query_lower = normalize_query_text(query)
    targets: List[str] = []

    spending_match = extract_spending_category_match(query)
    if spending_match is not None:
        targets.append(spending_match["label"])

    tracked_entities = [
        "FHSA",
        "TFSA",
        "RRSP",
        "GIC",
        "ETF",
        "portfolio",
        "market",
        "risk profile",
        "account summary",
        "food",
    ]
    for entity in tracked_entities:
        if entity.lower() in query_lower and entity not in targets:
            targets.append(entity)

    deduped: List[str] = []
    seen: set[str] = set()
    for target in targets:
        normalized = target.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(target)
    return deduped


def _intent_from_route_decision(
    query: str,
    route_decision: QueryRouteDecision,
    fallback_reason: Optional[str] = None,
) -> IntentSchema:
    query_lower = normalize_query_text(query)
    spending_scope = extract_spending_scope(query)
    targets = _extract_target_entities(query)
    polarity = _detect_polarity(query)

    route = route_decision.route
    if route == "safety_compliance":
        return IntentSchema(
            domain="safety",
            operator="validate",
            target_entities=targets,
            polarity="negative",
            confidence="high",
            needs_recommendation=False,
            fallback_reason=fallback_reason or route_decision.reason,
        )
    if route == "architecture_rules":
        return IntentSchema(
            domain="architecture",
            operator="explain",
            target_entities=targets,
            polarity="neutral",
            confidence="high",
            fallback_reason=fallback_reason,
        )
    if route == "spending_rules":
        operator: OperatorLiteral = "exclude" if spending_scope and spending_scope["mode"] == "exclude" else "include"
        if not extract_spending_category_match(query):
            operator = "summarize"
        return IntentSchema(
            domain="spending",
            operator=operator,
            target_entities=targets,
            polarity=polarity,
            confidence="high",
            fallback_reason=fallback_reason,
        )
    if route == "account_summary_rules":
        return IntentSchema(
            domain="account",
            operator="summarize",
            target_entities=targets,
            polarity="neutral",
            confidence="high",
            fallback_reason=fallback_reason,
        )
    if route == "portfolio_explanation":
        return IntentSchema(
            domain="portfolio",
            operator="explain",
            target_entities=targets,
            polarity="neutral",
            confidence="high",
            fallback_reason=fallback_reason,
        )
    if route == "performance_explanation":
        return IntentSchema(
            domain="performance",
            operator="explain",
            target_entities=targets,
            polarity="neutral",
            confidence="high",
            needs_rag=True,
            needs_portfolio_tools=True,
            fallback_reason=fallback_reason,
        )
    if route in {"market_snapshot_rules", "market_explanation"}:
        operator: OperatorLiteral = "summarize" if route == "market_snapshot_rules" else "explain"
        return IntentSchema(
            domain="market",
            operator=operator,
            target_entities=targets,
            polarity="neutral",
            confidence="high",
            needs_market_data=True,
            fallback_reason=fallback_reason,
        )
    if route == "risk_profile_rules":
        return IntentSchema(
            domain="recommendation",
            operator="validate",
            target_entities=targets or ["risk profile"],
            polarity="neutral",
            confidence="high",
            fallback_reason=fallback_reason,
        )
    if route == "recommendation_rules":
        operator = "deprioritize" if polarity == "negative" else "prioritize"
        return IntentSchema(
            domain="recommendation",
            operator=operator,
            target_entities=targets,
            polarity=polarity,
            confidence="high",
            needs_recommendation=True,
            fallback_reason=fallback_reason,
        )
    if route == "rag_knowledge":
        operator = "compare" if is_fhsa_tfsa_comparison_query(query) or any(term in query_lower for term in ["vs", "versus", "compare", "difference"]) else "explain"
        return IntentSchema(
            domain="knowledge",
            operator=operator,
            target_entities=targets,
            polarity="neutral",
            confidence="high",
            needs_rag=True,
            fallback_reason=fallback_reason,
        )
    if route == "hybrid_spending_recommendation":
        return IntentSchema(
            domain="recommendation",
            operator="prioritize",
            target_entities=targets,
            polarity=polarity,
            confidence="high",
            needs_recommendation=True,
            fallback_reason=fallback_reason,
        )
    if route == "hybrid_market_advice":
        return IntentSchema(
            domain="recommendation",
            operator="prioritize",
            target_entities=targets,
            polarity=polarity,
            confidence="high",
            needs_recommendation=True,
            needs_market_data=True,
            fallback_reason=fallback_reason,
        )
    if route == "hybrid_rule_advice":
        return IntentSchema(
            domain="recommendation",
            operator="prioritize",
            target_entities=targets,
            polarity=polarity,
            confidence="high",
            needs_recommendation=True,
            needs_rag=True,
            fallback_reason=fallback_reason,
        )
    if route == "hybrid_profile_rag_market":
        return IntentSchema(
            domain="recommendation",
            operator="prioritize",
            target_entities=targets,
            polarity=polarity,
            confidence="high",
            needs_recommendation=True,
            needs_rag=True,
            needs_market_data=True,
            fallback_reason=fallback_reason,
        )
    return IntentSchema(
        domain="knowledge" if any(term in query_lower for term in ["rule", "eligibility", "contribution", "withdrawal"]) else "recommendation" if "should i" in query_lower else "spending" if is_spending_query(query) else "architecture" if is_architecture_query(query) else "knowledge" if is_rag_knowledge_query(query) else "market" if is_market_explanation_query(query) or is_market_snapshot_query(query) else "portfolio" if is_portfolio_explanation_query(query) else "account" if is_account_summary_query(query) else "performance" if is_performance_explanation_query(query) else "recommendation" if is_recommendation_query(query) or is_risk_profile_query(query) else "knowledge",
        operator="general",
        target_entities=targets,
        polarity=polarity,
        confidence="low",
        fallback_reason=fallback_reason or route_decision.reason,
    )


def apply_discoverable_intent_overrides(intent: IntentSchema, query: str) -> IntentSchema:
    query_lower = normalize_query_text(query)
    targets = _extract_target_entities(query)
    if detect_safety_compliance_issue(query) is not None:
        return IntentSchema(
            domain="safety",
            operator="validate",
            target_entities=targets,
            polarity="negative",
            confidence="high",
            fallback_reason=intent.fallback_reason,
        )
    if is_fhsa_tfsa_comparison_query(query):
        return intent.model_copy(
            update={
                "domain": "knowledge",
                "operator": "compare",
                "target_entities": targets or ["FHSA", "TFSA"],
                "needs_rag": True,
                "needs_recommendation": False,
                "needs_market_data": False,
                "needs_portfolio_tools": False,
            }
        )
    if is_performance_explanation_query(query):
        return intent.model_copy(
            update={
                "domain": "performance",
                "operator": "explain",
                "needs_portfolio_tools": True,
                "needs_rag": True,
            }
        )
    if is_market_snapshot_query(query):
        return intent.model_copy(
            update={
                "domain": "market",
                "operator": "summarize",
                "needs_market_data": True,
            }
        )
    if is_market_explanation_query(query):
        return intent.model_copy(
            update={
                "domain": "market",
                "operator": "explain",
                "needs_market_data": True,
            }
        )
    spending_scope = extract_spending_scope(query)
    if spending_scope and is_spending_query(query) and not intent.needs_recommendation:
        operator = "exclude" if spending_scope.get("mode") == "exclude" else "include"
        return intent.model_copy(
            update={
                "domain": "spending",
                "operator": operator,
                "target_entities": targets,
            }
        )
    if is_rag_knowledge_query(query):
        operator = "compare" if any(term in query_lower for term in ["difference", "compare", "vs", "versus"]) else "explain"
        return intent.model_copy(
            update={
                "domain": "knowledge",
                "operator": operator,
                "needs_rag": True,
                "target_entities": targets,
            }
        )
    if is_account_summary_query(query):
        return intent.model_copy(
            update={
                "domain": "account",
                "operator": "summarize",
                "target_entities": targets,
            }
        )
    if is_portfolio_explanation_query(query):
        return intent.model_copy(
            update={
                "domain": "portfolio",
                "operator": "explain",
                "target_entities": targets,
            }
        )
    if any(
        phrase in query_lower
        for phrase in ["not recommend", "avoid", "do not recommend", "don't recommend", "should i avoid"]
    ):
        has_rule_signal = any(
            term in query_lower for term in ["rule", "rules", "eligibility", "contribution", "withdrawal", "tax"]
        )
        return intent.model_copy(
            update={
                "domain": "recommendation",
                "operator": "deprioritize",
                "polarity": "negative",
                "needs_recommendation": True,
                "needs_rag": has_rule_signal,
                "needs_market_data": False,
                "needs_portfolio_tools": False,
            }
        )
    return intent


def build_rules_fallback_intent(query: str, use_llm: bool, chat_history: Optional[List[dict]] = None) -> IntentSchema:
    decision = route_query(query, use_llm=use_llm, chat_history=chat_history)
    intent = _intent_from_route_decision(
        query,
        decision,
        fallback_reason=f"Rules fallback from route '{decision.route}'.",
    )
    query_lower = normalize_query_text(query)
    has_rule_signal = any(term in query_lower for term in ["rule", "rules", "eligibility", "contribution", "withdrawal", "tax"])
    has_market_signal = any(term in query_lower for term in ["market", "etf", "price", "prices", "snapshot", "ticker"])
    has_spending_signal = is_spending_query(query)
    has_recommendation_signal = is_recommendation_query(query) or any(
        term in query_lower for term in ["what should i do", "for me", "my profile", "my goals"]
    )

    if has_recommendation_signal:
        intent = intent.model_copy(update={"domain": "recommendation", "needs_recommendation": True})
    if has_rule_signal:
        intent = intent.model_copy(update={"needs_rag": True})
    if has_market_signal and intent.domain in {"market", "recommendation", "performance"}:
        intent = intent.model_copy(update={"needs_market_data": True})
    if intent.domain == "performance":
        intent = intent.model_copy(update={"needs_portfolio_tools": True})
    if has_spending_signal and has_recommendation_signal and intent.operator not in {"deprioritize", "compare"}:
        intent = intent.model_copy(update={"operator": "prioritize"})
    return apply_discoverable_intent_overrides(intent, query)


def parse_intent(query: str, use_llm: bool = True, chat_history: Optional[List[dict]] = None) -> IntentSchema:
    backend = os.getenv("INTENT_BACKEND", "llm").strip().lower()
    if backend != "llm" or not use_llm:
        return build_rules_fallback_intent(query, use_llm=use_llm, chat_history=chat_history)

    client = _get_openai_client()
    if client is None:
        return build_rules_fallback_intent(query, use_llm=use_llm, chat_history=chat_history)

    history_text = format_chat_history_as_text(chat_history, max_messages=6, include_route=True, max_chars=900)
    follow_up_note = "This looks like a short follow-up query." if looks_like_follow_up_query(query) else "This looks like a standalone query."
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    input_text = (
        f"Recent conversation:\n{history_text or 'No recent conversation.'}\n\n"
        f"Current question:\n{query}\n\n"
        f"Note:\n{follow_up_note}"
    )
    instructions = (
        "You are an intent parser for a Canadian financial copilot. "
        "Return only a structured intent schema. "
        "Infer whether the user is asking to include, exclude, compare, prioritize, deprioritize, explain, summarize, or validate. "
        "Negative phrases like 'not recommend', 'avoid', 'not spend on', 'excluding', and 'except' should map to negative polarity and exclude/deprioritize operators when appropriate. "
        "If the question mixes profile guidance with rules, set needs_recommendation and needs_rag to true. "
        "If the question needs current ETF snapshot data, set needs_market_data to true. "
        "If the question asks why returns changed, set domain=performance and needs_portfolio_tools=true. "
        "Use only the allowed enum values. Keep confidence realistic."
    )
    try:
        response = client.responses.parse(
            model=model,
            instructions=instructions,
            input=input_text,
            text_format=IntentSchema,
        )
        parsed = response.output_parsed
        if parsed.fallback_reason:
            return apply_discoverable_intent_overrides(parsed, query)
        return apply_discoverable_intent_overrides(parsed.model_copy(update={"fallback_reason": None}), query)
    except Exception as exc:
        return apply_discoverable_intent_overrides(
            _intent_from_route_decision(
                query,
                route_query(query, use_llm=use_llm, chat_history=chat_history),
                fallback_reason=f"LLM intent parsing failed, so rules fallback was used: {exc}",
            ),
            query,
        )


def plan_capabilities(intent: IntentSchema, query: str) -> CapabilityPlan:
    tool_calls: List[str] = []
    query_lower = normalize_query_text(query)
    has_spending_signal = is_spending_query(query)

    if intent.domain == "spending":
        tool_calls.append("spending_tool")
    if has_spending_signal and intent.domain == "recommendation":
        tool_calls.append("spending_tool")
    if intent.domain == "account":
        tool_calls.append("account_summary_tool")
    if intent.domain == "portfolio":
        tool_calls.append("portfolio_tool")
    if intent.domain == "performance" and "portfolio_performance_toolkit" not in tool_calls:
        tool_calls.append("portfolio_performance_toolkit")
    if intent.needs_portfolio_tools and "portfolio_performance_toolkit" not in tool_calls:
        tool_calls.append("portfolio_performance_toolkit")
    if intent.domain == "recommendation" or intent.needs_recommendation:
        tool_calls.append("recommendation_engine")
    if intent.needs_rag or intent.domain == "knowledge":
        tool_calls.append("reference_retrieval")
    if intent.needs_market_data or intent.domain == "market":
        tool_calls.append("market_snapshot")
    if intent.domain == "architecture":
        tool_calls.append("architecture_context")

    unique_tool_calls = list(dict.fromkeys(tool_calls))

    if intent.domain == "safety":
        return CapabilityPlan(
            tool_calls=[],
            requires_generation=False,
            requires_compliance_review=True,
            ui_route_label="Safety and compliance",
            route_reason="Safety screening blocked the request before the normal workflow.",
            legacy_route="safety_compliance",
            uses_rag=False,
            prefers_llm=False,
        )

    if "spending_tool" in unique_tool_calls and "recommendation_engine" in unique_tool_calls and not intent.needs_rag and not intent.needs_market_data:
        ui_route_label = "Hybrid spending + recommendation"
        legacy_route = "hybrid_spending_recommendation"
        route_reason = "The intent mixes spending facts with product guidance, so the workflow should combine both."
    elif "spending_tool" in unique_tool_calls and "recommendation_engine" in unique_tool_calls and intent.needs_rag:
        ui_route_label = "Hybrid recommendation + rules"
        legacy_route = "hybrid_rule_advice"
        route_reason = "The intent mixes spending context, recommendation logic, and retrieved rules."
    elif intent.domain == "recommendation" and intent.needs_rag and intent.needs_market_data:
        ui_route_label = "Hybrid profile + RAG + market"
        legacy_route = "hybrid_profile_rag_market"
        route_reason = "The intent needs profile guidance, retrieved rules, and market context."
    elif intent.domain == "recommendation" and intent.needs_rag:
        ui_route_label = "Hybrid recommendation + rules"
        legacy_route = "hybrid_rule_advice"
        route_reason = "The intent needs both recommendation logic and retrieved rule context."
    elif intent.domain == "recommendation" and intent.needs_market_data:
        ui_route_label = "Hybrid market + recommendation"
        legacy_route = "hybrid_market_advice"
        route_reason = "The intent needs both recommendation logic and current market context."
    elif intent.domain == "spending":
        ui_route_label = "Spending rules"
        legacy_route = "spending_rules"
        route_reason = "The intent is primarily about transaction totals or category spending."
    elif intent.domain == "account":
        ui_route_label = "Account summary"
        legacy_route = "account_summary_rules"
        route_reason = "The intent is about balances, liquidity, or account overview."
    elif intent.domain == "portfolio":
        ui_route_label = "Portfolio explanation"
        legacy_route = "portfolio_explanation"
        route_reason = "The intent is about portfolio positioning and holdings."
    elif intent.domain == "performance":
        ui_route_label = "Performance explanation"
        legacy_route = "performance_explanation"
        route_reason = "The intent asks why performance changed and needs analytics support."
    elif intent.domain == "market":
        if any(term in query_lower for term in ["price", "quote", "snapshot", "ticker", "latest"]):
            ui_route_label = "Live market snapshot"
            legacy_route = "market_snapshot_rules"
            route_reason = "The intent needs current ETF snapshot data."
        else:
            ui_route_label = "Market explanation"
            legacy_route = "market_explanation"
            route_reason = "The intent asks for market interpretation."
    elif intent.domain == "knowledge":
        ui_route_label = "RAG knowledge"
        legacy_route = "rag_knowledge"
        route_reason = "The intent asks for finance knowledge grounded in retrieved references."
    elif intent.domain == "architecture":
        ui_route_label = "Architecture rules"
        legacy_route = "architecture_rules"
        route_reason = "The intent asks how the app is built or how its data layers work."
    else:
        ui_route_label = "Recommendation rules" if intent.domain == "recommendation" else "General LLM answer"
        legacy_route = "recommendation_rules" if intent.domain == "recommendation" else "llm_general"
        route_reason = "The intent should be answered through the hybrid generation path."

    requires_generation = True
    prefers_llm = True
    deterministic_finance_domains = {"spending", "account", "portfolio", "performance", "market"}
    if intent.domain in deterministic_finance_domains:
        requires_generation = False
        prefers_llm = False
    elif (
        intent.domain == "recommendation"
        and not intent.needs_rag
        and not intent.needs_market_data
        and not has_spending_signal
    ):
        requires_generation = False
        prefers_llm = False

    return CapabilityPlan(
        tool_calls=unique_tool_calls,
        requires_generation=requires_generation,
        requires_compliance_review=True,
        ui_route_label=ui_route_label,
        route_reason=route_reason,
        legacy_route=legacy_route,
        uses_rag="reference_retrieval" in unique_tool_calls,
        prefers_llm=prefers_llm,
    )


def capability_plan_to_route_decision(plan: CapabilityPlan) -> QueryRouteDecision:
    return QueryRouteDecision(
        route=plan.legacy_route,
        label=plan.ui_route_label,
        reason=plan.route_reason,
        uses_rag=plan.uses_rag,
        prefers_llm=plan.prefers_llm,
    )
