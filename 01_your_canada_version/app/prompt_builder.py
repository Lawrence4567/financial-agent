from __future__ import annotations

import json

from conversation_memory import format_chat_history_as_text
from query_understanding import extract_spending_category_match, is_semantic_spending_query, normalize_query_text


def _is_profile_relevant_query(query: str) -> bool:
    query_lower = query.lower()
    return any(
        term in query_lower
        for term in [
            "my profile",
            "for me",
            "my goal",
            "my goals",
            "should i",
            "focus on",
            "recommend",
            "priority",
            "fit my goals",
        ]
    )


def _is_spending_relevant_query(query: str) -> bool:
    query_lower = normalize_query_text(query)
    return (
        any(
        term in query_lower
        for term in [
            "spending",
            "expense",
            "expenses",
            "cash flow",
            "cashflow",
            "monthly",
            "budget",
            "spent",
            "cost",
            "pay",
        ]
        )
        or is_semantic_spending_query(query)
        or extract_spending_category_match(query) is not None
    )


def _is_market_relevant_query(query: str) -> bool:
    query_lower = query.lower()
    return any(
        term in query_lower
        for term in [
            "market",
            "etf",
            "price",
            "prices",
            "quote",
            "quotes",
            "performance",
            "snapshot",
            "ticker",
            "market change",
            "market changes",
        ]
    )


def _is_account_relevant_query(query: str) -> bool:
    query_lower = normalize_query_text(query)
    return any(
        term in query_lower
        for term in [
            "account summary",
            "account overview",
            "net worth",
            "balance",
            "balances",
            "cash position",
        ]
    )


def _is_portfolio_relevant_query(query: str) -> bool:
    query_lower = normalize_query_text(query)
    return any(
        term in query_lower
        for term in [
            "portfolio",
            "holdings",
            "allocation",
            "asset mix",
            "return",
            "returns",
            "performance",
        ]
    )


def _build_spending_context(summary: dict) -> dict:
    return {
        "sample_window": "January 2025 to June 2025",
        "six_month_total_outflows": summary["total_debits"],
        "six_month_total_inflows": summary["total_credits"],
        "six_month_net_cash_flow": summary["net_cash_flow"],
        "average_monthly_cash_outflow": summary["average_monthly_spending"],
        "top_spending_categories": summary["top_spending_categories"],
        "all_category_spending_totals": summary.get("category_spending_totals", {}),
        "monthly_spending": summary["monthly_spending"],
    }


def _build_profile_context(summary: dict) -> dict:
    top_recommendations = summary.get("scored_recommendations", [])[:3]
    return {
        "user_profile": summary["user_profile"],
        "recommended_priority_order": summary["recommended_priority_order"],
        "priority_reason": summary["priority_reason"],
        "top_recommendations": [
            {
                "product_name": item["product_name"],
                "category": item["category"],
                "priority": item["priority"],
                "score": item["score"],
                "reasons": item["reasons"][:3],
                "watchouts": item["watchouts"][:2],
            }
            for item in top_recommendations
        ],
        "reason_tags": summary.get("recommendation_reason_tags", []),
    }


def _build_account_context(summary: dict) -> dict:
    overview = summary.get("account_overview", {})
    return {
        "total_assets": overview.get("total_assets"),
        "total_liabilities": overview.get("total_liabilities"),
        "net_worth": overview.get("net_worth"),
        "liquid_assets": overview.get("liquid_assets"),
        "registered_balance": overview.get("registered_balance"),
        "contribution_ytd": overview.get("contribution_ytd"),
        "accounts": overview.get("accounts", [])[:6],
    }


def _build_portfolio_context(summary: dict) -> dict:
    overview = summary.get("portfolio_overview", {})
    return {
        "total_market_value": overview.get("total_market_value"),
        "unrealized_gain": overview.get("unrealized_gain"),
        "asset_mix": overview.get("asset_mix", {}),
        "region_mix": overview.get("region_mix", {}),
        "top_holdings": overview.get("top_holdings", [])[:5],
        "latest_month": overview.get("latest_month", {}),
        "previous_month": overview.get("previous_month", {}),
    }


def _build_market_commentary_context(summary: dict) -> dict:
    market_commentary = summary.get("market_commentary", {})
    return {
        "current_market_story": market_commentary.get("current_market_story", {}),
        "current_themes": market_commentary.get("current_themes", [])[:3],
        "month_explanations": market_commentary.get("month_explanations", [])[:3],
    }


def _build_safety_context(summary: dict) -> dict:
    planning_guidance = summary.get("reference_knowledge", {}).get("planning_guidance", {})
    return {
        "advisor_safety_notes": planning_guidance.get("advisor_safety_notes", []),
        "budgeting_guidelines": planning_guidance.get("budgeting_guidelines", []),
    }


def _build_source_context(summary: dict) -> dict:
    return {
        "data_source_overview": summary.get("data_source_overview", {}),
        "route_priority_reason": summary.get("priority_reason"),
    }


def _build_retrieved_context(retrieval_result: dict | None) -> dict | None:
    if not retrieval_result:
        return None

    chunks = retrieval_result.get("chunks", [])
    if not chunks:
        return None

    return {
        "retrieval_mode": retrieval_result.get("retrieval_mode", "unknown"),
        "top_chunks": [
            {
                "title": chunk["title"],
                "section": chunk["section"],
                "score": chunk["score"],
                "text": chunk["text"],
            }
            for chunk in chunks[:4]
        ],
    }


def _build_market_context(market_snapshot: dict | None) -> dict | None:
    if not market_snapshot or market_snapshot.get("status") != "live":
        return None

    return {
        "provider": market_snapshot.get("provider"),
        "as_of": market_snapshot.get("as_of"),
        "quotes": [
            {
                "symbol": quote["symbol"],
                "label": quote["label"],
                "currency": quote["currency"],
                "last_close": quote["last_close"],
                "day_change_pct": quote["day_change_pct"],
                "period_change_pct": quote["period_change_pct"],
                "why_it_matters": quote["why_it_matters"],
            }
            for quote in market_snapshot.get("quotes", [])
        ],
    }


def assemble_prompt_context(
    query: str,
    summary: dict,
    route_label: str,
    chat_history: list[dict] | None = None,
    retrieval_result: dict | None = None,
    market_snapshot: dict | None = None,
    extra_sections: dict[str, object] | None = None,
) -> dict:
    sections: dict[str, object] = {
        "question": query,
        "route_context": {
            "route_label": route_label,
            "goal": "Answer the user clearly while keeping financial explanations grounded in the available app context.",
        },
        "safety_context": _build_safety_context(summary),
    }

    history_text = format_chat_history_as_text(chat_history, max_messages=6, include_route=True, max_chars=1000)
    if history_text:
        sections["conversation_context"] = history_text

    if _is_profile_relevant_query(query):
        sections["profile_context"] = _build_profile_context(summary)

    if _is_spending_relevant_query(query):
        sections["spending_context"] = _build_spending_context(summary)

    if _is_market_relevant_query(query):
        market_context = _build_market_context(market_snapshot)
        if market_context is not None:
            sections["market_context"] = market_context
        sections["market_commentary_context"] = _build_market_commentary_context(summary)

    if _is_account_relevant_query(query):
        sections["account_context"] = _build_account_context(summary)

    if _is_portfolio_relevant_query(query):
        sections["portfolio_context"] = _build_portfolio_context(summary)

    retrieved_context = _build_retrieved_context(retrieval_result)
    if retrieved_context is not None:
        sections["retrieved_context"] = retrieved_context

    if "profile_context" not in sections and "spending_context" not in sections:
        sections["source_context"] = _build_source_context(summary)

    if extra_sections:
        sections.update(extra_sections)

    return sections


def render_prompt_context(sections: dict) -> str:
    prompt_parts = []
    for key, value in sections.items():
        label = key.replace("_", " ").title()
        if isinstance(value, str):
            prompt_parts.append(f"{label}:\n{value}")
        else:
            prompt_parts.append(f"{label}:\n{json.dumps(value, indent=2, ensure_ascii=False)}")
    return "\n\n".join(prompt_parts)


def build_general_llm_request(
    query: str,
    summary: dict,
    instructions: str,
    model: str,
    route_label: str,
    chat_history: list[dict] | None = None,
    market_snapshot: dict | None = None,
    extra_sections: dict[str, object] | None = None,
) -> dict:
    sections = assemble_prompt_context(
        query=query,
        summary=summary,
        route_label=route_label,
        chat_history=chat_history,
        market_snapshot=market_snapshot,
        extra_sections=extra_sections,
    )
    return {
        "model": model,
        "instructions": instructions,
        "input": render_prompt_context(sections),
        "context_sections": sections,
    }


def build_rag_llm_request(
    query: str,
    retrieval_result: dict,
    summary: dict,
    model: str,
    route_label: str,
    chat_history: list[dict] | None = None,
    market_snapshot: dict | None = None,
    extra_sections: dict[str, object] | None = None,
) -> dict:
    query_lower = query.lower()
    is_comparison = "difference" in query_lower or "compare" in query_lower
    is_rule_question = any(
        term in query_lower for term in ["eligibility", "eligible", "contribution", "withdrawal", "rule", "rules"]
    )

    sections = assemble_prompt_context(
        query=query,
        summary=summary,
        route_label=route_label,
        chat_history=chat_history,
        retrieval_result=retrieval_result,
        market_snapshot=market_snapshot,
        extra_sections=extra_sections,
    )

    instructions = (
        "You are a beginner-friendly Canadian finance knowledge assistant. "
        "Answer using the retrieved finance reference context first. "
        "Do not invent contribution limits, legal thresholds, account conditions, or product terms that are not in the provided context. "
        "If the context does not contain an exact number or exact legal test, say the user should verify the latest CRA wording instead of guessing. "
        "Do not add recommendation sections unless the user explicitly asked what they should choose. "
        "Keep the answer clear, practical, and educational."
    )
    if is_comparison:
        instructions += (
            " For comparison questions, start with a neutral side-by-side explanation. "
            "Only add a short relevance note for the user's situation if profile context was provided. "
            "Do not add a recommendation section or tell the user which account is best unless profile context was provided."
        )
    if is_rule_question:
        instructions += (
            " For rule or eligibility questions, stay especially close to the retrieved source wording. "
            "Use careful language such as 'generally', 'official CRA rules', and 'verify the latest CRA wording' where appropriate."
        )

    return {
        "model": model,
        "instructions": instructions,
        "input": render_prompt_context(sections),
        "context_sections": sections,
    }
