from __future__ import annotations

from dataclasses import dataclass

from data_sources import fetch_market_snapshot
from intent_engine import CapabilityPlan
from query_router import QueryRouteDecision, is_market_snapshot_query
from retrieval_backend import get_retriever_backend


@dataclass
class OrchestratedContext:
    market_snapshot: dict | None = None
    retrieval_result: dict | None = None


def _legacy_plan_from_route(route_decision: QueryRouteDecision) -> CapabilityPlan:
    tool_calls: list[str] = []
    if route_decision.route in {
        "market_snapshot_rules",
        "market_explanation",
        "performance_explanation",
        "hybrid_market_advice",
        "hybrid_profile_rag_market",
    }:
        tool_calls.append("market_snapshot")
    if route_decision.uses_rag:
        tool_calls.append("reference_retrieval")
    if route_decision.route == "performance_explanation":
        tool_calls.append("portfolio_performance_toolkit")
    return CapabilityPlan(
        tool_calls=tool_calls,
        requires_generation=route_decision.prefers_llm,
        requires_compliance_review=True,
        ui_route_label=route_decision.label,
        route_reason=route_decision.reason,
        legacy_route=route_decision.route,
        uses_rag=route_decision.uses_rag,
        prefers_llm=route_decision.prefers_llm,
    )


def should_fetch_market_snapshot(query: str, plan_or_route: CapabilityPlan | QueryRouteDecision) -> bool:
    if isinstance(plan_or_route, QueryRouteDecision):
        return _legacy_plan_from_route(plan_or_route).tool_calls.count("market_snapshot") > 0 or is_market_snapshot_query(query)
    return "market_snapshot" in plan_or_route.tool_calls


def should_fetch_rag_context(plan_or_route: CapabilityPlan | QueryRouteDecision) -> bool:
    if isinstance(plan_or_route, QueryRouteDecision):
        return plan_or_route.uses_rag
    return "reference_retrieval" in plan_or_route.tool_calls


def gather_orchestrated_context(query: str, plan_or_route: CapabilityPlan | QueryRouteDecision) -> OrchestratedContext:
    plan = _legacy_plan_from_route(plan_or_route) if isinstance(plan_or_route, QueryRouteDecision) else plan_or_route
    context = OrchestratedContext()

    if should_fetch_market_snapshot(query, plan):
        context.market_snapshot = fetch_market_snapshot()

    if should_fetch_rag_context(plan):
        retriever = get_retriever_backend()
        context.retrieval_result = retriever.retrieve(query, top_k=4, filters=None).model_dump()

    return context
