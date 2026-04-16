from __future__ import annotations

from dataclasses import dataclass

from data_sources import fetch_market_snapshot
from query_router import QueryRouteDecision, is_market_snapshot_query
from rag_pipeline import retrieve_reference_context


@dataclass
class OrchestratedContext:
    market_snapshot: dict | None = None
    retrieval_result: dict | None = None


def should_fetch_market_snapshot(query: str, route_decision: QueryRouteDecision) -> bool:
    return route_decision.route in {
        "market_snapshot_rules",
        "market_explanation",
        "performance_explanation",
        "hybrid_market_advice",
        "hybrid_profile_rag_market",
    } or is_market_snapshot_query(query)


def should_fetch_rag_context(route_decision: QueryRouteDecision) -> bool:
    return route_decision.uses_rag


def gather_orchestrated_context(query: str, route_decision: QueryRouteDecision) -> OrchestratedContext:
    context = OrchestratedContext()

    if should_fetch_market_snapshot(query, route_decision):
        context.market_snapshot = fetch_market_snapshot()

    if should_fetch_rag_context(route_decision):
        context.retrieval_result = retrieve_reference_context(query, top_k=4)

    return context
