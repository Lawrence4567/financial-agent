from __future__ import annotations

from typing import Callable
from typing_extensions import TypedDict

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - optional dependency
    END = None
    StateGraph = None

from response_orchestrator import OrchestratedContext, gather_orchestrated_context
from query_router import QueryRouteDecision, route_query


class AnalysisGraphState(TypedDict, total=False):
    query: str
    use_llm: bool
    summary: dict
    chat_history: list[dict]
    request_metadata: dict
    access_decision: dict
    route_decision: QueryRouteDecision
    orchestrated_context: OrchestratedContext
    tool_outputs: dict
    answer_payload: dict


def langgraph_available() -> bool:
    return StateGraph is not None and END is not None


def should_use_langgraph_flow() -> bool:
    return langgraph_available()


def run_langgraph_analysis_flow(
    *,
    query: str,
    use_llm: bool,
    summary: dict,
    chat_history: list[dict] | None,
    request_metadata: dict,
    verify_access_fn: Callable[[dict, dict], dict],
    run_tools_fn: Callable[[str, dict, QueryRouteDecision, OrchestratedContext], dict],
    execute_route_fn: Callable[..., dict],
    compliance_fn: Callable[[dict, dict], dict],
) -> dict:
    if not langgraph_available():
        raise RuntimeError("LangGraph is not installed in this environment.")

    workflow = StateGraph(AnalysisGraphState)

    def verify_access_node(state: AnalysisGraphState) -> AnalysisGraphState:
        return {
            "access_decision": verify_access_fn(state["request_metadata"], state["summary"]),
        }

    def route_node(state: AnalysisGraphState) -> AnalysisGraphState:
        return {
            "route_decision": route_query(state["query"], use_llm=state["use_llm"], chat_history=state.get("chat_history")),
        }

    def context_node(state: AnalysisGraphState) -> AnalysisGraphState:
        route_decision: QueryRouteDecision = state["route_decision"]
        return {
            "orchestrated_context": gather_orchestrated_context(state["query"], route_decision),
        }

    def tool_node(state: AnalysisGraphState) -> AnalysisGraphState:
        route_decision: QueryRouteDecision = state["route_decision"]
        orchestrated_context: OrchestratedContext = state["orchestrated_context"]
        return {
            "tool_outputs": run_tools_fn(
                state["query"],
                state["summary"],
                route_decision,
                orchestrated_context,
            ),
        }

    def execute_node(state: AnalysisGraphState) -> AnalysisGraphState:
        route_decision: QueryRouteDecision = state["route_decision"]
        orchestrated_context: OrchestratedContext = state["orchestrated_context"]
        answer_payload = execute_route_fn(
            state["query"],
            state["use_llm"],
            state["summary"],
            route_decision,
            orchestrated_context,
            state.get("chat_history"),
            request_metadata=state.get("request_metadata"),
            access_decision=state.get("access_decision"),
            tool_outputs=state.get("tool_outputs"),
        )
        return {
            "answer_payload": answer_payload,
        }

    def compliance_node(state: AnalysisGraphState) -> AnalysisGraphState:
        return {
            "answer_payload": compliance_fn(state["answer_payload"], state["summary"]),
        }

    workflow.add_node("verify_access", verify_access_node)
    workflow.add_node("route_query", route_node)
    workflow.add_node("gather_context", context_node)
    workflow.add_node("run_tools", tool_node)
    workflow.add_node("generate_response", execute_node)
    workflow.add_node("compliance_check", compliance_node)

    workflow.set_entry_point("verify_access")
    workflow.add_edge("verify_access", "route_query")
    workflow.add_edge("route_query", "gather_context")
    workflow.add_edge("gather_context", "run_tools")
    workflow.add_edge("run_tools", "generate_response")
    workflow.add_edge("generate_response", "compliance_check")
    workflow.add_edge("compliance_check", END)

    app = workflow.compile()
    return app.invoke(
        {
            "query": query,
            "use_llm": use_llm,
            "summary": summary,
            "chat_history": chat_history or [],
            "request_metadata": request_metadata,
        }
    )
