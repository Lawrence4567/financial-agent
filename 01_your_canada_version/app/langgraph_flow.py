from __future__ import annotations

import os
from typing import Callable
from typing_extensions import TypedDict

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - optional dependency
    END = None
    StateGraph = None

from intent_engine import CapabilityPlan, IntentSchema
from response_orchestrator import OrchestratedContext, gather_orchestrated_context


class AnalysisGraphState(TypedDict, total=False):
    query: str
    use_llm: bool
    summary: dict
    chat_history: list[dict]
    request_metadata: dict
    access_decision: dict
    conversation_resolution: dict
    intent: IntentSchema
    capability_plan: CapabilityPlan
    orchestrated_context: OrchestratedContext
    tool_outputs: dict
    answer_payload: dict


def langgraph_available() -> bool:
    return StateGraph is not None and END is not None


def should_use_langgraph_flow() -> bool:
    backend = os.getenv("WORKFLOW_BACKEND", "langgraph").strip().lower()
    return backend == "langgraph" and langgraph_available()


def run_langgraph_analysis_flow(
    *,
    query: str,
    use_llm: bool,
    summary: dict,
    chat_history: list[dict] | None,
    request_metadata: dict,
    access_decision: dict,
    conversation_resolution: dict | None = None,
    parse_intent_fn: Callable[[str, bool, list[dict] | None], IntentSchema],
    plan_capabilities_fn: Callable[[IntentSchema, str], CapabilityPlan],
    run_tools_fn: Callable[[str, dict, IntentSchema, CapabilityPlan, OrchestratedContext], dict],
    validate_tool_results_fn: Callable[[dict, CapabilityPlan], dict],
    execute_plan_fn: Callable[..., dict],
    compliance_fn: Callable[[dict, dict], dict],
) -> dict:
    if not langgraph_available():
        raise RuntimeError("LangGraph is not installed in this environment.")

    workflow = StateGraph(AnalysisGraphState)

    def parse_intent_node(state: AnalysisGraphState) -> AnalysisGraphState:
        return {
            "intent": parse_intent_fn(
                state["query"],
                state["use_llm"],
                state.get("chat_history"),
            ),
        }

    def plan_capabilities_node(state: AnalysisGraphState) -> AnalysisGraphState:
        return {
            "capability_plan": plan_capabilities_fn(state["intent"], state["query"]),
        }

    def context_node(state: AnalysisGraphState) -> AnalysisGraphState:
        return {
            "orchestrated_context": gather_orchestrated_context(state["query"], state["capability_plan"]),
        }

    def tool_node(state: AnalysisGraphState) -> AnalysisGraphState:
        return {
            "tool_outputs": run_tools_fn(
                state["query"],
                state["summary"],
                state["intent"],
                state["capability_plan"],
                state["orchestrated_context"],
            ),
        }

    def validate_node(state: AnalysisGraphState) -> AnalysisGraphState:
        return {
            "tool_outputs": validate_tool_results_fn(
                state.get("tool_outputs", {}),
                state["capability_plan"],
            ),
        }

    def execute_node(state: AnalysisGraphState) -> AnalysisGraphState:
        answer_payload = execute_plan_fn(
            state["query"],
            state["use_llm"],
            state["summary"],
            state["intent"],
            state["capability_plan"],
            state["orchestrated_context"],
            state.get("chat_history"),
            request_metadata=state.get("request_metadata"),
            access_decision=state.get("access_decision"),
            tool_outputs=state.get("tool_outputs"),
            conversation_resolution=state.get("conversation_resolution"),
        )
        return {
            "answer_payload": answer_payload,
        }

    def compliance_node(state: AnalysisGraphState) -> AnalysisGraphState:
        return {
            "answer_payload": compliance_fn(state["answer_payload"], state["summary"]),
        }

    workflow.add_node("parse_intent", parse_intent_node)
    workflow.add_node("plan_capabilities", plan_capabilities_node)
    workflow.add_node("gather_context", context_node)
    workflow.add_node("run_tools", tool_node)
    workflow.add_node("validate_tool_results", validate_node)
    workflow.add_node("generate_answer", execute_node)
    workflow.add_node("compliance_check", compliance_node)

    workflow.set_entry_point("parse_intent")
    workflow.add_edge("parse_intent", "plan_capabilities")
    workflow.add_edge("plan_capabilities", "gather_context")
    workflow.add_edge("gather_context", "run_tools")
    workflow.add_edge("run_tools", "validate_tool_results")
    workflow.add_edge("validate_tool_results", "generate_answer")
    workflow.add_edge("generate_answer", "compliance_check")
    workflow.add_edge("compliance_check", END)

    app = workflow.compile()
    return app.invoke(
        {
            "query": query,
            "use_llm": use_llm,
            "summary": summary,
            "chat_history": chat_history or [],
            "request_metadata": request_metadata,
            "access_decision": access_decision,
            "conversation_resolution": conversation_resolution or {},
        }
    )
