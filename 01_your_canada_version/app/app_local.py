from __future__ import annotations

from html import escape
import os
from textwrap import shorten

import altair as alt
import pandas as pd
import streamlit as st

from conversation_memory import get_recent_chat_history
from data_sources import ENV_FILE_USED, fetch_market_snapshot
from demo_governance import build_request_metadata
from local_financial_qa import (
    analyze_financial_data_local,
    build_recommendation_cards,
    build_scenario_summary,
    build_summary,
    load_financial_context,
)
from rag_pipeline import get_rag_index_status

st.set_page_config(page_title="Canadian Wealth Insights", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .market-card {
        border: 1px solid #e5e7eb;
        border-top: 4px solid var(--accent);
        border-radius: 18px;
        padding: 1.1rem 1.15rem 1rem 1.15rem;
        background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
        min-height: 280px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
    }
    .market-symbol {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    .market-label {
        color: #6b7280;
        font-size: 0.95rem;
        min-height: 2.6rem;
        margin-bottom: 1rem;
    }
    .market-price {
        font-size: 2rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 1rem;
    }
    .market-chip-row {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .market-chip {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        font-size: 0.92rem;
        font-weight: 600;
        background: var(--bg);
        color: var(--fg);
    }
    .market-note {
        color: #6b7280;
        font-size: 0.94rem;
        line-height: 1.55;
    }
    .kpi-card {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1rem 1.1rem 1rem;
        background: linear-gradient(180deg, #ffffff 0%, #fcfcfd 100%);
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.03);
    }
    .kpi-label {
        color: #8a8f98;
        font-size: 0.96rem;
        margin-bottom: 1rem;
    }
    .kpi-value {
        color: #353848;
        font-size: 1.8rem;
        line-height: 1.25;
        font-weight: 800;
        margin-bottom: 1rem;
        overflow-wrap: anywhere;
    }
    .kpi-value.compact {
        font-size: 1.25rem;
        line-height: 1.35;
    }
    .kpi-note {
        color: #4b5563;
        font-size: 0.98rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "selected_scenario" not in st.session_state:
    st.session_state.selected_scenario = "buy_home_first"
if "developer_mode" not in st.session_state:
    st.session_state.developer_mode = False
if "entry_channel" not in st.session_state:
    st.session_state.entry_channel = "web_app"
if "entry_device" not in st.session_state:
    st.session_state.entry_device = "browser"
if "demo_session_id" not in st.session_state:
    st.session_state.demo_session_id = build_request_metadata(user_id="demo").session_id


@st.cache_data(ttl=900, show_spinner=False)
def load_cached_market_snapshot() -> dict:
    return fetch_market_snapshot()


SCENARIO_PRESETS = [
    {
        "id": "buy_home_first",
        "title": "Buy a Home First",
        "subtitle": "Condo down payment in the next two years",
        "description": "Prioritise first-home savings, while still protecting a practical cash buffer.",
        "overrides": {
            "financial_goals": "Save for a first condo down payment within two years while building stable cash reserves.",
            "short_term_goals": "Grow condo down payment savings and keep a three-month emergency fund.",
            "long_term_goal": "Continue long-term investing after the down payment plan is on track.",
        },
    },
    {
        "id": "emergency_fund_first",
        "title": "Build Safety First",
        "subtitle": "Emergency fund before more investing",
        "description": "Focus on liquidity and resilience before expanding tax-sheltered investing.",
        "overrides": {
            "financial_goals": "Build a six-month emergency fund before taking on more market risk.",
            "short_term_goals": "Move extra cash into liquid savings until the emergency fund is complete.",
            "long_term_goal": "Resume broader investing after the cash cushion is fully built.",
        },
    },
    {
        "id": "tax_optimization",
        "title": "Optimize Tax Savings",
        "subtitle": "Use tax shelters more deliberately",
        "description": "Lean harder into registered accounts and contribution planning.",
        "overrides": {
            "financial_goals": "Optimise tax-sheltered saving across FHSA, TFSA, and RRSP while keeping healthy cash flow.",
            "short_term_goals": "Review available contribution room and place new savings in the best tax wrapper first.",
            "long_term_goal": "Lower taxes over time while still growing long-term investments.",
        },
    },
    {
        "id": "retirement_first",
        "title": "Retirement First",
        "subtitle": "Long-term investing takes the lead",
        "description": "Shift the recommendation engine toward retirement accounts and diversified investing.",
        "overrides": {
            "financial_goals": "Accelerate retirement saving and long-term wealth building through diversified investing.",
            "short_term_goals": "Keep only a modest cash reserve while directing new savings toward retirement accounts.",
            "long_term_goal": "Reach financial independence with a retirement-first plan.",
            "preferred_investment_types": "ETF Investing, Robo-Advising",
        },
    },
]


def normalize_message(message):
    if isinstance(message, dict):
        return message
    role, content = message
    return {
        "role": role,
        "content": content,
        "analysis": None,
        "mode": "legacy",
        "request_preview": None,
        "warning": None,
        "error": None,
        "route_label": None,
        "route_reason": None,
        "workflow_backend": None,
        "recommendation_cards": None,
        "market_snapshot": None,
        "rag_sources": None,
        "request_metadata": None,
        "access_decision": None,
        "tool_outputs": None,
        "compliance_report": None,
        "audit_info": None,
    }


def render_structured_analysis(analysis: dict) -> None:
    with st.expander("Structured analysis"):
        st.write(f"- Intent: `{analysis['intent']}`")
        st.write(f"- Confidence: `{analysis['confidence']}`")
        st.write("Key insights:")
        for item in analysis["key_insights"]:
            st.write(f"- {item}")
        st.write("Recommended products:")
        for item in analysis["recommended_products"]:
            st.write(f"- {item}")
        st.write("Next actions:")
        for item in analysis["next_actions"]:
            st.write(f"- {item}")


def render_request_preview(preview: dict) -> None:
    with st.expander("LLM request preview"):
        st.write(f"- Model: `{preview['model']}`")
        if preview.get("requested_backend"):
            st.write(f"- Requested backend: `{preview['requested_backend']}`")
        if preview.get("actual_backend"):
            st.write(f"- Actual backend: `{preview['actual_backend']}`")
        st.write("Instructions:")
        st.code(preview["instructions"], language="text")
        if preview.get("context_sections"):
            st.write("Context sections:")
            for key in preview["context_sections"].keys():
                st.write(f"- {key}")
        st.write("Input:")
        st.code(preview["input"], language="text")


def render_rag_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander("Retrieved sources"):
        for source in sources:
            st.markdown(f"**{source['title']}**")
            st.caption(f"Section: {source['section']} | Score: {source['score']}")
            st.write(source["snippet"])
            st.caption(source["source_file"])


def render_governance_details(message: dict) -> None:
    request_metadata = message.get("request_metadata")
    access_decision = message.get("access_decision")
    tool_outputs = message.get("tool_outputs")
    compliance_report = message.get("compliance_report")
    audit_info = message.get("audit_info")

    if request_metadata or access_decision:
        with st.expander("Request governance"):
            if request_metadata:
                st.write(f"- User ID: `{request_metadata.get('user_id')}`")
                st.write(f"- Session: `{request_metadata.get('session_id')}`")
                st.write(f"- Channel: `{request_metadata.get('channel')}`")
                st.write(f"- Device: `{request_metadata.get('device')}`")
                st.write(f"- Timestamp: `{request_metadata.get('timestamp')}`")
            if access_decision:
                st.write(f"- Access allowed: `{access_decision.get('allowed')}`")
                st.write(f"- Authenticated: `{access_decision.get('authenticated')}`")
                st.write(f"- Account owner verified: `{access_decision.get('account_owner_verified')}`")
                st.write(f"- Allowed data scope: `{', '.join(access_decision.get('allowed_data_scope', []))}`")
                st.write(f"- Reason: {access_decision.get('reason')}")

    if tool_outputs:
        with st.expander("Analytics tool output"):
            st.json(tool_outputs)

    if compliance_report:
        with st.expander("Compliance check"):
            st.write(f"- Status: `{compliance_report.get('status')}`")
            st.write(f"- PII redacted: `{compliance_report.get('pii_redacted')}`")
            st.write(f"- Softened advice terms: `{', '.join(compliance_report.get('softened_advice_terms', [])) or 'none'}`")
            st.write(f"- Blocked terms: `{', '.join(compliance_report.get('blocked_terms', [])) or 'none'}`")

    if audit_info:
        with st.expander("Audit log"):
            st.write(f"- Event ID: `{audit_info.get('event_id')}`")
            st.write(f"- File: `{audit_info.get('path')}`")


def queue_sample_question(question: str) -> None:
    st.session_state.pending_query = question


def select_scenario(scenario_id: str) -> None:
    st.session_state.selected_scenario = scenario_id


def format_signed_percent(value: float) -> str:
    return f"{value:+.2f}%"


def market_delta_style(value: float) -> dict[str, str]:
    if value > 0:
        return {"accent": "#16a34a", "fg": "#166534", "bg": "#dcfce7"}
    if value < 0:
        return {"accent": "#dc2626", "fg": "#991b1b", "bg": "#fee2e2"}
    return {"accent": "#6b7280", "fg": "#4b5563", "bg": "#e5e7eb"}


def render_market_snapshot_cards(snapshot: dict) -> None:
    quotes = snapshot.get("quotes", [])
    if not quotes:
        return

    quote_cols = st.columns(len(quotes))
    for col, quote in zip(quote_cols, quotes):
        day_style = market_delta_style(quote["day_change_pct"])
        trend_style = market_delta_style(quote["period_change_pct"])
        accent_style = trend_style if quote["period_change_pct"] != 0 else day_style
        with col:
            st.markdown(
                f"""
                <div class="market-card" style="--accent: {accent_style['accent']};">
                    <div class="market-symbol">{escape(quote['symbol'])}</div>
                    <div class="market-label">{escape(quote['label'])}</div>
                    <div class="market-price">{escape(quote['currency'])} {quote['last_close']:,.2f}</div>
                    <div class="market-chip-row">
                        <span class="market-chip" style="--fg: {day_style['fg']}; --bg: {day_style['bg']};">
                            1-day {format_signed_percent(quote['day_change_pct'])}
                        </span>
                        <span class="market-chip" style="--fg: {trend_style['fg']}; --bg: {trend_style['bg']};">
                            5-day {format_signed_percent(quote['period_change_pct'])}
                        </span>
                    </div>
                    <div class="market-note">{escape(quote['why_it_matters'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_kpi_card(label: str, value: str, note: str, compact: bool = False) -> None:
    value_class = "kpi-value compact" if compact else "kpi-value"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div>
                <div class="kpi-label">{escape(label)}</div>
                <div class="{value_class}">{escape(value)}</div>
            </div>
            <div class="kpi-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_cards(cards, show_heading: bool = True) -> None:
    if not cards:
        return

    badge_help_map = {
        "Best first step": "The strongest place to begin right now",
        "Strong second option": "Very useful once the first step is underway",
        "Useful next move": "Helpful after the first two priorities are covered",
        "Consider later": "Reasonable later, but not before the top priorities",
    }

    primary_card = cards[:1]
    secondary_cards = cards[1:3]
    extra_cards = cards[3:]

    if show_heading:
        st.markdown("#### Recommended Plan")
        st.caption("One lead option, two strong supporting choices, and the rest tucked away.")

    if primary_card:
        card = primary_card[0]
        with st.container(border=True):
            st.markdown("##### Best First Step")
            st.markdown(f"**{card['product_name']}**")
            st.caption(card["category"])
            st.write(f"**Priority:** {card['display_priority']}")
            st.caption(badge_help_map.get(card["display_priority"], ""))
            st.write(f"**Why now:** {card['why_now']}")
            st.write(f"**Best for:** {card['best_for']}")
            st.write(f"**Watchout:** {card['watchout']}")

    if secondary_cards:
        st.markdown("##### Strong Supporting Options")
        cols = st.columns(len(secondary_cards))
        for col, card in zip(cols, secondary_cards):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{card['product_name']}**")
                    st.caption(card["category"])
                    st.write(f"**Priority:** {card['display_priority']}")
                    st.caption(badge_help_map.get(card["display_priority"], ""))
                    st.write(f"**Why now:** {card['why_now']}")
                    st.write(f"**Best for:** {card['best_for']}")
                    st.write(f"**Watchout:** {card['watchout']}")


def should_collapse_recommendation_plan(mode: str | None) -> bool:
    recommendation_heavy_modes = {
        "recommendation_rules",
        "hybrid_spending_llm",
        "hybrid_spending_rules",
        "hybrid_market_llm",
        "hybrid_rules",
        "hybrid_rule_llm",
        "hybrid_rule_rules",
        "hybrid_profile_rag_market_llm",
        "hybrid_profile_rag_market_rules",
    }
    return mode in recommendation_heavy_modes


def should_render_market_snapshot_cards(mode: str | None) -> bool:
    market_card_modes = {
        "market_snapshot_rules",
        "hybrid_market_llm",
        "hybrid_rules",
        "hybrid_profile_rag_market_llm",
        "hybrid_profile_rag_market_rules",
    }
    return mode in market_card_modes


def render_assistant_message(message: dict, developer_mode: bool = False) -> None:
    text = message.get("content") or message.get("answer") or ""
    if text:
        st.markdown(text)
    if developer_mode and message.get("route_label"):
        route_reason = message.get("route_reason")
        st.caption(
            f"Answer route: {message['route_label']}"
            + (f" | {route_reason}" if route_reason else "")
        )
    if developer_mode and message.get("workflow_backend"):
        st.caption(f"Workflow backend: {message['workflow_backend']}")
    if message.get("warning"):
        st.info(message["warning"])
    if message.get("error"):
        st.warning(
            "The OpenAI call failed, so the app showed the local rules-based answer instead.\n\n"
            f"Error details: {message['error']}"
        )
    if developer_mode and message.get("analysis"):
        render_structured_analysis(message["analysis"])
    if message.get("recommendation_cards"):
        if should_collapse_recommendation_plan(message.get("mode")):
            with st.expander("Recommended Plan", expanded=False):
                st.caption("Open to see the full ranked plan and product fit details.")
                render_recommendation_cards(message["recommendation_cards"], show_heading=False)
        else:
            render_recommendation_cards(message["recommendation_cards"])
    if (
        should_render_market_snapshot_cards(message.get("mode"))
        and message.get("market_snapshot")
        and message["market_snapshot"].get("status") == "live"
    ):
        render_market_snapshot_cards(message["market_snapshot"])
        st.caption(f"As of {message['market_snapshot'].get('as_of', 'unknown')} | Source: {message['market_snapshot'].get('provider', 'unknown')}")
    if developer_mode and message.get("rag_sources"):
        render_rag_sources(message["rag_sources"])
    if developer_mode and message.get("request_preview"):
        render_request_preview(message["request_preview"])
    if developer_mode:
        render_governance_details(message)


def build_request_metadata_for_ui(summary: dict) -> dict:
    return build_request_metadata(
        user_id=summary["user_profile"]["user_id"],
        session_id=st.session_state.demo_session_id,
        channel=st.session_state.entry_channel,
        device=st.session_state.entry_device,
    ).to_dict()


def clear_chat_history() -> None:
    st.session_state.chat_history = []
    st.session_state.pending_query = None


def render_chat_history_panel() -> None:
    history = get_recent_chat_history(st.session_state.chat_history, max_messages=8)
    if not history:
        st.caption("Conversation memory is empty right now.")
        return

    with st.expander("Recent conversation memory", expanded=False):
        st.caption("The assistant now uses recent turns to interpret follow-up questions.")
        for message in history:
            role_label = "You" if message["role"] == "user" else "Assistant"
            st.markdown(f"**{role_label}:** {message['content']}")


def submit_query(query: str, has_openai_key: bool, request_metadata: dict) -> None:
    if not query:
        return

    st.session_state.chat_history.append({"role": "user", "content": query, "analysis": None, "mode": "input"})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analysing local financial data..."):
            result = analyze_financial_data_local(
                query,
                use_llm=has_openai_key,
                chat_history=st.session_state.chat_history,
                request_metadata=request_metadata,
            )
        render_assistant_message(result, developer_mode=st.session_state.developer_mode)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "analysis": result.get("analysis"),
            "mode": result.get("mode"),
            "model": result.get("model"),
            "request_preview": result.get("request_preview"),
            "warning": result.get("warning"),
            "error": result.get("error"),
            "route_label": result.get("route_label"),
            "route_reason": result.get("route_reason"),
            "workflow_backend": result.get("workflow_backend"),
            "recommendation_cards": result.get("recommendation_cards"),
            "market_snapshot": result.get("market_snapshot"),
            "rag_sources": result.get("rag_sources"),
            "request_metadata": result.get("request_metadata"),
            "access_decision": result.get("access_decision"),
            "tool_outputs": result.get("tool_outputs"),
            "compliance_report": result.get("compliance_report"),
            "audit_info": result.get("audit_info"),
        }
    )


def get_scenario_results(summary: dict) -> list[dict]:
    scenario_results = []
    for scenario in SCENARIO_PRESETS:
        scenario_summary = build_scenario_summary(summary, scenario["overrides"])
        cards = []
        for item in scenario_summary.get("scored_recommendations", [])[:3]:
            cards.append(
                {
                    "product_name": item["product_name"],
                    "category": item["category"],
                    "display_priority": {
                        "High": "Start now",
                        "Medium": "Good next step",
                        "Conditional": "Consider after core goals",
                        "Later": "Lower priority for now",
                    }.get(item["priority"], item["priority"]),
                }
            )
        scenario_results.append(
            {
                "meta": scenario,
                "summary": scenario_summary,
                "cards": cards,
            }
        )
    return scenario_results


def render_market_snapshot_tab() -> None:
    st.subheader("Market Snapshot")

    snapshot = load_cached_market_snapshot()
    if snapshot["status"] != "live":
        st.info(snapshot.get("message", "The market snapshot is not available right now."))
        if snapshot.get("watchlist"):
            watchlist_text = ", ".join(item["symbol"] for item in snapshot["watchlist"])
            st.write(f"Configured ETF watchlist: {watchlist_text}")
        if snapshot.get("error"):
            st.caption(f"Fetch note: {snapshot['error']}")
        return

    if st.button("Refresh Snapshot", key="refresh_market_snapshot"):
        load_cached_market_snapshot.clear()
        snapshot = load_cached_market_snapshot()

    render_market_snapshot_cards(snapshot)
    st.caption(f"As of {snapshot['as_of']} | Source: {snapshot['provider']}")


def render_scenario_planner(summary: dict) -> None:
    st.title("Scenario Planner")

    scenario_results = get_scenario_results(summary)
    selected_id = st.session_state.selected_scenario

    st.markdown("#### Pick a planning scenario")
    scenario_cols = st.columns(2)
    for index, scenario_result in enumerate(scenario_results):
        scenario = scenario_result["meta"]
        with scenario_cols[index % 2]:
            st.button(
                scenario["title"],
                key=f"scenario_{scenario['id']}",
                use_container_width=True,
                on_click=select_scenario,
                args=(scenario["id"],),
                type="primary" if scenario["id"] == selected_id else "secondary",
            )
            st.caption(scenario["subtitle"])
            st.caption(scenario["description"])

    selected_result = next(
        (item for item in scenario_results if item["meta"]["id"] == st.session_state.selected_scenario),
        scenario_results[0],
    )
    selected_cards = selected_result["cards"]
    selected_summary = selected_result["summary"]

    st.markdown("#### What changes in this scenario")
    insight_cols = st.columns([1.3, 1, 1])
    with insight_cols[0]:
        with st.container(border=True):
            st.markdown(f"**{selected_result['meta']['title']}**")
            st.caption(selected_result["meta"]["subtitle"])
            st.write(selected_result["meta"]["description"])
            st.write(f"**Priority rule:** {selected_summary['priority_reason']}")
    with insight_cols[1]:
        with st.container(border=True):
            lead_card = selected_cards[0] if selected_cards else {}
            st.caption("Lead move")
            st.markdown(f"### {shorten(lead_card.get('product_name', 'Not available'), width=28, placeholder='...')}")
            st.write(lead_card.get("display_priority", ""))
    with insight_cols[2]:
        with st.container(border=True):
            st.caption("Reason tags")
            tags = selected_summary.get("recommendation_reason_tags", [])[:4]
            if tags:
                for tag in tags:
                    st.write(f"- {tag.replace('_', ' ')}")
            else:
                st.write("No special tags detected yet.")

    st.markdown("#### Scenario recommendation")
    render_recommendation_cards(build_recommendation_cards(selected_summary))

    st.markdown("#### Compare top recommendations")
    comparison_rows = []
    for scenario_result in scenario_results:
        top_products = scenario_result["cards"]
        comparison_rows.append(
            {
                "Scenario": scenario_result["meta"]["title"],
                "Lead recommendation": top_products[0]["product_name"] if len(top_products) > 0 else "N/A",
                "Second option": top_products[1]["product_name"] if len(top_products) > 1 else "N/A",
                "Third option": top_products[2]["product_name"] if len(top_products) > 2 else "N/A",
            }
        )
    st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)


def render_dashboard(summary: dict, context) -> None:
    st.title("Client Dashboard")
    st.caption("Review household cash flow, account structure, portfolio positioning, and the supporting case data behind the advisory workspace.")

    dashboard_cards = st.columns(3)
    with dashboard_cards[0]:
        render_kpi_card(
            "Net Worth",
            f"{summary['account_overview']['net_worth']:,.2f}",
            "Household assets less liabilities.",
        )
    with dashboard_cards[1]:
        render_kpi_card(
            "Liquid Assets",
            f"{summary['account_overview']['liquid_assets']:,.2f}",
            "Cash and short-term reserves available now.",
        )
    with dashboard_cards[2]:
        render_kpi_card(
            "Monthly Return",
            f"{summary['portfolio_overview']['latest_month'].get('monthly_return_pct', 0):+.2f}%",
            "Latest recorded household portfolio return.",
        )

    overview_tab, market_tab, profile_tab, data_tab = st.tabs(["Overview", "Market Snapshot", "Client Profile", "Data Explorer"])

    with overview_tab:
        monthly_df = pd.DataFrame(summary["monthly_spending"])
        if not monthly_df.empty:
            monthly_df["period"] = monthly_df["year"].astype(int).astype(str) + "-" + monthly_df["month"].astype(int).astype(str).str.zfill(2)
            st.subheader("Monthly Spending Trend")
            trend_chart = (
                alt.Chart(monthly_df)
                .mark_area(line={"color": "#0f766e"}, color=alt.Gradient(
                    gradient="linear",
                    stops=[alt.GradientStop(color="#99f6e4", offset=0), alt.GradientStop(color="#ffffff", offset=1)],
                    x1=1,
                    x2=1,
                    y1=1,
                    y2=0,
                ))
                .encode(
                    x=alt.X("period:N", sort=None, title="Month"),
                    y=alt.Y("amount:Q", title="Monthly spending"),
                    tooltip=["period", alt.Tooltip("amount:Q", format=",.2f")],
                )
                .properties(height=300)
            )
            st.altair_chart(trend_chart, use_container_width=True)

        top_spending_df = pd.DataFrame(
            {
                "category": list(summary["top_spending_categories"].keys()),
                "amount": list(summary["top_spending_categories"].values()),
            }
        )
        if not top_spending_df.empty:
            st.subheader("Top Spending Categories")
            spending_chart = (
                alt.Chart(top_spending_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#f97316")
                .encode(
                    x=alt.X("category:N", sort="-y", title="Category"),
                    y=alt.Y("amount:Q", title="Amount"),
                    tooltip=["category", alt.Tooltip("amount:Q", format=",.2f")],
                )
                .properties(height=300)
            )
            st.altair_chart(spending_chart, use_container_width=True)

        st.subheader("Recommendation Snapshot")
        snapshot_cards = [
            f"{card['product_name']} - {card['display_priority']}"
            for card in build_recommendation_cards(summary)[:3]
        ]
        for item in snapshot_cards:
            st.write(f"- {item}")

        st.subheader("Account Snapshot")
        account_preview = pd.DataFrame(summary["account_overview"]["accounts"][:5])[["account_name", "account_type", "balance", "goal"]]
        st.dataframe(account_preview, width="stretch", hide_index=True)

        st.subheader("Portfolio Snapshot")
        holdings_preview = pd.DataFrame(summary["portfolio_overview"]["top_holdings"][:5])[["symbol", "holding_name", "asset_class", "market_value", "one_month_return_pct"]]
        st.dataframe(holdings_preview, width="stretch", hide_index=True)

    with market_tab:
        render_market_snapshot_tab()

    with profile_tab:
        profile = summary["user_profile"]
        left, right = st.columns(2)
        with left:
            st.write(f"**Name:** {profile['name']}")
            st.write(f"**Age:** {profile['age']}")
            st.write(f"**Occupation:** {profile['occupation']}")
            st.write(f"**Annual Income:** {profile['annual_income']}")
            st.write(f"**Risk Tolerance:** {profile['risk_tolerance']}")
        with right:
            st.write(f"**Short-Term Goal:** {profile['short_term_goals']}")
            st.write(f"**Long-Term Goal:** {profile['long_term_goal']}")
            st.write(f"**Investment Experience:** {profile['investment_experience']}")
            st.write(f"**Preferred Investments:** {profile['preferred_investment_types']}")
            st.write(f"**Priority Rule:** {summary['priority_reason']}")

    with data_tab:
        st.subheader("Transactions")
        st.dataframe(context.transactions, width="stretch")
        st.subheader("Monthly Summary")
        st.dataframe(context.monthly, width="stretch")
        st.subheader("Account Summary")
        st.dataframe(context.account_summary, width="stretch")
        st.subheader("Portfolio Holdings")
        st.dataframe(context.portfolio_holdings, width="stretch")
        st.subheader("Portfolio Performance")
        st.dataframe(context.portfolio_performance, width="stretch")
        st.subheader("User Profile")
        st.dataframe(context.user_info, width="stretch")
        if context.product_catalog is not None:
            st.subheader("Product Catalog")
            st.dataframe(context.product_catalog, width="stretch")


def render_copilot(has_openai_key: bool, developer_mode: bool = False) -> None:
    st.title("Canadian Wealth Insights")
    st.caption("A client-facing advisory workspace for household cash flow, account strategy, portfolio explanation, and Canada-first financial guidance.")

    st.markdown("#### Try a sample question")
    sample_cols = st.columns(2)
    sample_questions = [
        "Show my household account summary.",
        "Explain my current portfolio allocation.",
        "Why did my returns change this month?",
        "Explain market changes and what they mean for this portfolio.",
    ]
    for index, question in enumerate(sample_questions):
        with sample_cols[index % 2]:
            st.button(question, key=f"sample_{index}", use_container_width=True, on_click=queue_sample_question, args=(question,))

    st.subheader("Chat")
    chat_control_cols = st.columns([1, 5])
    with chat_control_cols[0]:
        st.button("Clear chat", key="clear_chat_button", on_click=clear_chat_history, use_container_width=True)
    with chat_control_cols[1]:
        render_chat_history_panel()
    for raw_message in st.session_state.chat_history:
        message = normalize_message(raw_message)
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message, developer_mode=developer_mode)
            else:
                st.markdown(message["content"])

    request_metadata = build_request_metadata_for_ui(summary)
    user_query = st.chat_input("Ask about cash flow, account balances, portfolio positioning, market context, FHSA, TFSA, RRSP, or risk profile...")
    queued_query = st.session_state.pending_query
    if queued_query:
        st.session_state.pending_query = None
        submit_query(queued_query, has_openai_key, request_metadata)
    elif user_query:
        submit_query(user_query, has_openai_key, request_metadata)


context = load_financial_context()
summary = build_summary(context)

with st.sidebar:
    page = st.radio("Workspace", ["Copilot", "Scenario Planner", "Dashboard"], label_visibility="visible")
    st.header("Run Mode")
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    rag_status = get_rag_index_status()
    mode_label = "Hybrid mode" if has_openai_key else "Rules only"
    st.write(f"- Current mode: `{mode_label}`")
    if has_openai_key:
        st.write(f"- Model: `{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}`")
    st.write(f"- Environment file: `{ENV_FILE_USED}`" if ENV_FILE_USED else "- Environment file: `not found`")
    st.session_state.developer_mode = st.checkbox(
        "Developer mode",
        value=st.session_state.developer_mode,
        help="Show routing, retrieved sources, and LLM request details.",
    )
    if st.session_state.developer_mode:
        st.caption("System details are visible for learning and debugging.")

    with st.expander("Data Layers", expanded=False):
        st.write("- Client case data: household transactions, accounts, holdings, and performance files in `artifacts_canada/`")
        st.write("- Reference knowledge: Canadian rules, planning guidance, and market commentary in `reference_canada/`")
        st.write("- Retrieval layer: finance reference index for retrieval-backed explanations")
        st.write("- Market layer: curated watchlist plus an optional live Yahoo Finance ETF snapshot")
        st.write("- Decision engine: deterministic scoring for product-priority questions")
        st.write(f"- RAG index status: `{rag_status['status']}`")
        if rag_status.get("chunk_count"):
            st.write(f"- RAG chunks: `{rag_status['chunk_count']}`")
        if rag_status.get("embedding_provider"):
            st.write(f"- Retrieval backend: `{rag_status['embedding_provider']}`")
    with st.expander("Entry Metadata", expanded=False):
        st.session_state.entry_channel = st.selectbox(
            "Channel",
            ["web_app", "mobile_app", "advisor_platform"],
            index=["web_app", "mobile_app", "advisor_platform"].index(st.session_state.entry_channel),
        )
        st.session_state.entry_device = st.selectbox(
            "Device",
            ["browser", "ios_phone", "android_phone", "advisor_desktop"],
            index=["browser", "ios_phone", "android_phone", "advisor_desktop"].index(st.session_state.entry_device),
        )
        st.write(f"- Session ID: `{st.session_state.demo_session_id}`")

if page == "Copilot":
    render_copilot(has_openai_key, developer_mode=st.session_state.developer_mode)
elif page == "Scenario Planner":
    render_scenario_planner(summary)
else:
    render_dashboard(summary, context)
