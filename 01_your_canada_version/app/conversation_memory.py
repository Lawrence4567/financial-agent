from __future__ import annotations

import re
from typing import Any

_KNOWN_TOPIC_TERMS = [
    "FHSA",
    "TFSA",
    "RRSP",
    "GIC",
    "ETF",
    "High-Interest Savings",
    "portfolio",
    "market",
    "risk profile",
    "rent",
    "groceries",
    "dining",
    "food",
    "spending",
    "cash flow",
    "budget",
]
_CJK_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def _is_recommendation_route(route_label: str | None) -> bool:
    return "recommendation" in (route_label or "").lower()


def normalize_chat_history(chat_history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in chat_history or []:
        if isinstance(message, dict):
            normalized.append(
                {
                    "role": message.get("role", "user"),
                    "content": str(message.get("content", "")),
                    "route_label": message.get("route_label"),
                    "mode": message.get("mode"),
                    "user_query": message.get("user_query"),
                    "effective_query": message.get("effective_query"),
                }
            )
        elif isinstance(message, (list, tuple)) and len(message) >= 2:
            normalized.append(
                {
                    "role": str(message[0]),
                    "content": str(message[1]),
                    "route_label": None,
                    "mode": None,
                    "user_query": None,
                    "effective_query": None,
                }
            )
    return normalized


def get_recent_chat_history(chat_history: list[dict[str, Any]] | None, max_messages: int = 6) -> list[dict[str, Any]]:
    normalized = normalize_chat_history(chat_history)
    if max_messages <= 0:
        return []
    return normalized[-max_messages:]


def format_chat_history_as_text(
    chat_history: list[dict[str, Any]] | None,
    max_messages: int = 6,
    include_route: bool = False,
    max_chars: int = 1200,
) -> str:
    messages = get_recent_chat_history(chat_history, max_messages=max_messages)
    lines: list[str] = []
    total_chars = 0

    for message in messages:
        prefix = "User" if message["role"] == "user" else "Assistant"
        if include_route and message.get("route_label"):
            prefix = f"{prefix} ({message['route_label']})"
        content = " ".join(message.get("content", "").split())
        if not content:
            continue
        line = f"{prefix}: {content}"
        total_chars += len(line)
        if total_chars > max_chars:
            break
        lines.append(line)

    return "\n".join(lines)


def get_last_assistant_route(chat_history: list[dict[str, Any]] | None) -> str | None:
    for message in reversed(normalize_chat_history(chat_history)):
        if message.get("role") == "assistant" and message.get("route_label"):
            return str(message["route_label"])
    return None


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def get_last_assistant_message(chat_history: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    for message in reversed(normalize_chat_history(chat_history)):
        if message.get("role") == "assistant":
            return message
    return None


def get_previous_user_message(
    chat_history: list[dict[str, Any]] | None,
    current_query: str | None = None,
) -> dict[str, Any] | None:
    current_normalized = _normalize_text(current_query).lower()
    skipped_current = not current_normalized
    for message in reversed(normalize_chat_history(chat_history)):
        if message.get("role") != "user":
            continue
        content_normalized = _normalize_text(message.get("content")).lower()
        if not skipped_current and content_normalized == current_normalized:
            skipped_current = True
            continue
        return message
    return None


def detect_requested_output_language(query: str) -> str | None:
    query_lower = _normalize_text(query).lower()
    if any(phrase in query_lower for phrase in ["both languages", "bilingual", "中英文", "双语"]):
        return "bilingual"
    if any(
        phrase in query_lower
        for phrase in ["answer in chinese", "reply in chinese", "respond in chinese", "用中文", "中文回答", "讲中文", "中文"]
    ):
        return "simplified_chinese"
    if any(
        phrase in query_lower
        for phrase in ["answer in english", "reply in english", "respond in english", "用英文", "英文回答"]
    ):
        return "english"
    return None


def detect_follow_up_style(query: str) -> str | None:
    query_lower = _normalize_text(query).lower()
    if any(
        phrase in query_lower
        for phrase in [
            "make that simpler",
            "make it simpler",
            "simpler",
            "simplify",
            "plain english",
            "shorter",
            "more direct",
            "simple version",
        ]
    ):
        return "simplify"
    why_follow_ups = {"why", "why?", "how so", "why though", "why is that", "and why", "explain why"}
    if query_lower in why_follow_ups or (query_lower.startswith("why ") and len(query_lower.split()) <= 6):
        return "why"
    if query_lower.startswith("what about") or query_lower.startswith("how about") or (" instead" in query_lower and len(query_lower.split()) <= 8):
        return "alternative"
    if detect_requested_output_language(query) is not None:
        return "language"
    return None


def _extract_known_topics(text: str | None) -> list[str]:
    normalized = _normalize_text(text)
    lowered = normalized.lower()
    matches: list[str] = []
    for term in _KNOWN_TOPIC_TERMS:
        if term.lower() in lowered and term not in matches:
            matches.append(term)
    return matches


def _query_looks_chinese(text: str | None) -> bool:
    return len(_CJK_CHAR_PATTERN.findall(text or "")) >= 2


def resolve_follow_up_query(
    query: str,
    chat_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_query = _normalize_text(query)
    style = detect_follow_up_style(normalized_query)
    previous_user = get_previous_user_message(chat_history, current_query=query)
    previous_assistant = get_last_assistant_message(chat_history)
    last_route = previous_assistant.get("route_label") if previous_assistant else None
    previous_user_text = _normalize_text((previous_user or {}).get("user_query") or (previous_user or {}).get("content"))
    previous_answer_text = _normalize_text((previous_assistant or {}).get("content"))
    previous_effective_query = _normalize_text((previous_assistant or {}).get("effective_query"))
    previous_topics = _extract_known_topics(previous_user_text)
    assistant_topics = _extract_known_topics(previous_answer_text)
    current_topics = _extract_known_topics(normalized_query)
    base_topic_text = previous_effective_query or previous_user_text

    if not style or (not previous_user and not previous_assistant):
        return {
            "effective_query": query,
            "is_follow_up": False,
            "style": style,
            "reason": None,
            "previous_route_label": last_route,
        }

    effective_query = query
    reason = None

    if style == "simplify" and base_topic_text:
        effective_query = f"{base_topic_text}\n\nPlease answer again in simpler and shorter language."
        reason = "The current query looks like a simplification follow-up, so the previous user question was reused as the topic."
    elif style == "language" and base_topic_text:
        language = detect_requested_output_language(query)
        if language == "simplified_chinese":
            effective_query = f"{base_topic_text}\n\nPlease answer again in Simplified Chinese 中文."
        elif language == "bilingual":
            effective_query = f"{base_topic_text}\n\nPlease answer again in both Chinese and English 中英文."
        else:
            effective_query = f"{base_topic_text}\n\nPlease answer again in English."
        reason = "The current query looks like a language follow-up, so the previous user question was reused as the topic."
    elif style == "why":
        if _is_recommendation_route(last_route):
            focus = assistant_topics[0] if assistant_topics else (previous_topics[0] if previous_topics else "that recommendation")
            effective_query = f"Why is {focus} the best fit for me right now based on my profile and goals?"
        elif last_route == "Spending rules":
            effective_query = "What do these spending patterns say about my cash habits and budget?"
        elif last_route == "Account summary":
            effective_query = "What do these account balances, liquidity levels, and net worth numbers mean for my financial position?"
        elif last_route == "Portfolio explanation":
            effective_query = "What does this portfolio positioning mean for diversification, risk, and my goals?"
        elif last_route == "Performance explanation":
            effective_query = "What mainly drove this month's portfolio return change?"
        elif last_route == "Market explanation":
            effective_query = "What does this market context mean for my investments?"
        elif previous_user_text:
            effective_query = f"Explain the reasoning behind this answer more directly: {previous_user_text}"
        reason = "The current query looks like a why-follow-up, so the previous topic was expanded into a full question."
    elif style == "alternative":
        alternative = current_topics[0] if current_topics else None
        if _is_recommendation_route(last_route) and alternative:
            effective_query = f"Based on my profile, what about {alternative} instead of the option you just recommended?"
        elif last_route == "Spending rules" and alternative:
            effective_query = f"How much did I spend on {alternative} instead?"
        elif previous_user_text and alternative:
            effective_query = f"What about {alternative} instead in relation to this earlier question: {previous_user_text}"
        reason = "The current query looks like an alternative follow-up, so the previous topic was kept and the new option was made explicit."

    is_follow_up = effective_query != query
    explicit_language = detect_requested_output_language(query)
    if is_follow_up and explicit_language is None and _query_looks_chinese(query):
        effective_query = f"请用中文回答：{effective_query}"
    return {
        "effective_query": effective_query,
        "is_follow_up": is_follow_up,
        "style": style,
        "reason": reason,
        "previous_route_label": last_route,
        "previous_user_query": previous_user_text or None,
    }


def looks_like_follow_up_query(query: str) -> bool:
    query_lower = _normalize_text(query).lower()
    follow_up_starters = [
        "what about",
        "how about",
        "and ",
        "what if",
        "then what",
        "that one",
        "those",
        "it",
        "this",
        "can you make",
        "make that",
        "answer in",
    ]
    has_pronoun_shortcut = re.fullmatch(r"(why|why\?|it|this|that|those)\b.*", query_lower) is not None
    has_short_why = query_lower in {"why", "why?", "why is that", "how so", "why though"} or (
        query_lower.startswith("why ") and len(query_lower.split()) <= 6
    )
    return (
        any(query_lower.startswith(starter) for starter in follow_up_starters)
        or has_pronoun_shortcut
        or has_short_why
        or len(query_lower.split()) <= 5
    )
