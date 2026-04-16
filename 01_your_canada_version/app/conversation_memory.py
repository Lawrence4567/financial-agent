from __future__ import annotations

from typing import Any


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
                }
            )
        elif isinstance(message, (list, tuple)) and len(message) >= 2:
            normalized.append(
                {
                    "role": str(message[0]),
                    "content": str(message[1]),
                    "route_label": None,
                    "mode": None,
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


def looks_like_follow_up_query(query: str) -> bool:
    query_lower = " ".join(str(query).lower().split())
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
    ]
    return any(query_lower.startswith(starter) for starter in follow_up_starters) or len(query_lower.split()) <= 5
