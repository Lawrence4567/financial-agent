from __future__ import annotations

import os

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
except ImportError:  # pragma: no cover - optional dependency
    ChatPromptTemplate = None
    RunnableLambda = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


class LangChainAdapterUnavailable(RuntimeError):
    """Raised when the optional LangChain adapter is requested but unavailable."""


def should_use_langchain_backend() -> bool:
    return os.getenv("LLM_BACKEND", "langchain").strip().lower() in {"langchain", "langchain_core"}


def langchain_backend_available() -> bool:
    return ChatPromptTemplate is not None and RunnableLambda is not None and OpenAI is not None


def _normalize_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _chat_prompt_value_to_openai_input(prompt_value: object) -> str:
    messages = getattr(prompt_value, "to_messages", lambda: [])()
    rendered = []
    for message in messages:
        role = getattr(message, "type", "message").upper()
        content = _normalize_message_content(getattr(message, "content", ""))
        rendered.append(f"{role}:\n{content}")
    return "\n\n".join(rendered)


def _call_openai_text(input_text: str, api_key: str, model: str) -> str:
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=input_text,
    )
    return response.output_text.strip()


def invoke_langchain_text(*, instructions: str, user_input: str, api_key: str, model: str) -> str:
    if not langchain_backend_available():
        raise LangChainAdapterUnavailable(
            "LLM_BACKEND=langchain was requested, but `langchain-core` is not installed in this environment."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{instructions}"),
            ("human", "{user_input}"),
        ]
    )
    chain = (
        prompt
        | RunnableLambda(_chat_prompt_value_to_openai_input)
        | RunnableLambda(lambda prompt_text: _call_openai_text(prompt_text, api_key=api_key, model=model))
    )
    return chain.invoke(
        {
            "instructions": instructions,
            "user_input": user_input,
        }
    )
