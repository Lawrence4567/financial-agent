from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from data_sources import ARTIFACTS_DIR


AUDIT_LOG_DIR = ARTIFACTS_DIR / "audit_logs"
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "app_audit.jsonl"


@dataclass(frozen=True)
class RequestMetadata:
    user_id: str
    session_id: str
    channel: str
    device: str
    timestamp: str
    sso_provider: str = "demo_local_login"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    authenticated: bool
    account_owner_verified: bool
    reason: str
    allowed_data_scope: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComplianceReport:
    status: str
    pii_redacted: bool
    softened_advice_terms: list[str]
    blocked_terms: list[str]
    final_answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_request_metadata(
    *,
    user_id: str | int,
    session_id: str | None = None,
    channel: str = "web_app",
    device: str = "browser",
    timestamp: str | None = None,
    sso_provider: str = "demo_local_login",
) -> RequestMetadata:
    return RequestMetadata(
        user_id=str(user_id),
        session_id=session_id or f"session-{uuid4().hex[:12]}",
        channel=channel,
        device=device,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        sso_provider=sso_provider,
    )


def enforce_identity_and_access(
    request_metadata: RequestMetadata,
    *,
    authorized_user_id: str | int,
) -> AccessDecision:
    expected_user_id = str(authorized_user_id)
    authenticated = bool(request_metadata.user_id and request_metadata.session_id)
    account_owner_verified = authenticated and request_metadata.user_id == expected_user_id
    allowed = authenticated and account_owner_verified

    if allowed:
        reason = (
            "Identity verified before AI reasoning. The request is limited to the current user's "
            "portfolio, accounts, transactions, and approved market/reference data."
        )
        allowed_scope = [
            "user_profile",
            "account_summary",
            "portfolio_holdings",
            "portfolio_performance",
            "transactions",
            "reference_knowledge",
            "market_context",
        ]
    elif not authenticated:
        reason = "The request is missing login or session details, so access is denied."
        allowed_scope = []
    else:
        reason = "The authenticated user does not match the portfolio owner in the loaded demo case."
        allowed_scope = []

    return AccessDecision(
        allowed=allowed,
        authenticated=authenticated,
        account_owner_verified=account_owner_verified,
        reason=reason,
        allowed_data_scope=allowed_scope,
    )


def _soften_advice_language(answer: str) -> tuple[str, list[str], list[str]]:
    softened_terms: list[str] = []
    blocked_terms: list[str] = []
    protected_terms = {
        "__PROTECTED_GIC_SINGULAR__": "Guaranteed Investment Certificate",
        "__PROTECTED_GIC_PLURAL__": "Guaranteed Investment Certificates",
    }
    updates = [
        (r"\byou should buy\b", "you may consider reviewing"),
        (r"\byou should sell\b", "you may consider reviewing"),
        (r"\byou should\b", "you may consider"),
        (r"\byou must\b", "you may want to"),
        (r"\bdefinitely buy\b", "review carefully"),
        (r"\bwill definitely\b", "may"),
        (r"\bguaranteed\b", "not guaranteed"),
        (r"\brisk-free\b", "lower-risk"),
        (r"\bcan(?:not|'t) lose\b", "still carries risk"),
    ]

    updated_answer = answer
    for token, original in protected_terms.items():
        updated_answer = re.sub(re.escape(original), token, updated_answer, flags=re.IGNORECASE)
    for pattern, replacement in updates:
        if re.search(pattern, updated_answer, flags=re.IGNORECASE):
            updated_answer = re.sub(pattern, replacement, updated_answer, flags=re.IGNORECASE)
            lowered = pattern.replace("\\b", "")
            if "guaranteed" in lowered or "risk-free" in lowered or "lose" in lowered:
                blocked_terms.append(lowered)
            else:
                softened_terms.append(lowered)

    for token, original in protected_terms.items():
        updated_answer = updated_answer.replace(token, original)

    return updated_answer, softened_terms, blocked_terms


def _redact_known_pii(answer: str, user_profile: dict[str, Any] | None) -> tuple[str, bool]:
    if not user_profile:
        return answer, False

    redacted = answer
    pii_found = False
    for key in ["name", "email", "contact"]:
        value = user_profile.get(key)
        if value and str(value) in redacted:
            redacted = redacted.replace(str(value), "[redacted]")
            pii_found = True
    return redacted, pii_found


def apply_response_guardrails(
    answer: str,
    *,
    user_profile: dict[str, Any] | None = None,
) -> ComplianceReport:
    pii_safe_answer, pii_redacted = _redact_known_pii(answer, user_profile)
    softened_answer, softened_terms, blocked_terms = _soften_advice_language(pii_safe_answer)
    status = "passed_with_adjustments" if pii_redacted or softened_terms or blocked_terms else "passed"
    return ComplianceReport(
        status=status,
        pii_redacted=pii_redacted,
        softened_advice_terms=softened_terms,
        blocked_terms=blocked_terms,
        final_answer=softened_answer,
    )


def write_audit_log(entry: dict[str, Any]) -> dict[str, Any]:
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    event_id = f"audit-{uuid4().hex}"
    payload = {
        "event_id": event_id,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    with AUDIT_LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {
        "event_id": event_id,
        "path": str(AUDIT_LOG_FILE),
    }


def audit_log_path() -> str:
    return str(AUDIT_LOG_FILE)
