from __future__ import annotations

from typing import Any


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def detect_reason_tags(user_profile: dict, net_cash_flow: float) -> list[str]:
    goal_text = " ".join(
        str(user_profile.get(field, ""))
        for field in ["financial_goals", "short_term_goals", "long_term_goal"]
    ).lower()
    preference_text = str(user_profile.get("preferred_investment_types", "")).lower()
    risk_tolerance = str(user_profile.get("risk_tolerance", "")).lower()
    investment_experience = str(user_profile.get("investment_experience", "")).lower()
    annual_income = float(user_profile.get("annual_income") or 0)

    tags: list[str] = []

    if _contains_any(goal_text, ["first home", "first condo", "down payment", "home"]):
        tags.append("first_home_goal")
    if _contains_any(goal_text, ["emergency fund", "cash reserve", "cash reserves", "short-term"]):
        tags.append("emergency_fund_goal")
    if _contains_any(goal_text, ["retirement", "financial independence"]):
        tags.append("retirement_goal")
    if "etf" in preference_text:
        tags.append("etf_preference")
    if risk_tolerance == "medium":
        tags.append("medium_risk")
    if risk_tolerance == "high":
        tags.append("high_risk")
    if investment_experience in {"beginner", "novice"}:
        tags.append("needs_simple_investing")
    if annual_income >= 85000:
        tags.append("rrsp_tax_benefit_relevant")
    if net_cash_flow > 0:
        tags.append("positive_cash_flow")

    return tags


def _priority_from_score(score: int) -> str:
    if score >= 85:
        return "High"
    if score >= 60:
        return "Medium"
    if score >= 35:
        return "Conditional"
    return "Later"


def _build_catalog_map(product_catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        item["category"]: item
        for item in product_catalog
    }


def score_product_recommendations(
    *,
    user_profile: dict,
    product_catalog: list[dict[str, Any]],
    reference_knowledge: dict[str, Any],
    priority_order: list[str],
    priority_reason: str,
    net_cash_flow: float,
) -> dict[str, Any]:
    tags = detect_reason_tags(user_profile, net_cash_flow)
    catalog_map = _build_catalog_map(product_catalog)
    account_map = {
        account["category"]: account
        for account in reference_knowledge.get("account_knowledge", {}).get("accounts", [])
    }

    categories = [
        "FHSA",
        "High-Interest Savings",
        "TFSA",
        "RRSP",
        "GIC",
        "ETF Investing",
        "Robo-Advising",
        "Chequing",
    ]
    recommendations: list[dict[str, Any]] = []

    for category in categories:
        if category not in catalog_map:
            continue

        score = 0
        reasons: list[str] = []
        reason_tags: list[str] = []

        def add(points: int, reason: str, tag: str | None = None) -> None:
            nonlocal score
            score += points
            reasons.append(reason)
            if tag and tag not in reason_tags:
                reason_tags.append(tag)

        if category in priority_order:
            add(34 - (priority_order.index(category) * 8), f"Local planning rules place {category} near the top for this goal mix.")

        if category == "FHSA":
            add(12, "This account is directly linked to first-home saving.")
            if "first_home_goal" in tags:
                add(48, "Your profile includes a first-home or condo down-payment goal.", "first_home_goal")
            if "positive_cash_flow" in tags:
                add(8, "Positive cash flow gives room for regular FHSA contributions.", "positive_cash_flow")

        if category == "High-Interest Savings":
            add(10, "This is a strong core product for liquid cash savings.")
            if "emergency_fund_goal" in tags:
                add(48, "Your profile highlights an emergency fund or cash reserve goal.", "emergency_fund_goal")
            if "positive_cash_flow" in tags:
                add(8, "Positive cash flow can be directed into cash reserves consistently.", "positive_cash_flow")
            if "first_home_goal" in tags:
                add(14, "Liquid savings can support a near-term home down-payment plan.", "first_home_goal")

        if category == "TFSA":
            add(20, "TFSA stays flexible across cash savings and investing.")
            if "positive_cash_flow" in tags:
                add(10, "Positive cash flow supports ongoing TFSA contributions.", "positive_cash_flow")
            if "emergency_fund_goal" in tags:
                add(8, "TFSA can support flexible medium-term saving after core cash reserves are set.", "emergency_fund_goal")
            if "etf_preference" in tags:
                add(12, "Your profile already mentions ETFs as a preferred investment type.", "etf_preference")
            if "retirement_goal" in tags:
                add(10, "TFSA also supports long-term wealth building.", "retirement_goal")

        if category == "RRSP":
            add(8, "RRSP is useful for retirement-focused long-term saving.")
            if "retirement_goal" in tags:
                add(38, "Your long-term goals include retirement or financial independence.", "retirement_goal")
            if "rrsp_tax_benefit_relevant" in tags:
                add(15, "Your income level makes the RRSP tax deduction more relevant.", "rrsp_tax_benefit_relevant")
            if "emergency_fund_goal" in tags:
                add(-14, "RRSP is usually less flexible than cash products for short-term reserves.")
            if "first_home_goal" in tags:
                add(-8, "FHSA is usually the more direct first-home vehicle before RRSP becomes the next focus.")

        if category == "GIC":
            add(5, "GIC can be useful for short or medium-term money that needs stability.")
            if "first_home_goal" in tags:
                add(8, "Part of a down-payment plan can fit a stable product if the timeline is clear.", "first_home_goal")
            if "emergency_fund_goal" in tags:
                add(-18, "Emergency money usually needs liquidity, which many GICs do not provide.")
            if "positive_cash_flow" in tags:
                add(5, "Extra surplus cash can support laddered savings products.", "positive_cash_flow")

        if category == "ETF Investing":
            add(6, "ETF investing is designed for longer-term growth.")
            if "etf_preference" in tags:
                add(28, "Your profile explicitly lists ETFs as a preferred investment type.", "etf_preference")
            if "retirement_goal" in tags:
                add(20, "Long-term retirement goals fit diversified ETF investing.", "retirement_goal")
            if "medium_risk" in tags or "high_risk" in tags:
                add(12, "Your risk tolerance supports some market exposure.", "medium_risk" if "medium_risk" in tags else "high_risk")
            if "emergency_fund_goal" in tags:
                add(-12, "Emergency savings should usually be built before adding more market risk.")
            if "first_home_goal" in tags:
                add(-6, "Near-term home savings often matter before adding more long-term market exposure.")

        if category == "Robo-Advising":
            add(4, "Managed ETF portfolios can simplify diversified investing.")
            if "needs_simple_investing" in tags:
                add(18, "A simpler investing experience can be helpful for beginners.", "needs_simple_investing")
            if "medium_risk" in tags:
                add(10, "A medium-risk profile can fit a balanced managed portfolio.", "medium_risk")
            if "retirement_goal" in tags:
                add(8, "Managed investing can still support long-term goals.", "retirement_goal")

        if category == "Chequing":
            add(10, "Chequing remains useful as the operating account for payroll and bills.")
            if net_cash_flow > 0:
                add(3, "Stable payroll inflows support a basic daily banking setup.", "positive_cash_flow")

        account_reference = account_map.get(category, {})
        watchouts = account_reference.get("watchouts", [])[:2]
        recommendations.append(
            {
                "category": category,
                "product_name": catalog_map[category]["product_name"],
                "priority": _priority_from_score(score),
                "score": score,
                "best_for": catalog_map[category].get("best_for"),
                "description": catalog_map[category].get("description"),
                "reasons": reasons[:3],
                "reason_tags": reason_tags,
                "watchouts": watchouts,
            }
        )

    recommendations.sort(
        key=lambda item: (
            item["score"],
            item["category"] in priority_order,
        ),
        reverse=True,
    )

    return {
        "reason_tags": tags,
        "priority_reason": priority_reason,
        "recommendations": recommendations,
    }
