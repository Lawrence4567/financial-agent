from __future__ import annotations

from typing import Any

from query_understanding import normalize_query_text


def _round(value: float) -> float:
    return round(float(value), 2)


def _weight_percent(part: float, whole: float) -> float:
    if not whole:
        return 0.0
    return round(float(part) / float(whole) * 100, 2)


def analyze_portfolio_performance_tools(summary: dict, query: str) -> dict[str, Any]:
    portfolio = summary.get("portfolio_overview", {})
    holdings = portfolio.get("top_holdings", [])
    history = portfolio.get("performance_history", [])
    latest_month = portfolio.get("latest_month", {})
    previous_month = portfolio.get("previous_month", {})
    query_lower = normalize_query_text(query)

    negative_months = [
        row for row in history
        if float(row.get("monthly_return_pct", 0.0)) < 0
    ]
    most_recent_negative_month = negative_months[-1] if negative_months else None
    latest_return = float(latest_month.get("monthly_return_pct", 0.0)) if latest_month else 0.0

    focus_month = latest_month
    focus_reason = "latest_available_month"
    user_expected_loss = any(
        term in query_lower
        for term in ["go down", "went down", "down this month", "portfolio down", "why did my portfolio go down"]
    )
    if user_expected_loss and latest_return >= 0 and most_recent_negative_month:
        focus_month = most_recent_negative_month
        focus_reason = "most_recent_negative_month_used_because_latest_month_is_positive"

    focus_month = focus_month or {}
    focus_month_name = focus_month.get("month", latest_month.get("month", "unknown"))
    focus_index = next(
        (
            index
            for index, item in enumerate(history)
            if item.get("month") == focus_month_name
        ),
        len(history) - 1,
    )
    focus_previous_month = history[focus_index - 1] if focus_index > 0 else previous_month
    previous_return = float(focus_previous_month.get("monthly_return_pct", 0.0)) if focus_previous_month else 0.0
    focus_return = float(focus_month.get("monthly_return_pct", 0.0))
    market_impact = float(focus_month.get("market_impact", 0.0))
    income = float(focus_month.get("income", 0.0))
    fees = float(focus_month.get("fees", 0.0))
    net_investment_result = market_impact + income - fees
    starting_value = float(focus_month.get("starting_value", 0.0))

    total_market_value = float(portfolio.get("total_market_value", 0.0))
    asset_class_weights = {
        asset_class: _weight_percent(value, total_market_value)
        for asset_class, value in portfolio.get("asset_mix", {}).items()
    }
    region_weights = {
        region: _weight_percent(value, total_market_value)
        for region, value in portfolio.get("region_mix", {}).items()
    }

    holding_contribution_proxy = []
    for item in holdings:
        weight_pct = _weight_percent(item.get("market_value", 0.0), total_market_value)
        return_pct = float(item.get("one_month_return_pct", 0.0))
        contribution_proxy = round(weight_pct * return_pct / 100, 2)
        holding_contribution_proxy.append(
            {
                "symbol": item.get("symbol"),
                "holding_name": item.get("holding_name"),
                "weight_pct": weight_pct,
                "one_month_return_pct": round(return_pct, 2),
                "contribution_proxy_pct_points": contribution_proxy,
                "region": item.get("region"),
                "asset_class": item.get("asset_class"),
            }
        )

    holding_contribution_proxy.sort(
        key=lambda item: item["contribution_proxy_pct_points"],
        reverse=True,
    )

    monthly_returns = [float(item.get("monthly_return_pct", 0.0)) for item in history]
    negative_return_count = sum(1 for value in monthly_returns if value < 0)
    volatility_summary = {
        "months_observed": len(monthly_returns),
        "negative_months": negative_return_count,
        "worst_month_return_pct": round(min(monthly_returns), 2) if monthly_returns else 0.0,
        "best_month_return_pct": round(max(monthly_returns), 2) if monthly_returns else 0.0,
        "average_absolute_monthly_move_pct": round(
            sum(abs(value) for value in monthly_returns) / len(monthly_returns), 2
        ) if monthly_returns else 0.0,
    }

    driver_assessment = [
        f"Market impact was {market_impact:,.2f} in {focus_month_name}, which was the main direct driver of that month's result.",
        (
            "Equity exposure remains the biggest return engine because equity ETFs are the largest asset-class weight."
            if asset_class_weights.get("Equity ETF", 0.0) >= 50
            else "The portfolio is relatively balanced, so no single asset class dominates the full risk picture."
        ),
        (
            "U.S. exposure is meaningful, so broader U.S. equity sentiment can move month-to-month performance."
            if region_weights.get("United States", 0.0) >= 20
            else "U.S. exposure is present but not the only regional driver."
        ),
    ]
    if any(item.get("symbol") == "XRE" for item in holdings):
        driver_assessment.append(
            "The real-estate sleeve is small, but it can still drag returns in rate-sensitive risk-off months."
        )

    return {
        "tool_name": "portfolio_performance_toolkit",
        "focus_month": focus_month_name,
        "focus_reason": focus_reason,
        "latest_loaded_month": latest_month.get("month"),
        "latest_loaded_return_pct": round(latest_return, 2),
        "selected_month_metrics": {
            "starting_value": _round(starting_value),
            "net_contributions": _round(float(focus_month.get("net_contributions", 0.0))),
            "market_impact": _round(market_impact),
            "income": _round(income),
            "fees": _round(fees),
            "ending_value": _round(float(focus_month.get("ending_value", 0.0))),
            "monthly_return_pct": _round(focus_return),
            "net_investment_result": _round(net_investment_result),
            "change_vs_previous_month_pct_points": _round(focus_return - previous_return),
            "primary_driver": focus_month.get("primary_driver", ""),
        },
        "exposure_breakdown": {
            "asset_class_weights_pct": asset_class_weights,
            "region_weights_pct": region_weights,
        },
        "top_holding_contribution_proxy": holding_contribution_proxy[:5],
        "volatility_analysis": volatility_summary,
        "driver_assessment": driver_assessment,
    }
