from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency for external market data
    yf = None


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
REPO_ROOT = PROJECT_DIR.parent


def load_project_dotenv() -> Optional[Path]:
    # Search the most likely beginner-friendly locations in a predictable order.
    candidates = [
        REPO_ROOT / ".env",
        PROJECT_DIR / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate)
            return candidate
    load_dotenv()
    return None


ENV_FILE_USED = load_project_dotenv()

DATA_DIR_ENV = os.getenv("FINANCE_DATA_DIR", "../data/artifacts_canada")
ARTIFACTS_DIR = Path(DATA_DIR_ENV)
if not ARTIFACTS_DIR.is_absolute():
    ARTIFACTS_DIR = APP_DIR / ARTIFACTS_DIR

REFERENCE_DIR = PROJECT_DIR / "data" / "reference_canada"


@dataclass
class FinancialContext:
    transactions: pd.DataFrame
    monthly: pd.DataFrame
    user_info: pd.DataFrame
    product_catalog: pd.DataFrame
    account_summary: pd.DataFrame
    portfolio_holdings: pd.DataFrame
    portfolio_performance: pd.DataFrame
    reference_knowledge: dict[str, Any]
    market_context: dict[str, Any]
    market_commentary: dict[str, Any]
    source_overview: dict[str, str]


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_monthly_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    debit_rows = transactions[transactions["type"].str.upper() == "DR"].copy()
    debit_rows["year"] = debit_rows["date"].dt.year
    debit_rows["month"] = debit_rows["date"].dt.month
    return (
        debit_rows.groupby(["year", "month", "category"], dropna=False)["amount"]
        .sum()
        .reset_index()
        .sort_values(["year", "month", "category"])
    )


def load_transactions() -> pd.DataFrame:
    transactions = pd.read_csv(ARTIFACTS_DIR / "cat.csv")
    transactions["amount"] = pd.to_numeric(transactions["amount"], errors="coerce").fillna(0.0)
    transactions["date"] = pd.to_datetime(transactions["date"], format="%d-%m-%Y", errors="coerce")
    return transactions


def load_user_info() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "user_info.csv")


def load_product_catalog() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "product_catalog.csv")


def load_account_summary() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "account_summary.csv")


def load_portfolio_holdings() -> pd.DataFrame:
    holdings = pd.read_csv(ARTIFACTS_DIR / "portfolio_holdings.csv")
    numeric_columns = ["market_value", "cost_basis", "weight", "one_month_return_pct", "ytd_return_pct"]
    for column in numeric_columns:
        holdings[column] = pd.to_numeric(holdings[column], errors="coerce").fillna(0.0)
    return holdings


def load_portfolio_performance() -> pd.DataFrame:
    performance = pd.read_csv(ARTIFACTS_DIR / "portfolio_performance.csv")
    numeric_columns = ["starting_value", "net_contributions", "market_impact", "income", "fees", "ending_value", "monthly_return_pct"]
    for column in numeric_columns:
        performance[column] = pd.to_numeric(performance[column], errors="coerce").fillna(0.0)
    return performance


def load_reference_knowledge() -> dict[str, Any]:
    return {
        "account_knowledge": _load_json_file(REFERENCE_DIR / "account_knowledge.json"),
        "planning_guidance": _load_json_file(REFERENCE_DIR / "planning_guidance.json"),
        "official_account_rules": _load_json_file(REFERENCE_DIR / "official_account_rules.json"),
    }


def load_market_context() -> dict[str, Any]:
    return _load_json_file(REFERENCE_DIR / "market_context.json")


def load_market_commentary() -> dict[str, Any]:
    return _load_json_file(REFERENCE_DIR / "market_commentary.json")


def _normalize_watchlist_item(item: dict[str, Any]) -> dict[str, Any]:
    yahoo_symbol = item.get("yahoo_symbol") or item.get("symbol")
    return {
        "symbol": item.get("symbol", yahoo_symbol),
        "yahoo_symbol": yahoo_symbol,
        "label": item.get("label", yahoo_symbol),
        "why_it_matters": item.get("why_it_matters", ""),
        "currency": item.get("currency", "CAD"),
    }


def load_market_watchlist() -> list[dict[str, Any]]:
    market_context = load_market_context()
    return [
        _normalize_watchlist_item(item)
        for item in market_context.get("watchlist", market_context.get("starter_watchlist", []))
    ]


def _extract_ticker_history(download_frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if download_frame.empty:
        return pd.DataFrame()

    if isinstance(download_frame.columns, pd.MultiIndex):
        first_level = download_frame.columns.get_level_values(0)
        if ticker in first_level:
            return download_frame[ticker].dropna(how="all")
        return pd.DataFrame()

    return download_frame.dropna(how="all")


def fetch_market_snapshot(period: str = "5d", interval: str = "1d", timeout: int = 8) -> dict[str, Any]:
    watchlist = load_market_watchlist()
    provider = "Yahoo Finance via yfinance"

    if not watchlist:
        return {
            "status": "watchlist_empty",
            "provider": provider,
            "quotes": [],
            "message": "No ETF watchlist is configured yet.",
        }

    if yf is None:
        return {
            "status": "dependency_missing",
            "provider": provider,
            "quotes": [],
            "watchlist": watchlist,
            "message": "Install yfinance to enable the external ETF snapshot.",
        }

    tickers = [item["yahoo_symbol"] for item in watchlist if item.get("yahoo_symbol")]
    try:
        download_frame = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - network-dependent branch
        return {
            "status": "error",
            "provider": provider,
            "quotes": [],
            "watchlist": watchlist,
            "message": "The Yahoo Finance request did not succeed right now.",
            "error": str(exc),
        }

    if download_frame is None or download_frame.empty:
        return {
            "status": "empty",
            "provider": provider,
            "quotes": [],
            "watchlist": watchlist,
            "message": "Yahoo Finance returned no ETF rows for the current watchlist.",
        }

    quotes: list[dict[str, Any]] = []
    for item in watchlist:
        ticker_history = _extract_ticker_history(download_frame, item["yahoo_symbol"])
        if ticker_history.empty or "Close" not in ticker_history.columns:
            continue

        closes = ticker_history["Close"].dropna()
        if closes.empty:
            continue

        last_close = float(closes.iloc[-1])
        previous_close = float(closes.iloc[-2]) if len(closes) > 1 else last_close
        first_close = float(closes.iloc[0])
        day_change = last_close - previous_close
        day_change_pct = (day_change / previous_close * 100) if previous_close else 0.0
        period_change_pct = ((last_close - first_close) / first_close * 100) if first_close else 0.0

        quotes.append(
            {
                "symbol": item["symbol"],
                "yahoo_symbol": item["yahoo_symbol"],
                "label": item["label"],
                "currency": item.get("currency", "CAD"),
                "last_close": round(last_close, 2),
                "day_change": round(day_change, 2),
                "day_change_pct": round(day_change_pct, 2),
                "period_change_pct": round(period_change_pct, 2),
                "why_it_matters": item.get("why_it_matters", ""),
            }
        )

    latest_index = pd.to_datetime(download_frame.index.max()) if len(download_frame.index) else None
    as_of = latest_index.strftime("%Y-%m-%d") if latest_index is not None else "unknown"

    return {
        "status": "live" if quotes else "empty",
        "provider": provider,
        "quotes": quotes,
        "watchlist": watchlist,
        "as_of": as_of,
        "message": "Live ETF snapshot loaded successfully." if quotes else "No ETF prices were available in the latest response.",
    }


def load_financial_context() -> FinancialContext:
    transactions = load_transactions()
    monthly_path = ARTIFACTS_DIR / "monthly_analysis.csv"
    monthly = pd.read_csv(monthly_path) if monthly_path.exists() else build_monthly_summary(transactions)

    return FinancialContext(
        transactions=transactions,
        monthly=monthly,
        user_info=load_user_info(),
        product_catalog=load_product_catalog(),
        account_summary=load_account_summary(),
        portfolio_holdings=load_portfolio_holdings(),
        portfolio_performance=load_portfolio_performance(),
        reference_knowledge=load_reference_knowledge(),
        market_context=load_market_context(),
        market_commentary=load_market_commentary(),
        source_overview={
            "user_data": str(ARTIFACTS_DIR),
            "product_data": str(ARTIFACTS_DIR / "product_catalog.csv"),
            "account_data": str(ARTIFACTS_DIR / "account_summary.csv"),
            "portfolio_data": str(ARTIFACTS_DIR / "portfolio_holdings.csv"),
            "performance_data": str(ARTIFACTS_DIR / "portfolio_performance.csv"),
            "reference_data": str(REFERENCE_DIR),
            "market_data": str(REFERENCE_DIR / "market_context.json"),
            "market_commentary": str(REFERENCE_DIR / "market_commentary.json"),
            "external_market_data": "Yahoo Finance via yfinance (optional runtime fetch)",
        },
    )
