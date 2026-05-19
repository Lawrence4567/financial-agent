from __future__ import annotations

import re


SPENDING_ROUTE_TERMS = {
    "spend",
    "spent",
    "spending",
    "expense",
    "expenses",
    "habit",
    "habits",
    "cash habit",
    "cash habits",
    "spending habit",
    "spending habits",
    "money habit",
    "money habits",
    "budget",
    "cash flow",
    "cashflow",
    "cost",
    "costs",
    "pay",
    "paid",
    "payment",
    "payments",
    "total",
    "花",
    "花了",
    "花费",
    "消费",
    "支出",
    "多少钱",
    "这个月",
    "本月",
    "最近",
}

NEGATION_HINTS = {
    "not",
    "dont",
    "don't",
    "do not",
    "did not",
    "didn't",
    "exclude",
    "excluding",
    "except",
    "other than",
    "outside",
    "non",
}

SPENDING_CATEGORY_GROUPS = {
    "Food": {
        "categories": ["Groceries", "Dining"],
        "aliases": [
            "food",
            "meal",
            "meals",
            "eat",
            "eating",
        ],
    },
    "Groceries": {
        "categories": ["Groceries"],
        "aliases": ["groceries", "grocery", "supermarket"],
    },
    "Dining": {
        "categories": ["Dining"],
        "aliases": ["dining", "restaurant", "restaurants", "takeout", "take-out", "ubereats", "uber eats"],
    },
    "Rent": {
        "categories": ["Rent"],
        "aliases": ["rent", "housing", "apartment", "condo"],
    },
    "Transportation": {
        "categories": ["Transportation"],
        "aliases": ["transport", "transportation", "transit", "ttc", "commute", "commuting"],
    },
    "Utilities": {
        "categories": ["Utilities"],
        "aliases": ["utilities", "hydro", "electricity", "gas bill", "utility"],
    },
    "Telecom": {
        "categories": ["Telecom"],
        "aliases": ["phone", "internet", "telecom", "mobile"],
    },
    "Travel": {
        "categories": ["Travel"],
        "aliases": ["travel", "trip", "flights", "flight"],
    },
}


def _contains_alias(query_lower: str, alias: str) -> bool:
    return re.search(rf"\b{re.escape(alias.lower())}\b", query_lower) is not None


def normalize_query_text(query: str) -> str:
    query_lower = " ".join(str(query).lower().split())
    expansions: list[str] = []

    if any(_contains_alias(query_lower, term) for term in ["spent", "pay", "paid", "cost", "costs", "payment", "payments"]):
        expansions.extend(["spend", "spending", "expense"])
    if any(term in query_lower for term in ["花", "花了", "花费", "消费", "支出", "多少钱"]):
        expansions.extend(["spend", "spending", "expense", "total"])
    if any(_contains_alias(query_lower, term) for term in ["habit", "habits"]):
        if any(_contains_alias(query_lower, term) for term in ["cash", "money", "budget", "spending"]):
            expansions.extend(["spending", "cash flow", "budget"])

    category_match = extract_spending_category_match(query)
    if category_match:
        expansions.extend(["spending", "expense"])
        expansions.extend(alias.lower() for alias in category_match["categories"])
        if category_match["label"].lower() == "food":
            expansions.extend(["food", "groceries", "dining"])
        spending_scope = extract_spending_scope(query, category_match=category_match)
        if spending_scope and spending_scope["mode"] == "exclude":
            expansions.extend(["excluding", "other than", "non"])

    return " ".join(part for part in [query_lower, *expansions] if part)


def extract_spending_category_match(query: str) -> dict | None:
    query_lower = " ".join(str(query).lower().split())

    if any(_contains_alias(query_lower, alias) for alias in SPENDING_CATEGORY_GROUPS["Food"]["aliases"]):
        return {
            "label": "Food",
            "categories": SPENDING_CATEGORY_GROUPS["Food"]["categories"],
        }

    matched_labels: list[str] = []
    matched_categories: list[str] = []
    for label, config in SPENDING_CATEGORY_GROUPS.items():
        if label == "Food":
            continue
        if any(_contains_alias(query_lower, alias) for alias in config["aliases"]):
            matched_labels.append(label)
            matched_categories.extend(config["categories"])

    unique_categories = list(dict.fromkeys(matched_categories))
    if not unique_categories:
        return None
    if set(unique_categories) == {"Groceries", "Dining"}:
        return {
            "label": "Food",
            "categories": ["Groceries", "Dining"],
        }
    if len(matched_labels) == 1:
        return {
            "label": matched_labels[0],
            "categories": unique_categories,
        }
    return {
        "label": ", ".join(matched_labels),
        "categories": unique_categories,
    }


def is_negative_spending_filter(query: str, category_match: dict | None = None) -> bool:
    query_lower = " ".join(str(query).lower().split())
    category_match = category_match or extract_spending_category_match(query)
    if category_match is None:
        return False

    has_negation = any(hint in query_lower for hint in NEGATION_HINTS)
    has_exclusion_phrase = any(
        phrase in query_lower
        for phrase in [
            "not spend on",
            "not spending on",
            "dont spend on",
            "don't spend on",
            "do not spend on",
            "except",
            "excluding",
            "other than",
            "outside of",
            "outside",
            "non-food",
            "non food",
        ]
    )
    return has_negation and has_exclusion_phrase


def extract_spending_scope(query: str, category_match: dict | None = None) -> dict | None:
    category_match = category_match or extract_spending_category_match(query)
    if category_match is None:
        return None

    mode = "exclude" if is_negative_spending_filter(query, category_match=category_match) else "include"
    return {
        "mode": mode,
        "label": category_match["label"],
        "categories": category_match["categories"],
    }


def is_semantic_spending_query(query: str) -> bool:
    normalized_query = normalize_query_text(query)

    if extract_spending_category_match(query) is not None:
        return True

    if any(term in normalized_query for term in SPENDING_ROUTE_TERMS):
        return True

    amount_patterns = [
        r"how much .* spent",
        r"how much .* spend",
        r"how much .* pay",
        r"what did i spend",
        r"what did i pay",
        r"money on",
        r"cash habits?",
        r"spending habits?",
        r"money habits?",
        r"understand .* cash habits",
    ]
    return any(re.search(pattern, normalized_query) for pattern in amount_patterns)
