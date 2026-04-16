# Canada Workspace Map

This folder contains the Canada-focused advisory workspace that is designed for your portfolio and interview story.

## app/

Contains the runnable application code.

- `app_local.py`: Streamlit UI and page layout
- `local_financial_qa.py`: summary building, routing, tool execution, response generation, compliance, and audit logging
- `data_sources.py`: data source layer for transactions, accounts, holdings, performance, and market context
- `demo_governance.py`: request metadata, identity/access checks, compliance guardrails, and audit logging
- `analytics_tools.py`: portfolio performance attribution, exposure breakdown, and volatility analysis
- `recommendation_engine.py`: deterministic product scoring and recommendation ranking
- `query_router.py`: route classification
- `response_orchestrator.py`: fetch RAG context and market data only when needed
- `rag_pipeline.py`: local chunking, embeddings, and retrieval

## data/

Contains the client-case data and reference content.

- `artifacts_canada/user_info.csv`: household profile
- `artifacts_canada/cat.csv`: transaction history
- `artifacts_canada/account_summary.csv`: account balances and liquidity profile
- `artifacts_canada/portfolio_holdings.csv`: current holdings and asset mix
- `artifacts_canada/portfolio_performance.csv`: monthly portfolio performance history
- `artifacts_canada/product_catalog.csv`: representative Canadian banking and investing categories
- `reference_canada/account_knowledge.json`: account and product reference knowledge
- `reference_canada/planning_guidance.json`: planning rules and budgeting guidance
- `reference_canada/official_account_rules.json`: official-rule style finance summaries
- `reference_canada/market_context.json`: watchlist and market-layer configuration
- `reference_canada/market_commentary.json`: market narrative and monthly explanation notes
- `artifacts_canada/reference_rag_index.json`: generated local RAG index
- `artifacts_canada/audit_logs/`: request audit traces

## docs/

Contains local setup notes and design references.

- `README_LOCAL.md`: how to run the local advisory workspace
- `PROJECT_MAP_CANADA.md`: this file

## Recommended reading order

1. `app/app_local.py`
2. `app/local_financial_qa.py`
3. `app/demo_governance.py`
4. `app/analytics_tools.py`
5. `app/query_router.py`
6. `app/rag_pipeline.py`
7. `data/artifacts_canada/portfolio_performance.csv`
8. `data/reference_canada/market_commentary.json`
