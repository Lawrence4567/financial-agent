# Canada Workspace Map

Use this file as a quick directory guide for `01_your_canada_version/`.

For project overview and local setup, go back to the root `README.md`.
For detailed architecture, read `SYSTEM_ARCHITECTURE.md`.

## Workspace Layout

### `app/`

Runnable application code:

- `app_local.py`: Streamlit UI and page layout
- `local_financial_qa.py`: main controller for conversation resolution, intent/tool planning, evidence validation, generation, compliance, and audit logging
- `intent_engine.py`: LLM-first structured intent and tool planning with rules fallback
- `data_sources.py`: data loading and market snapshot access
- `demo_governance.py`: internal request records, access checks, guardrails, and audit logging
- `analytics_tools.py`: portfolio attribution, exposure, and volatility analysis
- `recommendation_engine.py`: deterministic product scoring and ranking
- `query_router.py`: rules fallback and legacy route compatibility
- `response_orchestrator.py`: load retrieval and market context only when the capability plan needs them
- `retrieval_backend.py`: retrieval abstraction layer with a local-index backend
- `rag_pipeline.py`: local chunking, embeddings, index build, and retrieval
- `langgraph_flow.py`: constrained workflow orchestration path
- `prompt_builder.py`: structured prompt assembly

### `data/artifacts_canada/`

Operational and generated data:

- `user_info.csv`: household profile
- `cat.csv`: transaction history
- `account_summary.csv`: balances and liquidity profile
- `portfolio_holdings.csv`: current holdings and asset mix
- `portfolio_performance.csv`: monthly performance history
- `product_catalog.csv`: representative Canadian products
- `reference_rag_index.json`: generated local RAG index
- `audit_logs/`: request audit traces

### `data/reference_canada/`

Reference knowledge used by rules and RAG:

- `account_knowledge.json`: account and product knowledge
- `planning_guidance.json`: planning rules and budgeting guidance
- `official_account_rules.json`: official-rule style summaries
- `market_context.json`: watchlist and market-layer configuration
- `market_commentary.json`: market narrative and month-level commentary

### `docs/`

Supporting documentation:

- `ARCHITECTURE_DIAGRAMS.md`: all project diagrams in one place
- `PROJECT_MAP_CANADA.md`: this file
- `SYSTEM_ARCHITECTURE.md`: detailed design and runtime flow
- `AWS_DEPLOYMENT.md`: cloud deployment notes

## Recommended Reading Order

1. `app/app_local.py`
2. `app/local_financial_qa.py`
3. `app/intent_engine.py`
4. `app/response_orchestrator.py`
5. `app/demo_governance.py`
6. `app/query_router.py`
7. `app/retrieval_backend.py`
8. `app/analytics_tools.py`
9. `app/rag_pipeline.py`
10. `app/langgraph_flow.py`
11. `data/artifacts_canada/portfolio_performance.csv`
12. `data/reference_canada/market_commentary.json`
