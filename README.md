# Financial Advisory GenAI

This repository contains a Canada-first financial advisory demo app built as a learning-friendly GenAI project.

Main app path:

`01_your_canada_version/`

## What This Project Shows

This is not only a simple chat demo. The app demonstrates a fuller advisory workflow:

- `request context`: create an internal request record for session, timestamp, and access checks
- `identity and access control`: check access before AI reasoning
- `intent routing`: combine explicit rules, semantic matching, and workflow orchestration
- `structured data retrieval`: use holdings, performance, accounts, and transactions
- `RAG`: retrieve grounded context from local Canada finance reference files
- `analytics tools`: explain portfolio attribution, exposure, and volatility
- `LLM generation`: optionally use OpenAI for richer grounded answers
- `compliance guardrails`: soften unsafe wording and apply basic redaction
- `audit logging`: write each request to a JSONL audit trail

## Documentation Guide

Use the docs folder by topic instead of reading every file from the top:

- [Architecture Diagrams](./01_your_canada_version/docs/ARCHITECTURE_DIAGRAMS.md): all important project diagrams in one place
- [System Architecture Design](./01_your_canada_version/docs/SYSTEM_ARCHITECTURE.md): detailed architecture, runtime flow, RAG, LangChain, and LangGraph
- [Canada Workspace Map](./01_your_canada_version/docs/PROJECT_MAP_CANADA.md): folder guide and recommended reading order
- [AWS Deployment Guide](./01_your_canada_version/docs/AWS_DEPLOYMENT.md): cloud deployment options for this repo

## Main Workspace Structure

Important folders inside `01_your_canada_version/`:

- `app/`: Streamlit UI, routing, orchestration, analytics tools, and LLM integration
- `data/artifacts_canada/`: synthetic client-case data, product data, generated RAG index, and audit logs
- `data/reference_canada/`: local finance reference knowledge, market context, and commentary
- `docs/`: architecture, workspace map, and deployment docs
- `requirements-local.txt`: local Python dependencies

## Local Setup

From the repo root:

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate it in PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r 01_your_canada_version\requirements-local.txt
   ```

4. Create a `.env` file from `.env.example`

5. Run the app:

   ```powershell
   streamlit run 01_your_canada_version\app\app_local.py
   ```

You can also run it with the local virtual environment Python directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run 01_your_canada_version\app\app_local.py --server.port 8506 --browser.gatherUsageStats false
```

## Environment Variables

The most important variables are:

- `OPENAI_API_KEY`: optional; leave empty for rules-only mode
- `OPENAI_MODEL`: optional model override for generation
- `OPENAI_EMBEDDING_MODEL`: optional model override for embeddings
- `LLM_BACKEND`: optional backend selector
- `FINANCE_DATA_DIR`: optional data path override

## Suggested Demo Questions

- `Show my household account summary.`
- `Explain my current portfolio allocation.`
- `Why did my portfolio go down this month?`
- `Explain market changes and what they mean for this portfolio.`
- `Based on my profile, should I focus on FHSA, TFSA, or RRSP?`

## Architecture Summary

This app uses a hybrid design:

- `rules engine`: deterministic recommendation scoring and finance summaries
- `RAG`: retrieval over account, planning, official-rule, and market-commentary content
- `analytics tool layer`: grounded portfolio analysis for explanation routes
- `LangChain`: optional prompt pipeline
- `LangGraph`: optional workflow orchestration
- `OpenAI Responses API`: optional final language generation
- `Yahoo Finance via yfinance`: optional live ETF snapshot

The core architecture idea is:

`rules decide what is stable, RAG explains what is documented, and LLMs improve how the answer is written.`

For diagrams, use [Architecture Diagrams](./01_your_canada_version/docs/ARCHITECTURE_DIAGRAMS.md).

## Example End-to-End Scenario

For a question like `Why did my portfolio go down this month?`, the app:

1. validates identity and allowed scope
2. classifies the query
3. loads the needed portfolio and market context
4. retrieves relevant market commentary through RAG
5. runs analytics tools
6. generates a grounded explanation
7. applies compliance review
8. writes an audit log

## Learning Order

If you want to learn the codebase step by step, start here:

1. `01_your_canada_version/app/app_local.py`
2. `01_your_canada_version/app/local_financial_qa.py`
3. `01_your_canada_version/app/demo_governance.py`
4. `01_your_canada_version/app/query_router.py`
5. `01_your_canada_version/app/response_orchestrator.py`
6. `01_your_canada_version/app/analytics_tools.py`
7. `01_your_canada_version/app/rag_pipeline.py`
8. `01_your_canada_version/app/langgraph_flow.py`
9. `01_your_canada_version/app/prompt_builder.py`
