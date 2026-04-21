# System Architecture Design

## 1. Document Purpose

This document focuses on detailed architecture.

Use the root `README.md` for project overview and local setup.
Use `PROJECT_MAP_CANADA.md` for folder structure and reading order.
Use `ARCHITECTURE_DIAGRAMS.md` for all diagrams in one place.

This file explains:

- module responsibilities
- data architecture
- runtime workflow
- RAG pipeline
- LangChain implementation
- LangGraph orchestration
- fallback strategy

## 2. System Scope

### In Scope

- local Streamlit UI
- sample Canadian client-case finance data
- deterministic recommendation engine
- local RAG over reference JSON files
- optional OpenAI-based answer generation
- optional Yahoo Finance ETF snapshot
- LangGraph-based execution flow

### Out of Scope

- real customer banking integration
- transaction write-back
- compliance-grade regulated advice
- production authentication and authorization
- external vector database
- multi-user backend service

## 3. Business Goal

The app is designed as a learning-first financial copilot for a Canadian personal finance use case.

Its goal is to help a user:

- understand spending patterns
- compare account types such as `FHSA`, `TFSA`, and `RRSP`
- receive beginner-friendly product guidance
- see simple ETF market context
- learn how a hybrid rules + RAG + LLM architecture works

## 4. Architecture Style

The application uses a hybrid architecture:

- `UI layer`: Streamlit app
- `application layer`: query routing, orchestration, prompt assembly
- `domain logic layer`: spending analysis and recommendation scoring
- `knowledge layer`: local reference documents and RAG index
- `integration layer`: OpenAI Responses API and Yahoo Finance

This is not a microservice architecture. It is a modular monolith, which is a good choice for a beginner project because it is easier to read, run, and extend.

## 5. System Context

### Primary Users

- beginner learner
- single local developer
- internal demo user

### External Dependencies

- `OpenAI API`: optional LLM and embeddings
- `Yahoo Finance via yfinance`: optional ETF market snapshot
- local filesystem: CSV, JSON, and generated RAG index

## 6. High-Level Component View

### 6.1 Main Modules

| Module | File | Responsibility |
|---|---|---|
| UI | `app/app_local.py` | Streamlit pages, chat, dashboard, scenario planner, developer mode |
| Data Source Layer | `app/data_sources.py` | Load CSV/JSON data, environment variables, market snapshot fetch |
| Main Analysis Entry | `app/local_financial_qa.py` | Main runtime entry for question answering and fallback handling |
| Governance Layer | `app/demo_governance.py` | Internal request record, access checks, compliance guardrails, and audit logging |
| Analytics Tool Layer | `app/analytics_tools.py` | Deterministic portfolio attribution, exposure, and volatility analysis |
| Query Router | `app/query_router.py` | Detect user intent and choose the route |
| Context Orchestrator | `app/response_orchestrator.py` | Fetch RAG context and market context only when needed |
| LangGraph Flow | `app/langgraph_flow.py` | Build `route_query -> gather_context -> run_tools -> generate_response -> compliance_check` over a validated request state |
| Recommendation Engine | `app/recommendation_engine.py` | Score products with deterministic rules |
| RAG Pipeline | `app/rag_pipeline.py` | Build local chunks, embeddings, index, and retrieval |
| Prompt Builder | `app/prompt_builder.py` | Build structured prompt context for LLM calls |
| LangChain Adapter | `app/langchain_adapter.py` | Optional `langchain-core` prompt pipeline |

### 6.2 Logical Responsibilities

1. `app_local.py` handles interaction.
2. `analyze_financial_data_local()` is the main controller.
3. `route_query()` decides what kind of question the user asked.
4. `gather_orchestrated_context()` loads only the needed context.
5. `run_analysis_tools_for_route()` adds deterministic portfolio analytics when the route needs them.
6. `execute_route_analysis()` produces the final answer path.
7. `apply_compliance_to_payload()` softens unsafe wording and attaches guardrail metadata.
8. Rules-based fallback keeps the app usable even if external AI features fail.

## 7. Data Architecture

### 7.1 Operational Data

Stored in `data/artifacts_canada/`:

- `user_info.csv`: user persona
- `cat.csv`: transaction history
- `account_summary.csv`: account balances and liquidity fields
- `portfolio_holdings.csv`: current holdings and asset mix
- `portfolio_performance.csv`: monthly return history
- `product_catalog.csv`: product definitions

These files are loaded by `data_sources.py`.

### 7.2 Reference Knowledge

Stored in `data/reference_canada/`:

- `account_knowledge.json`
- `planning_guidance.json`
- `official_account_rules.json`
- `market_context.json`
- `market_commentary.json`

These files provide explanation-focused knowledge, planning rules, safety notes, market watchlist configuration, and month-level market narratives.

### 7.3 Derived Data

Stored in `data/artifacts_canada/reference_rag_index.json`:

- chunk metadata
- chunk text
- content hash
- optional embedding vectors
- provider and model metadata

This means the current project uses a local JSON index, not a dedicated vector database.

## 8. Runtime Architecture

### 8.1 Main Entry Flow

The main question-answering runtime starts from:

- UI event in `app_local.py`
- call to `analyze_financial_data_local(query, use_llm=True/False)`

### 8.2 Runtime Diagram

See [Architecture Diagrams](./ARCHITECTURE_DIAGRAMS.md#2-runtime-processing-flow).

### 8.3 Runtime Steps

1. Create an internal request record: user id, session id, and timestamp.
2. Load the minimal identity file from `user_info.csv`.
3. Enforce identity and access control before loading full client context.
4. Load full local finance context only after access is allowed.
5. Build a summary object.
6. Re-check the request against the loaded profile and run the early safety screen.
7. Choose the workflow engine: LangGraph if available, otherwise the Python fallback.
8. Route the query.
9. Gather optional market and RAG context.
10. Run analytics tools for routes that need grounded portfolio analysis.
11. Execute the selected answer path.
12. Run a post-generation compliance check.
13. Write an audit log.
14. Return structured output to Streamlit.
15. Render answer, cards, market snapshot, and optional developer diagnostics.

## 9. Query Routing Design

The router in `query_router.py` uses a hybrid routing pattern: keyword checks, safety checks, and semantic example matching.

### Supported Route Types

| Route | Meaning |
|---|---|
| `architecture_rules` | explain app structure and data layers |
| `spending_rules` | deterministic spending summary |
| `account_summary_rules` | household balances, liquidity, and net-worth summary |
| `portfolio_explanation` | portfolio holdings, asset mix, and positioning explanation |
| `performance_explanation` | why recent returns changed using performance data and market commentary |
| `risk_profile_rules` | risk tolerance and suitability explanation from the profile snapshot |
| `market_snapshot_rules` | fetch ETF snapshot |
| `market_explanation` | explain market backdrop and what it means for the portfolio |
| `recommendation_rules` | deterministic product prioritization |
| `rag_knowledge` | retrieve reference knowledge first |
| `hybrid_spending_recommendation` | combine spending and recommendation logic |
| `hybrid_market_advice` | combine recommendation and market context |
| `hybrid_rule_advice` | combine profile guidance and RAG rules |
| `hybrid_profile_rag_market` | combine profile + rules + market in one answer |
| `llm_general` | general LLM answer if no stronger route exists |
| `rules_fallback` | safe local fallback |

### Design Choice

This router is practical and still easy to learn:

- easy to read
- predictable
- easy to debug
- supports both explicit rules and semantic routing without a heavy classifier service

## 10. Recommendation Engine Design

`recommendation_engine.py` implements deterministic scoring.

### Inputs

- user profile
- product catalog
- reference knowledge
- priority order
- cash flow

### Logic

The engine:

- detects reason tags, such as `first_home_goal` and `positive_cash_flow`
- applies category-specific scoring rules
- calculates score and priority
- returns ranked recommendations

### Why This Matters

This gives the project a non-LLM decision layer. That is good system design because not every answer should depend on a model.

## 11. RAG Architecture

### 11.1 RAG Goal

The RAG layer helps answer finance knowledge questions using local reference documents instead of guessing.

### 11.2 Source Documents

The current RAG sources come from:

- `account_knowledge.json`
- `planning_guidance.json`
- `official_account_rules.json`
- `market_context.json`
- `market_commentary.json`

### 11.3 Chunk Construction

`rag_pipeline.py` converts reference JSON into normalized chunks:

- account chunks
- planning rule chunks
- budgeting guideline chunks
- safety note chunks
- official rule chunks
- market context overview chunks
- market watchlist chunks
- market story and theme chunks
- month-level market explanation chunks

Each chunk stores:

- `id`
- `title`
- `section`
- `source_file`
- `text`
- `content_hash`
- optional `vector`

### 11.4 Index Build Strategy

When the index is built:

1. reference files are loaded
2. chunks are generated
3. the system checks whether `OPENAI_API_KEY` exists
4. if yes, embeddings are created with `text-embedding-3-small`
5. if not, vectors are skipped and lexical retrieval is used
6. the index is written to `reference_rag_index.json`

#### Offline Index Build Flow

See [Architecture Diagrams](./ARCHITECTURE_DIAGRAMS.md#4-rag-index-build-flow).

### 11.5 Retrieval Strategy

When a RAG query arrives:

1. load or build the local index
2. detect retrieval mode
3. if embeddings exist:
   - embed the query
   - compute cosine similarity
   - combine semantic score, lexical score, and entity bonus
4. otherwise:
   - use lexical scoring only
5. sort chunks by score
6. return top `k` chunks

#### Online Query-Time Retrieval Flow

See [Architecture Diagrams](./ARCHITECTURE_DIAGRAMS.md#5-rag-query-time-retrieval-flow).

### 11.6 Important Design Note

This project uses a light local RAG architecture:

- no external vector DB
- no document store service
- no retrieval API server
- retrieval happens inside the Python app process

This is a smart beginner-friendly design because it keeps the system small while still showing true RAG concepts.

## 12. Prompt Engineering Design

`prompt_builder.py` assembles structured context before sending it to the model.

### Context Sections

Depending on the query, prompts may include:

- `route_context`
- `safety_context`
- `profile_context`
- `spending_context`
- `market_context`
- `retrieved_context`
- `source_context`

### Benefits

- cleaner prompts
- easier debugging
- route-specific context packing
- safer RAG answers because retrieved text is explicit

## 13. LangChain Implementation

### 13.1 Current Role

`langchain_adapter.py` provides an optional `LangChain` integration.

### 13.2 How It Works

1. `should_use_langchain_backend()` checks `LLM_BACKEND`.
2. `invoke_langchain_text()` builds a `ChatPromptTemplate`.
3. The template output is converted into OpenAI input text.
4. `RunnableLambda` executes the pipeline.
5. The final call still goes to the OpenAI Responses API.

### 13.3 What LangChain Is Doing Here

In this project, `LangChain` is used mainly as a prompt pipeline:

- prompt templating
- runnable composition
- backend abstraction

It is not currently being used for:

- agent tool-calling
- memory store
- retrieval framework ownership

### 13.4 Why This Is Reasonable

This is a good incremental adoption:

- keeps architecture understandable
- shows practical LangChain usage
- avoids unnecessary complexity

## 14. LangGraph Implementation

### 14.1 Current Role

`langgraph_flow.py` implements the orchestration graph.

### 14.2 Graph Nodes

The graph now has five nodes:

1. `route_query`
2. `gather_context`
3. `run_tools`
4. `generate_response`
5. `compliance_check`

### 14.3 Workflow Diagram

See [Architecture Diagrams](./ARCHITECTURE_DIAGRAMS.md#6-langgraph-workflow-diagram).

### 14.4 Graph State

The shared state includes:

- `query`
- `use_llm`
- `summary`
- `access_decision`
- `route_decision`
- `orchestrated_context`
- `answer_payload`

### 14.5 Why LangGraph Fits

LangGraph is useful here because the app already behaves like a workflow engine after request validation is complete:

- detect intent
- collect needed context
- execute one route
- produce final payload

Even though the graph is simple now, it creates a clean path for future upgrades:

- add human review
- add retries
- add tool nodes
- add evaluator nodes
- add multi-step decision branches

## 15. End-to-End Answer Paths

### Path A: Rules-Only Answer

Used when:

- no OpenAI key exists
- the route is clearly deterministic
- LLM call fails

Flow:

`User -> Router -> Local logic -> UI`

### Path B: General LLM Answer

Used when:

- no stronger specific route is matched
- LLM is enabled

Flow:

`User -> Router -> Prompt builder -> LangChain or OpenAI SDK -> UI`

### Path C: RAG Answer

Used when:

- the question is about finance rules or account knowledge

Flow:

`User -> Router -> RAG retrieve -> Prompt builder -> LLM synthesis or chunk-summary fallback -> UI`

### Path D: Hybrid Answer

Used when:

- the question combines profile, recommendations, rules, or market data

Flow:

`User -> Router -> Gather mixed context -> execute route -> render answer + cards + sources`

### Path E: Performance Explanation

Used when:

- the question asks why portfolio returns changed

Flow:

`User -> Access check -> Router -> Gather market + RAG context -> Run analytics tools -> Generate grounded explanation -> Compliance -> Audit log -> UI`

## 16. Error Handling and Fallback Strategy

This app is designed to degrade gracefully.

### Fallback Rules

- if `LangGraph` is unavailable or fails, use Python orchestration fallback
- if `LangChain` is unavailable or fails, use direct OpenAI SDK fallback
- if OpenAI structured parsing fails, use plain text response
- if LLM generation fails, use deterministic local rules or source-summary fallback
- if embeddings are unavailable, use lexical retrieval
- if Yahoo Finance fails, show a safe message instead of breaking the UI

## 17. Security and Safety Notes

Current safety controls include:

- internal request records for demo traceability
- identity and access control before AI reasoning
- local data only for core profile logic
- educational wording instead of regulated advice
- explicit safety notes in reference knowledge
- route separation between recommendation logic and market snapshot
- RAG instruction to avoid inventing legal thresholds or contribution limits
- post-generation guardrails for overly strong advice language
- file-based audit logging for demo traceability

Current limitations:

- no secrets manager
- no policy engine
- access control is still demo-local rather than production-grade IAM
- audit logging is file-based rather than a managed observability stack

These are acceptable for a local learning project, but they would matter in production.

## 18. Strengths of the Current Design

- clear module boundaries
- beginner-friendly architecture
- real hybrid AI pattern
- deterministic recommendation engine
- local RAG without heavy infrastructure
- graceful fallback design
- strong explainability through developer mode

## 19. Current Gaps and Future Improvements

### Current Gaps

- router is keyword-based, not model-based
- no persistent conversation memory
- no evaluation framework
- no unit/integration test suite shown in this architecture layer
- no production API/backend separation

### Recommended Next Improvements

1. add automated tests for routing, scoring, and retrieval
2. separate application service layer from UI layer further
3. add structured logging
4. add prompt/version tracking
5. add source citation rendering
6. move RAG index build into a dedicated script or startup task
7. add a classifier route or policy layer for more robust intent detection

## 20. Final Architecture Summary

This app is a modular monolith that combines:

- local finance datasets
- deterministic recommendation logic
- local JSON-based RAG
- optional OpenAI generation
- optional LangChain prompt pipeline
- optional LangGraph workflow orchestration
- optional Yahoo Finance market snapshot

The most important architecture idea is this:

`rules decide what is stable, RAG explains what is documented, and LLMs improve how the answer is written.`

That is a strong and standard hybrid AI system design for a beginner-friendly demo project.
