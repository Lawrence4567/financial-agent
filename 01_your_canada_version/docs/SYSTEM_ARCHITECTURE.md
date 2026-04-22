# System Architecture Design

## 1. Document Purpose

This document explains the detailed architecture of the Canada-first advisory demo after the hybrid workflow refactor.

Use the root `README.md` for project overview and local setup.
Use `PROJECT_MAP_CANADA.md` for folder structure and reading order.
Use `ARCHITECTURE_DIAGRAMS.md` for the visual diagrams.

This file focuses on:

- module responsibilities
- data architecture
- runtime workflow
- intent parsing and capability planning
- retrieval architecture
- deterministic tools and evidence objects
- LangChain and LangGraph roles
- fallback and safety strategy

## 2. System Scope

### In Scope

- local Streamlit UI
- sample Canadian client-case finance data
- deterministic spending, account, portfolio, and recommendation logic
- local retrieval over reference JSON files
- LLM-based intent parsing with rules fallback
- LLM-based answer synthesis from grounded evidence
- optional Yahoo Finance ETF snapshot
- LangGraph-based constrained workflow

### Out of Scope

- real customer banking integration
- transaction write-back
- compliance-grade regulated advice
- production authentication and authorization
- real external vector database infrastructure
- open-ended autonomous agent loops
- multi-user backend service

## 3. Business Goal

The app is a learning-first financial copilot for a Canadian personal finance use case.

Its goal is to help a user:

- understand spending patterns, including include/exclude questions
- compare account types such as `FHSA`, `TFSA`, and `RRSP`
- receive beginner-friendly product prioritization or de-prioritization
- explain portfolio positioning and performance using grounded data
- learn how a hybrid GenAI system works in practice

The key design principle is:

`LLM decides workflow, Python decides facts.`

That means:

- the LLM helps interpret the question and shape the answer
- deterministic tools compute money, rankings, balances, and analytics
- retrieval provides documented finance context
- compliance reviews the final wording

## 4. Architecture Style

The application is a modular monolith with six logical layers:

- `UI layer`: Streamlit app and chat rendering
- `workflow layer`: request validation, intent parsing, capability planning, orchestration
- `deterministic tool layer`: spending, account, portfolio, performance, and recommendation tools
- `knowledge layer`: local finance reference files and retrieval backend abstraction
- `generation layer`: prompt assembly and LLM synthesis
- `governance layer`: access checks, safety guardrails, and audit logging

This is not a microservice architecture. For a beginner-friendly project, a modular monolith is a good choice because it keeps the system readable and runnable on one machine.

## 5. High-Level Component View

### 5.1 Main Modules

| Module | File | Responsibility |
|---|---|---|
| UI | `app/app_local.py` | Streamlit pages, chat, dashboard, developer mode, and debug rendering |
| Main Analysis Entry | `app/local_financial_qa.py` | Main controller for request execution, tool runs, generation, compliance, and fallbacks |
| Intent Engine | `app/intent_engine.py` | Build `IntentSchema`, apply rules fallback, and create `CapabilityPlan` |
| Rules Fallback Router | `app/query_router.py` | Legacy route detection, safety checks, and route compatibility fallback |
| Context Orchestrator | `app/response_orchestrator.py` | Gather retrieval and market context only when the capability plan needs them |
| Retrieval Backend | `app/retrieval_backend.py` | Uniform retrieval interface with local-index backend and vector-store placeholder |
| Governance Layer | `app/demo_governance.py` | Internal request record, access checks, compliance guardrails, and audit logging |
| Analytics Tool Layer | `app/analytics_tools.py` | Deterministic portfolio attribution, exposure, and volatility analysis |
| Recommendation Engine | `app/recommendation_engine.py` | Deterministic scoring and ranking for product guidance |
| RAG Pipeline | `app/rag_pipeline.py` | Build local chunks, embeddings, JSON index, and retrieval primitives |
| Prompt Builder | `app/prompt_builder.py` | Build structured context for grounded generation |
| LangChain Adapter | `app/langchain_adapter.py` | Optional `langchain-core` prompt pipeline |
| LangGraph Flow | `app/langgraph_flow.py` | Build the constrained workflow from intent parsing to compliance |
| Data Source Layer | `app/data_sources.py` | Load CSV/JSON data, environment variables, and market snapshot inputs |

### 5.2 Logical Responsibilities

1. `app_local.py` handles interaction and developer mode rendering.
2. `analyze_financial_data_local()` is the main runtime controller.
3. `parse_intent()` creates a structured `IntentSchema`.
4. `plan_capabilities()` decides which tools and retrieval steps are needed.
5. `gather_orchestrated_context()` loads only the required retrieval and market context.
6. `run_analysis_tools_for_plan()` computes deterministic evidence.
7. `validate_tool_results()` checks that the tool outputs are consistent and usable.
8. `answer_with_capability_generation()` generates the final answer from evidence when generation is allowed.
9. `apply_compliance_to_payload()` softens risky wording and attaches guardrail metadata.
10. `append_audit_event()` writes a JSONL audit trail.

## 6. Data Architecture

### 6.1 Operational Data

Stored in `data/artifacts_canada/`:

- `user_info.csv`: user persona
- `cat.csv`: transaction history
- `account_summary.csv`: account balances and liquidity fields
- `portfolio_holdings.csv`: current holdings and asset mix
- `portfolio_performance.csv`: monthly return history
- `product_catalog.csv`: product definitions

These files are loaded by `data_sources.py`.

### 6.2 Reference Knowledge

Stored in `data/reference_canada/`:

- `account_knowledge.json`
- `planning_guidance.json`
- `official_account_rules.json`
- `market_context.json`
- `market_commentary.json`

These files provide explanation-focused knowledge, planning rules, safety notes, watchlist configuration, and market narratives.

### 6.3 Derived Data

Stored in `data/artifacts_canada/reference_rag_index.json`:

- chunk metadata
- chunk text
- content hash
- optional embedding vectors
- provider and model metadata

This means the default retrieval backend is local and file-based. The refactor adds a retrieval abstraction, but it still uses the local JSON index by default.

## 7. Core Runtime Workflow

The main runtime starts from:

- a UI event in `app_local.py`
- a call to `analyze_financial_data_local(query, use_llm=True/False)`

### Runtime Steps

1. Create an internal request record with user id, session id, and timestamp.
2. Load minimal identity data from `user_info.csv`.
3. Enforce identity and access control before loading the full finance context.
4. Load the local finance context only after access is allowed.
5. Build a compact summary object for downstream logic.
6. Re-check safety and allowed scope against the loaded profile.
7. Choose the workflow engine: `LangGraph` when enabled, otherwise Python fallback orchestration.
8. Parse structured intent.
9. Build a capability plan from that intent.
10. Gather only the required market and retrieval context.
11. Run deterministic tools to produce evidence objects.
12. Validate tool outputs and prepare an evidence summary.
13. Generate the answer from evidence, unless access or safety short-circuits the request.
14. Run a compliance review on the outgoing answer payload.
15. Write an audit event.
16. Return the structured result to Streamlit for rendering.

## 8. Intent Parsing and Capability Planning

This is the biggest architectural change in the refactor.

The old flow was:

`Router -> Rules -> Optional LLM`

The new flow is:

`Intent Parsing -> Capability Planning -> Tools / Retrieval -> LLM Synthesis -> Compliance`

### 8.1 IntentSchema

`IntentSchema` is the shared public type for the workflow.

| Field | Meaning |
|---|---|
| `domain` | Main business area such as `spending`, `recommendation`, `market`, or `knowledge` |
| `operator` | The action, such as `summarize`, `explain`, `compare`, `include`, `exclude`, `prioritize`, or `deprioritize` |
| `target_entities` | Key entities like `food`, `FHSA`, or `TFSA` |
| `polarity` | `positive`, `negative`, or `neutral` |
| `needs_recommendation` | Whether recommendation logic is needed |
| `needs_rag` | Whether retrieval over reference knowledge is needed |
| `needs_market_data` | Whether market snapshot or market commentary is needed |
| `needs_portfolio_tools` | Whether portfolio/performance analytics are needed |
| `response_style` | Desired answer style such as `brief`, `explanatory`, or `tabular` |
| `confidence` | Intent confidence level |
| `fallback_reason` | Why rules fallback was used instead of LLM intent parsing |

### 8.2 How Intent Parsing Works

`parse_intent()` follows a hybrid strategy:

1. Try LLM-based structured parsing when `INTENT_BACKEND=llm` and OpenAI is available.
2. If that fails, falls back to the existing rules-based router and query understanding logic.
3. Apply discoverable overrides for cases that the LLM or router may still miss.

Important intent cases now handled explicitly:

- spending include vs exclude
- negative recommendation such as `what should I avoid`
- account comparison such as `FHSA vs TFSA`
- mixed questions that need spending + recommendation + rules
- follow-up references using recent chat history

### 8.3 CapabilityPlan

`CapabilityPlan` is the workflow contract after intent parsing.

| Field | Meaning |
|---|---|
| `tool_calls` | Ordered list of tools to run |
| `requires_generation` | Whether the answer should go through the generation layer |
| `requires_compliance_review` | Whether post-generation guardrails are needed |
| `ui_route_label` | Friendly label kept for UI compatibility |
| `route_reason` | Debug explanation for why the plan was chosen |
| `legacy_route` | Backward-compatible route name |
| `uses_rag` | Whether retrieval is part of the plan |
| `prefers_llm` | Whether the plan should try LLM synthesis first |

### 8.4 Supported Tool Calls

The current planner can request these tool calls:

- `spending_tool`
- `account_summary_tool`
- `portfolio_tool`
- `portfolio_performance_toolkit`
- `recommendation_engine`
- `reference_retrieval`
- `market_snapshot`
- `architecture_context`

`ui_route_label` is still returned for developer mode and UI compatibility, but it is no longer the main decision source.

## 9. Deterministic Tool Layer

The finance facts come from Python, not from the LLM.

This is especially important for:

- money values
- time windows
- product ranking order
- account summaries
- portfolio statistics

### 9.1 Evidence-First Design

Each major route now follows this pattern:

1. run a deterministic tool
2. build a structured evidence object
3. optionally build a fallback draft
4. pass the evidence into the generation layer

This keeps the system grounded and auditable.

### 9.2 Main Evidence Builders

`local_financial_qa.py` now contains structured builders such as:

- spending evidence
- recommendation evidence
- account evidence
- portfolio evidence
- architecture evidence
- retrieval evidence
- market snapshot evidence

The validation step checks that the tool outputs are present, consistent, and safe to narrate.

## 10. Retrieval Architecture

### 10.1 Retrieval Goal

The retrieval layer answers finance-rule and explanation questions using local reference documents instead of guessing.

### 10.2 Source Documents

The current sources are:

- `account_knowledge.json`
- `planning_guidance.json`
- `official_account_rules.json`
- `market_context.json`
- `market_commentary.json`

### 10.3 Retrieval Backend Abstraction

The refactor introduces a formal retrieval interface:

`retrieve(query: str, top_k: int, filters: dict | None) -> RetrievalResult`

`RetrievalResult` includes:

- `backend`
- `chunks`
- `filters_applied`
- `query_used`

Current implementations:

- `LocalIndexRetriever`: the active backend
- `VectorStoreRetriever`: placeholder adapter for future expansion

If `RETRIEVAL_BACKEND` is set to an unsupported future value, the system safely falls back to the local index backend.

### 10.4 Index Build Strategy

When the index is built:

1. reference files are loaded
2. chunks are generated
3. the system checks whether `OPENAI_API_KEY` exists
4. if yes, embeddings are created with `text-embedding-3-small`
5. if not, vectors are skipped and lexical retrieval is used
6. the index is written to `reference_rag_index.json`

### 10.5 Query-Time Retrieval Strategy

When retrieval is requested:

1. the retrieval backend is chosen
2. the local index is loaded or built
3. the system chooses embedding or lexical retrieval mode
4. chunks are ranked
5. the top chunks are returned in a typed `RetrievalResult`
6. the result is passed into prompt assembly and developer mode

## 11. Prompt and Generation Layer

`prompt_builder.py` assembles the context before the model is called.

Depending on the plan, prompts may include:

- route and safety context
- profile and spending context
- market context
- retrieved reference context
- tool evidence
- evidence summary

### 11.1 Unified Generation Exit

`answer_with_capability_generation()` is now the main generation path for non-refusal answers.

This means:

- most user-visible answers go through one generation layer
- access-denied and safety-refusal cases still short-circuit
- deterministic fallback drafts remain available when generation fails

### 11.2 Guardrail Principle

The LLM is not allowed to invent:

- money values
- recommendation scores
- contribution rules not present in sources
- unsupported market facts

The model is used to explain evidence, not to replace it.

## 12. LangChain and LangGraph Roles

### 12.1 LangChain

`langchain_adapter.py` still provides an optional prompt pipeline.

Its current role is:

- prompt templating
- runnable composition
- backend abstraction

It is not the main reasoning engine. The main reasoning pattern now comes from `IntentSchema`, `CapabilityPlan`, and the workflow graph.

### 12.2 LangGraph

`langgraph_flow.py` implements the constrained orchestration graph.

The graph nodes are:

1. `parse_intent`
2. `plan_capabilities`
3. `gather_context`
4. `run_tools`
5. `validate_tool_results`
6. `generate_answer`
7. `compliance_check`

### 12.3 Shared Graph State

The state now includes:

- `query`
- `use_llm`
- `summary`
- `access_decision`
- `intent`
- `capability_plan`
- `orchestrated_context`
- `tool_outputs`
- `answer_payload`

### 12.4 Why This Graph Design Fits

This graph is intentionally constrained.

It gives the project:

- explicit workflow stages
- easier debugging
- better developer-mode visibility
- safer financial behavior than an open-ended agent loop

It does not try to create an unlimited autonomous agent. That would add risk without much learning value for this project.

## 13. End-to-End Answer Paths

### Path A: Access or Safety Refusal

Used when:

- the user is out of allowed scope
- the query triggers the safety/compliance guard

Flow:

`User -> Validation -> Refusal payload -> UI`

### Path B: Deterministic Fallback Answer

Used when:

- OpenAI is unavailable
- LLM generation fails
- intent parsing falls back to rules mode

Flow:

`User -> Intent parser fallback -> Capability plan -> Tool evidence -> Deterministic fallback draft -> UI`

### Path C: Knowledge or Hybrid Answer

Used when:

- the question needs rules, profile context, and retrieval
- the capability plan requests both tools and RAG

Flow:

`User -> Intent parsing -> Capability plan -> Retrieval + tools -> LLM synthesis -> Compliance -> UI`

### Path D: Portfolio / Performance Explanation

Used when:

- the user asks about portfolio positioning or performance changes

Flow:

`User -> Access check -> Intent parsing -> Market + retrieval context -> Analytics tools -> Evidence validation -> Grounded generation -> Compliance -> Audit -> UI`

## 14. Developer Mode and Observability

Developer mode now exposes the new workflow metadata directly in the UI.

The assistant message can show:

- `IntentSchema`
- `CapabilityPlan`
- `generation_source`
- `fallback_reason`
- `evidence_summary`

This is helpful because the user can now see:

- what the system thought the question meant
- what tools it decided to call
- whether generation came from LLM or fallback logic
- why a fallback happened

The audit log also records request-level metadata and tool usage for traceability.

## 15. Error Handling and Fallback Strategy

The app is designed to degrade gracefully.

### Fallback Rules

- if access is denied, return immediately with an access-gate payload
- if safety validation fails, return immediately with a safety payload
- if `WORKFLOW_BACKEND` is not `langgraph` or the graph fails, use Python orchestration fallback
- if `INTENT_BACKEND=rules_fallback`, skip LLM intent parsing
- if LLM intent parsing fails, fall back to rules parsing and store `fallback_reason`
- if LLM generation fails, return the deterministic draft while preserving evidence metadata
- if the retrieval backend is unavailable, fall back to the local index backend
- if embeddings are unavailable, use lexical retrieval
- if Yahoo Finance fails, show a safe message instead of breaking the UI

## 16. Security and Safety Notes

Current safety controls include:

- internal request records for demo traceability
- identity and access control before AI reasoning
- local data only for core profile logic
- educational wording instead of regulated advice
- explicit safety notes in reference knowledge
- post-generation guardrails for overly strong advice language
- file-based audit logging for demo traceability

Current limitations:

- no secrets manager
- no policy engine
- access control is still demo-local rather than production IAM
- audit logging is file-based rather than a managed observability stack

These limits are acceptable for a learning project, but they matter in production.

## 17. Strengths of the Current Design

- clear separation between understanding, planning, computation, and generation
- deterministic finance facts with better GenAI orchestration
- local retrieval without heavy infrastructure
- graceful fallback design
- strong explainability through developer mode
- beginner-friendly code structure with more realistic hybrid AI patterns

## 18. Current Gaps and Future Improvements

### Current Gaps

- intent parsing is still single-shot rather than evaluator-driven
- vector-store retrieval is only a placeholder adapter today
- no persistent long-term conversation memory
- no production backend separation
- no full model evaluation dashboard yet

### Recommended Next Improvements

1. add more intent and capability-plan test coverage
2. add source citation rendering in the user-facing answer
3. add prompt and workflow version tracking
4. add a richer evaluator loop for ambiguous intent cases
5. add a real vector database only when retrieval scale justifies it
6. separate the application service layer from Streamlit further if moving toward production

## 19. Final Architecture Summary

This app is a modular monolith that combines:

- local finance datasets
- deterministic finance and recommendation tools
- local JSON-based retrieval
- typed intent parsing and capability planning
- optional OpenAI generation
- optional LangChain prompt composition
- optional LangGraph workflow orchestration
- optional Yahoo Finance market snapshot

The most important architecture idea is now:

`LLM decides workflow, Python decides facts, retrieval grounds explanations, and compliance reviews the final wording.`
