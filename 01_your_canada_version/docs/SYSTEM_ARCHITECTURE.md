# System Architecture Design

## 1. Document Purpose

This document explains the system architecture [系统架构] of the Canada-first financial advisory app [金融顾问应用] in a standard, beginner-friendly way.

It covers:

- business goal [业务目标]
- application structure [应用结构]
- module responsibilities [模块职责]
- data architecture [数据架构]
- runtime workflow [运行流程]
- RAG pipeline [检索增强生成流程]
- LangChain implementation [LangChain 实现方式]
- LangGraph orchestration [LangGraph 编排方式]
- fallback strategy [降级策略]

Main app path:

`01_your_canada_version/`

## 2. System Scope

### In Scope

- local Streamlit UI [本地界面]
- sample Canadian client-case finance data [加拿大客户案例金融数据]
- deterministic recommendation engine [确定性推荐引擎]
- local RAG over reference JSON files [基于本地参考文件的 RAG]
- optional OpenAI-based answer generation [可选的 OpenAI 答案生成]
- optional Yahoo Finance ETF snapshot [可选的 Yahoo Finance ETF 快照]
- LangGraph-based execution flow [基于 LangGraph 的执行流程]

### Out of Scope

- real customer banking integration [真实银行系统接入]
- transaction write-back [交易回写]
- compliance-grade regulated advice [合规级受监管投资建议]
- production authentication and authorization [生产级身份认证与授权]
- external vector database [外部向量数据库]
- multi-user backend service [多用户后端服务]

## 3. Business Goal

The app is designed as a learning-first financial copilot [金融副驾驶] for a Canadian personal finance use case.

Its goal is to help a user:

- understand spending patterns [消费模式]
- compare account types such as `FHSA`, `TFSA`, and `RRSP`
- receive beginner-friendly product guidance [产品建议]
- see simple ETF market context [ETF 市场上下文]
- learn how a hybrid rules + RAG + LLM architecture works [规则 + RAG + LLM 混合架构]

## 4. Architecture Style

The application uses a hybrid architecture [混合架构]:

- `UI layer [界面层]`: Streamlit app
- `application layer [应用层]`: query routing, orchestration, prompt assembly
- `domain logic layer [领域逻辑层]`: spending analysis and recommendation scoring
- `knowledge layer [知识层]`: local reference documents and RAG index
- `integration layer [集成层]`: OpenAI Responses API and Yahoo Finance

This is not a microservice architecture [微服务架构]. It is a modular monolith [模块化单体应用], which is a good choice for a beginner project because it is easier to read, run, and extend.

## 5. System Context

### Primary Users

- portfolio reviewer [作品集评审者]
- interview panel [面试官]
- beginner learner [初学者]
- single local developer [本地开发者]

### External Dependencies

- `OpenAI API`: optional LLM and embeddings [嵌入]
- `Yahoo Finance via yfinance`: optional ETF market snapshot
- local filesystem [本地文件系统]: CSV, JSON, and generated RAG index

## 6. High-Level Component View

### 6.1 Main Modules

| Module | File | Responsibility |
|---|---|---|
| UI | `app/app_local.py` | Streamlit pages, chat, dashboard, scenario planner, developer mode |
| Data Source Layer | `app/data_sources.py` | Load CSV/JSON data, environment variables, market snapshot fetch |
| Main Analysis Entry | `app/local_financial_qa.py` | Main runtime entry for question answering and fallback handling |
| Query Router | `app/query_router.py` | Detect user intent and choose the route |
| Context Orchestrator | `app/response_orchestrator.py` | Fetch RAG context and market context only when needed |
| LangGraph Flow | `app/langgraph_flow.py` | Build `route -> gather_context -> execute_route` graph |
| Recommendation Engine | `app/recommendation_engine.py` | Score products with deterministic rules |
| RAG Pipeline | `app/rag_pipeline.py` | Build local chunks, embeddings, index, and retrieval |
| Prompt Builder | `app/prompt_builder.py` | Build structured prompt context for LLM calls |
| LangChain Adapter | `app/langchain_adapter.py` | Optional `langchain-core` prompt pipeline |

### 6.2 Logical Responsibilities

1. `app_local.py` handles interaction [交互].
2. `analyze_financial_data_local()` is the main controller [主控制入口].
3. `route_query()` decides what kind of question the user asked.
4. `gather_orchestrated_context()` loads only the needed context.
5. `execute_route_analysis()` produces the final answer path.
6. `answer_with_llm()` and `answer_with_rag()` call the LLM when available.
7. Rules-based fallback [规则降级] keeps the app usable even if external AI features fail.

## 7. Directory Structure

```text
financial_advisory_genai/
├─ README.md
├─ .env.example
└─ 01_your_canada_version/
   ├─ app/
   │  ├─ app_local.py
   │  ├─ data_sources.py
   │  ├─ local_financial_qa.py
   │  ├─ query_router.py
   │  ├─ response_orchestrator.py
   │  ├─ rag_pipeline.py
   │  ├─ prompt_builder.py
   │  ├─ langchain_adapter.py
   │  ├─ langgraph_flow.py
   │  └─ recommendation_engine.py
   ├─ data/
   │  ├─ artifacts_canada/
   │  │  ├─ cat.csv
   │  │  ├─ user_info.csv
   │  │  ├─ product_catalog.csv
   │  │  └─ reference_rag_index.json
   │  └─ reference_canada/
   │     ├─ account_knowledge.json
   │     ├─ planning_guidance.json
   │     ├─ official_account_rules.json
   │     └─ market_context.json
   └─ docs/
      ├─ README_LOCAL.md
      ├─ PROJECT_MAP_CANADA.md
      └─ SYSTEM_ARCHITECTURE.md
```

## 8. Data Architecture

### 8.1 Operational Data [业务数据]

Stored in `data/artifacts_canada/`:

- `user_info.csv`: user persona [用户画像]
- `cat.csv`: transaction history [交易历史]
- `product_catalog.csv`: product definitions [产品目录]

These files are loaded by `data_sources.py`.

### 8.2 Reference Knowledge [参考知识]

Stored in `data/reference_canada/`:

- `account_knowledge.json`
- `planning_guidance.json`
- `official_account_rules.json`
- `market_context.json`

These files provide explanation-focused knowledge [解释型知识], planning rules [规划规则], safety notes [安全提示], and market watchlist configuration [观察列表配置].

### 8.3 Derived Data [派生数据]

Stored in `data/artifacts_canada/reference_rag_index.json`:

- chunk metadata [分块元数据]
- chunk text [分块文本]
- content hash [内容哈希]
- optional embedding vectors [可选嵌入向量]
- provider and model metadata [提供方与模型元数据]

This means the current project uses a local JSON index [本地 JSON 索引], not a dedicated vector database [向量数据库].

## 9. Runtime Architecture

### 9.1 Main Entry Flow

The main question-answering runtime starts from:

- UI event in `app_local.py`
- call to `analyze_financial_data_local(query, use_llm=True/False)`

### 9.2 Runtime Steps

1. Capture request metadata [请求元数据]: user id, session id, channel, device, timestamp.
2. Load local finance context [金融上下文].
3. Build a summary object [汇总对象].
4. Enforce identity and access control [身份与权限控制] before AI reasoning.
5. Try LangGraph orchestration if installed.
6. Route the query.
7. Gather optional market and RAG context.
8. Run analytics tools for routes that need grounded portfolio analysis.
9. Execute the selected answer path.
10. Run a post-generation compliance check [生成后合规检查].
11. Write an audit log [审计日志].
12. Return structured output to Streamlit.
13. Render answer, cards, market snapshot, and optional developer diagnostics.

## 10. Query Routing Design

The router in `query_router.py` uses keyword-based intent detection [基于关键词的意图识别].

### Supported Route Types

| Route | Meaning |
|---|---|
| `architecture_rules` | explain app structure and data layers |
| `spending_rules` | deterministic spending summary |
| `market_snapshot_rules` | fetch ETF snapshot |
| `recommendation_rules` | deterministic product prioritization |
| `rag_knowledge` | retrieve reference knowledge first |
| `hybrid_spending_recommendation` | combine spending and recommendation logic |
| `hybrid_market_advice` | combine recommendation and market context |
| `hybrid_rule_advice` | combine profile guidance and RAG rules |
| `hybrid_profile_rag_market` | combine profile + rules + market in one answer |
| `llm_general` | general LLM answer if no stronger route exists |
| `rules_fallback` | safe local fallback |

### Design Choice

This router is simple but educational:

- easy to read [易读]
- predictable [可预测]
- easy to debug [易调试]
- good for learning orchestration before using a classifier model [在引入分类模型前先学习编排]

## 11. Recommendation Engine Design

`recommendation_engine.py` implements deterministic scoring [确定性评分].

### Inputs

- user profile [用户画像]
- product catalog [产品目录]
- reference knowledge [参考知识]
- priority order [优先级顺序]
- cash flow [现金流]

### Logic

The engine:

- detects reason tags [原因标签], such as `first_home_goal` and `positive_cash_flow`
- applies category-specific scoring rules [分类评分规则]
- calculates score and priority
- returns ranked recommendations [排序推荐]

### Why This Matters

This gives the project a non-LLM decision layer [非 LLM 决策层]. That is good system design because not every answer should depend on a model.

## 12. RAG Architecture

### 12.1 RAG Goal

The RAG layer helps answer finance knowledge questions using local reference documents instead of guessing [猜测].

### 12.2 Source Documents

The current RAG sources come from:

- `account_knowledge.json`
- `planning_guidance.json`
- `official_account_rules.json`

### 12.3 Chunk Construction

`rag_pipeline.py` converts reference JSON into normalized chunks [标准化文本分块]:

- account chunks
- planning rule chunks
- budgeting guideline chunks
- safety note chunks
- official rule chunks

Each chunk stores:

- `id`
- `title`
- `section`
- `source_file`
- `text`
- `content_hash`
- optional `vector`

### 12.4 Index Build Strategy

When the index is built:

1. reference files are loaded
2. chunks are generated
3. the system checks whether `OPENAI_API_KEY` exists
4. if yes, embeddings are created with `text-embedding-3-small`
5. if not, vectors are skipped and lexical retrieval [词法检索] is used
6. the index is written to `reference_rag_index.json`

### 12.5 Retrieval Strategy

When a RAG query arrives:

1. load or build the local index
2. detect retrieval mode
3. if embeddings exist:
   - embed the query
   - compute cosine similarity [余弦相似度]
   - combine semantic score [语义分数], lexical score [词法分数], and entity bonus [实体加分]
4. otherwise:
   - use lexical scoring only
5. sort chunks by score
6. return top `k` chunks

### 12.6 Important Design Note

This project uses a light local RAG architecture [轻量本地 RAG 架构]:

- no external vector DB
- no document store service
- no retrieval API server
- retrieval happens inside the Python app process [进程内检索]

This is a smart beginner-friendly design because it keeps the system small while still showing true RAG concepts.

## 13. Prompt Engineering Design

`prompt_builder.py` assembles structured context [结构化上下文] before sending it to the model.

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

- cleaner prompts [更清晰的提示词]
- easier debugging [更容易调试]
- route-specific context packing [按路由打包上下文]
- safer RAG answers because retrieved text is explicit [检索文本显式提供]

## 14. LangChain Implementation

### 14.1 Current Role

`langchain_adapter.py` provides an optional `LangChain` integration [集成].

### 14.2 How It Works

1. `should_use_langchain_backend()` checks `LLM_BACKEND`.
2. `invoke_langchain_text()` builds a `ChatPromptTemplate`.
3. The template output is converted into OpenAI input text.
4. `RunnableLambda` executes the pipeline.
5. The final call still goes to the OpenAI Responses API.

### 14.3 What LangChain Is Doing Here

In this project, `LangChain` is used mainly as a prompt pipeline [提示词流水线]:

- prompt templating [模板化]
- runnable composition [可运行链式组合]
- backend abstraction [后端抽象]

It is not currently being used for:

- agent tool-calling [工具调用代理]
- memory store [记忆存储]
- retrieval framework ownership

### 14.4 Why This Is Reasonable

This is a good incremental adoption [渐进式引入]:

- keeps architecture understandable
- shows practical LangChain usage
- avoids unnecessary complexity [不必要复杂性]

## 15. LangGraph Implementation

### 15.1 Current Role

`langgraph_flow.py` implements the orchestration graph [编排图].

### 15.2 Graph Nodes

The graph now has six nodes:

1. `verify_access`
2. `route_query`
3. `gather_context`
4. `run_tools`
5. `generate_response`
6. `compliance_check`

### 15.3 Graph State

The shared state [共享状态] includes:

- `query`
- `use_llm`
- `summary`
- `route_decision`
- `orchestrated_context`
- `answer_payload`

### 15.4 Why LangGraph Fits

LangGraph is useful here because the app already behaves like a workflow engine [工作流引擎]:

- detect intent
- collect needed context
- execute one route
- produce final payload

Even though the graph is simple now, it creates a clean path for future upgrades:

- add human review [人工审核]
- add retries [重试]
- add tool nodes [工具节点]
- add evaluator nodes [评估节点]
- add multi-step decision branches [多步分支]

## 16. End-to-End Answer Paths

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

## 17. Error Handling and Fallback Strategy

This app is designed to degrade gracefully [优雅降级].

### Fallback Rules

- if `LangGraph` is unavailable or fails, use Python orchestration fallback
- if `LangChain` is unavailable or fails, use direct OpenAI SDK fallback
- if OpenAI structured parsing fails, use plain text response
- if LLM generation fails, use deterministic local rules or source-summary fallback
- if embeddings are unavailable, use lexical retrieval
- if Yahoo Finance fails, show safe message instead of breaking the UI

This is strong architecture practice [良好的架构实践] because the user still gets an answer even when optional components fail.

## 18. Security and Safety Notes

Current safety controls [安全控制] include:

- request metadata capture for demo flows
- identity and access control before AI reasoning
- local data only for core profile logic
- educational wording instead of regulated advice
- explicit safety notes in reference knowledge
- route separation between recommendation logic and market snapshot
- RAG instruction to avoid inventing legal thresholds or contribution limits
- post-generation guardrails for overly strong advice language
- file-based audit logging for demo traceability

Current limitations:

- no secrets manager [密钥管理]
- no policy engine [策略引擎]
- access control is still demo-local rather than production-grade IAM [身份访问管理]
- audit logging is file-based rather than a managed observability stack [可观测性栈]

These are acceptable for a local learning project, but they would matter in production [生产环境].

## 19. Deployment View

Current deployment style [部署方式]:

- local machine execution [本地机器运行]
- Streamlit web server [Streamlit Web 服务]
- file-based storage [基于文件的存储]
- optional internet calls to OpenAI and Yahoo Finance

Required runtime dependencies are listed in:

`01_your_canada_version/requirements-local.txt`

Key packages:

- `streamlit`
- `pandas`
- `openai`
- `python-dotenv`
- `yfinance`
- `langchain-core`
- `langgraph`

## 20. Strengths of the Current Design

- clear module boundaries [清晰模块边界]
- beginner-friendly architecture [适合初学者的架构]
- real hybrid AI pattern [真实混合 AI 模式]
- deterministic recommendation engine
- local RAG without heavy infrastructure
- graceful fallback design
- strong explainability [可解释性] through developer mode

## 21. Current Gaps and Future Improvements

### Current Gaps

- router is keyword-based, not model-based
- no persistent conversation memory [持久会话记忆]
- no evaluation framework [评估框架]
- no unit/integration test suite shown in this architecture layer
- no production API/backend separation

### Recommended Next Improvements

1. add automated tests for routing, scoring, and retrieval
2. separate application service layer from UI layer further
3. add structured logging [结构化日志]
4. add prompt/version tracking [提示词版本跟踪]
5. add source citation rendering [来源引用渲染]
6. move RAG index build into a dedicated script or startup task
7. add a classifier route or policy layer for more robust intent detection

## 22. Final Architecture Summary

This app is a modular monolith [模块化单体应用] that combines:

- local finance datasets
- deterministic recommendation logic
- local JSON-based RAG
- optional OpenAI generation
- optional LangChain prompt pipeline
- optional LangGraph workflow orchestration
- optional Yahoo Finance market snapshot

The most important architecture idea is this:

`rules decide what is stable, RAG explains what is documented, and LLMs improve how the answer is written.`

That is a strong and standard hybrid AI system design [标准混合 AI 系统设计] for a beginner portfolio project.
