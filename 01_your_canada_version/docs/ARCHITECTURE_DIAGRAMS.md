# Architecture Diagrams

Use this file as the single place for project diagrams when you explain the system to an interviewer.

Recommended speaking order:

1. `System Architecture Diagram`: the big picture
2. `Runtime Processing Flow`: what happens during one request
3. `End-to-End Demo Flow`: one realistic question journey
4. `RAG Index Build Flow`: how knowledge becomes searchable
5. `RAG Query-Time Retrieval Flow`: how the app finds relevant chunks
6. `LangGraph Workflow Diagram`: the orchestration path
7. `Reference-Layer Diagram`: how reference data supports answers

## 1. System Architecture Diagram

Use this when you want to explain the whole system at a high level.

Focus:

- major layers
- main data sources
- LLM-first intent/tool planner and capability planner
- orchestration path
- where LLM, RAG, tools, and compliance fit

![System Architecture Diagram](./assets/system_architecture_diagram.svg)

## 2. Runtime Processing Flow

Use this when you want to explain what the app does during one live request.

Focus:

- request lifecycle
- validation order
- when data is loaded
- when conversation resolution, intent/tool planning, tools, evidence validation, generation, and audit happen

This diagram uses a fixed SVG layout so it stays readable in VS Code preview.

![Runtime Processing Flow](./assets/runtime_processing_flow.svg)

## 3. End-to-End Demo Flow

Use this when you want to walk an interviewer through one concrete scenario such as:

`Why did my portfolio go down this month?`

Focus:

- step-by-step request journey
- access control
- intent/tool planning and capability planning
- optional RAG and tool usage
- compliance and audit

![End-to-End Demo Flow](./assets/end_to_end_demo_flow.svg)

## 4. RAG Index Build Flow

Use this to explain the offline preprocessing step.

Focus:

- reference files become chunks
- embeddings are optional
- output is a local index, not a vector database

![RAG Index Build Flow](./assets/rag_index_build_flow.svg)

## 5. RAG Query-Time Retrieval Flow

Use this to explain the online retrieval step.

Focus:

- index load
- retrieval backend choice
- ranking logic
- retrieved context for answer generation

![RAG Query-Time Retrieval Flow](./assets/rag_query_time_retrieval_flow.svg)

## 6. LangGraph Workflow Diagram

Use this to explain only the orchestration graph.

Focus:

- node order
- clean workflow design
- separation between intent/tool planning, deterministic tools, evidence validation, generation, and compliance

![LangGraph Workflow Diagram](./assets/langgraph_workflow_diagram.svg)

## 7. Reference-Layer Diagram

Use this to explain where the knowledge files fit.

Focus:

- which reference files feed the RAG layer
- how that layer supports explanation and hybrid answers

![Reference-Layer Diagram](./assets/reference_layer_diagram.svg)
