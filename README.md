# ĐỀ CƯƠNG ĐÀO TẠO AI CHI TIẾT

> Tổng hợp đầy đủ từ bảng training plan. Gồm 4 module chính, kéo dài từ **14/05/2026** đến **09/06/2026**.

---

## 📘 MODULE 1: [L2_AI_AIF] AI Fundamentals & RAG (v1.0)

> **Thời gian:** 14/05/2026 → 15/05/2026 (2 ngày)

---

### Ngày 40 (14/05/2026): Introduction to AI & Generative AI & Introduction to RAG & Theoretical Foundations

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Introduction to AI & Generative AI** | Self Learning | 2h | HuuTN | Lecture |
| 2 | **Workshop (Guided Lab):** Setup Environment (Python venv/Conda), install dependencies (LangChain, OpenAI/Anthropic SDKs), setup API key from Bedrock. Step-by-step: Explore LLM APIs (first completion call), Understanding RAG components (Visualizing flow from Query → Vector Store → Response) | Virtual training | 2h | HuuTN | Lecture |
| 3 | **Lab: Understanding RAG components** & **Lab: Implementing Indexing & Retrieval techniques** | Offline | 3h | — | Assignment/Lab |

**Kiến thức cần nắm:**
- Introduction to AI & GenAI
- RAG Theoretical Foundations
- Concept/Lecture

**Tài liệu:**
- Slide: AI Fundamental
- 🔗 https://docs.langchain.com/oss/python/integrations/providers/aws
- 🔗 https://reference.langchain.com/python/langchain-aws/vectorstores/s3_vectors/base/AmazonS3Vectors

**Lab files:**
- `fresher/1-Basic-AI-Fundamentals/3-Assignments/LAB1-understanding-rag-components-questions.ipynb` (approach sử dụng ChatOpenAI)
- `fresher/01-Basic-AI-Fundamentals/03-Assignments/lab2-implementing-indexing-and-retrieval-techniques-questions.ipynb`
- ⚠️ Cần chạy và convert sang Bedrock Langchain LLM nếu cần. Chạy để test API hoạt động.

---

### Ngày 41 (15/05/2026): Modern RAG Architecture (Indexing, Retrieval, Generation) & Course Revision

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Modern RAG Architecture (Indexing, Retrieval, Generation)** | Self Learning | 4h | HuuTN | Lecture |
| 2 | **Self review / Cross review** | Offline | 30m | HuuTN | Support/Guide |
| 3 | **Course revision** | Offline | 2h | — | Assignment/Lab |
| 4 | **Theory Exam:** Basic quiz on AI and RAG systems | Offline | 45m | HienVT61 | Test/Quiz |

**Kiến thức cần nắm:**
- Modern RAG Architecture (Indexing, Retrieval, Generation)
- LangChain Framework & Core Components
- Generation Strategies & Prompt Engineering

**Tài liệu:**
- `fresher/01-Basic-AI-Fundamentals/01-Knowledge/03-modern-rag-architecture.pdf`
- `fresher/01-Basic-AI-Fundamentals/01-Knowledge/04-langchain-framework-and-core-components.pdf`
- 🔗 https://www.promptingguide.ai/techniques/fewshot

**Lab files:**
- `fresher/01-Basic-AI-Fundamentals/03-Assignments/lab2-implementing-indexing-and-retrieval-techniques-questions.ipynb`

---
---

## 📗 MODULE 2: [L2_AI_RAGO] AI Advanced: RAG & Optimization (v1.0)

> **Thời gian:** 18/05/2026 → 25/05/2026 (6 ngày)

---

### Ngày 42 (18/05/2026): Advanced Indexing

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Advanced Indexing** | Self Learning | 3h | HuuTN | Lecture |
| 2 | **Quiz 01** | Offline | 30m | HienVT61 | Test/Quiz |
| 3 | **Lab: Build Semantic Chunker & HNSW Vector DB** | Offline | 2h 30m | — | Assignment/Lab |
| 4 | **Self review / Cross review** | Offline | 1h | HuuTN | Support/Guide |

**Kiến thức cần nắm:**
- Semantic Chunking: Core Idea & Process
- HNSW Index: Mechanism & Hierarchy Layering
- HNSW Parameters: M, ef_construction, ef_search
- Comparison: Recursive vs. Semantic Chunking

**Tài liệu:**
- `fresher/02-Advanced-AI-Advanced/01-rag-and-optimization/01-Knowledge/01-advanced-indexing.pdf`

**Lab files:**
- `fresher/02-Advanced-AI-Advanced/01-rag-and-optimization/03-Assignments/01-advanced-indexing-assignment.pdf`

---

### Ngày 43 (19/05/2026): Hybrid Search

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Hybrid Search** | Self Learning | 3h | HuuTN | Lecture |
| 2 | **Quiz 02** | Offline | 30m | HienVT61 | Test/Quiz |
| 3 | **Lab: Combine Vector Search with BM25 & Apply RRF** | Offline | 2h 30m | — | Assignment/Lab |
| 4 | **Self review / Cross review** | Offline | 1h | HuuTN | Support/Guide |

**Kiến thức cần nắm:**
- Limitations of Vector Search & BM25 Introduction
- BM25 Core Principles: TF Saturation & IDF
- Hybrid Search: Sparse & Dense Retriever Parallel Execution
- Reciprocal Rank Fusion (RRF) Formula & Calculation

**Tài liệu:**
- `fresher/02-Advanced-AI-Advanced/01-rag-and-optimization/01-Knowledge/02-hybrid-search.pdf`

**Lab files:**
- `fresher/02-Advanced-AI-Advanced/01-rag-and-optimization/03-Assignments/02-hybrid-search-assignment.pdf`

---

### Ngày 44 (20/05/2026): Query Transformation

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Query Transformation** | Self Learning | 3h | HuuTN | Lecture |
| 2 | **Quiz 03** | Offline | 30m | HienVT61 | Test/Quiz |
| 3 | **Lab: Build HyDE Prompts & Query Decomposition Loop** | Offline | 2h 30m | — | Assignment/Lab |
| 4 | **Self review / Cross review** | Offline | 1h | HuuTN | Support/Guide |

**Kiến thức cần nắm:**
- Query Issues: User Intent & Semantic Asymmetry
- Hypothetical Document Embeddings (HyDE) Mechanism
- Query Decomposition: Breakdown, Retrieval & Synthesis

**Tài liệu:**
- `fresher/02-Advanced-AI-Advanced/01-rag-and-optimization/01-Knowledge/03-query-transformation.pdf`

**Lab files:**
- `fresher/02-Advanced-AI-Advanced/01-rag-and-optimization/03-Assignments/03-query-transformation-assignment.pdf`

---

### Ngày 45 (21/05/2026): Post-Retrieval Processing

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Post-Retrieval Processing** | Self Learning | 3h | HuuTN | Lecture |
| 2 | **Quiz 04** | Offline | 30m | HienVT61 | Test/Quiz |
| 3 | **Workshop (Review & Guided Lab):** 1. Review Assignment for training unit 01, 02, 03. 2. Step-by-step: Implement Cross-Encoder Pipeline & MMR Retrieval. Sharing: Best Practices cho Advanced Indexing, Hybrid Search, Query Transformation, Post-Retrieval Processing | Virtual training | 2h | HuuTN | Lecture |
| 4 | **Self review / Cross review** | Offline | 1h | HuuTN | Support/Guide |

**Kiến thức cần nắm:**
- Information Noise & Weakness of Bi-Encoder
- Cross-Encoder Architecture for Re-ranking
- Maximal Marginal Relevance (MMR) & Diversity Filtering

**Tài liệu:**
- `fresher/02-Advanced-AI-Advanced/01-rag-and-optimization/01-Knowledge/04-post-retrieval.pdf`
- 🔗 https://viblo.asia/p/bi-kip-vo-cong-thuong-thua-giup-cai-tien-ung-dung-retrieval-augmented-generation-rag-AZoJjra2JY7

---

### Ngày 46 (22/05/2026): GraphRAG Implementation

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **GraphRAG Implementation** | Self Learning | 3h | HuuTN | Lecture |
| 2 | **Quiz 05** | Offline | 30m | HienVT61 | Test/Quiz |
| 3 | **Lab: FSoft_HR.pdf Extraction & GraphCypherQAChain** | Offline | 2h 30m | — | Assignment/Lab |
| 4 | **Self review / Cross review** | Offline | 1h | HuuTN | Support/Guide |

**Kiến thức cần nắm:**
- GraphRAG Architecture breakdown
- Entity Extraction & Domain-Specific Data Classes
- Ingesting Structured Data into Neo4j Schema
- Cypher Query Generation & NLP Translation

**Tài liệu:**
- `fresher/02-Advanced-AI-Advanced/01-rag-and-optimization/01-Knowledge/05-graph-rag-implementation.pdf`

**Lab files:**
- `fresher/02-Advanced-AI-Advanced/01-rag-and-optimization/03-Assignments/05-graph-rag-implementation-assignment.pdf`

---

### Ngày 47 (25/05/2026): Final Module Evaluation — RAG & Optimization

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Workshop (Review & Guided Lab):** Prepare and guiding what is the expect of the final exam | Virtual training | 2h | HuuTN | Lecture |
| 2 | **Part 1: Theory Exam (Quiz Concept)** | Offline | 1h 30m | HienVT61 | Exam |
| 3 | **Part 2: Practical Exam (Coding)** | Offline | 4h | HienVT61 | Exam |

---
---

## 📕 MODULE 3: [L2_AI_LLMO] LLMOps and Evaluation (v1.0)

> **Thời gian:** 26/05/2026 → 29/05/2026 (4 ngày)

---

### Ngày 48 (26/05/2026): Ragas Evaluation Metrics

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Ragas Evaluation Metrics** | Self Learning | 3h | HuuTN | Lecture |
| 2 | **Quiz** | Offline | 30m | HienVT61 | Test/Quiz |
| 3 | **Lab: Implementing Ragas Metrics** | Offline | 2h 30m | — | Assignment/Lab |
| 4 | **Self review / Cross review** | Offline | 1h | HuuTN | Support/Guide |

**Kiến thức cần nắm:**
- Ragas Evaluation Metrics: Introduction to the "RAG Triad" (Faithfulness, Answer Relevance, Context Relevance)
- Metric Deep Dive: Understanding Context Precision and Context Recall for retrieval performance
- Component-Wise vs. End-to-End Evaluation: How to isolate issues in the retriever vs. the generator

**Tài liệu:**
- `fresher/02-Advanced-AI-Advanced/02-llmops-and-evaluation/01-Knowledge/01-ragas-evaluation-metrics.pdf`

**Lab files:**
- `fresher/02-Advanced-AI-Advanced/02-llmops-and-evaluation/03-Assignments/01-ragas-evaluation-metrics-assignment.pdf`

---

### Ngày 49 (27/05/2026): Observability: LangFuse & LangSmith

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Observability: LangFuse & LangSmith** | Self Learning | 3h | HuuTN | Lecture |
| 2 | **Quiz** | Offline | 30m | HienVT61 | Test/Quiz |
| 3 | **Lab: Setting up Tracing & Dashboards** | Offline | 2h 30m | — | Assignment/Lab |
| 4 | **Self review / Cross review** | Offline | 1h | HuuTN | Support/Guide |

**Kiến thức cần nắm:**
- Observability: The importance of "Tracing" in non-deterministic AI systems
- Key Pillars: Latency tracking, Token cost monitoring, and Prompt Versioning

**Tài liệu:**
- `fresher/02-Advanced-AI-Advanced/02-llmops-and-evaluation/01-Knowledge/02-observability.pdf`

**Lab files:**
- `fresher/02-Advanced-AI-Advanced/02-llmops-and-evaluation/03-Assignments/02-observability-assignment.pdf`

---

### Ngày 50 (28/05/2026): Experiment Comparison: Naive, Graph, Hybrid

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Experiment Comparison: Naive, Graph, Hybrid** | Self Learning | 3h | HuuTN | Lecture |
| 2 | **Quiz** | Offline | 30m | HienVT61 | Test/Quiz |
| 3 | **Workshop (Review & Guided Lab):** 1. Review Assignment for training unit 01, 02. 2. Step-by-step: Lab: Running RAG Architecture Experiments. Sharing: Best Practices for evaluation metrics, how to observability Agent application | Virtual training | 3h | HuuTN | Lecture |
| 4 | **Self review / Cross review** | Offline | 1h | HuuTN | Support/Guide |

**Kiến thức cần nắm:**
- Experiment Comparison: Naive, Graph, Hybrid

**Tài liệu:**
- `fresher/02-Advanced-AI-Advanced/02-llmops-and-evaluation/01-Knowledge/03-experiment-comparison.pdf`

---

### Ngày 51 (29/05/2026): Final Module Evaluation — LLMOps and Evaluation

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Workshop (Review & Guided Lab):** Prepare and guiding what is the expect of the final exam | Virtual training | 2h | HuuTN | Lecture |
| 2 | **Part 1: Theory Exam (Quiz Concept)** | Offline | 1h | HienVT61 | Exam |
| 3 | **Part 2: Practical Exam (Coding)** | Offline | 4h | HienVT61 | Exam |

---
---

## 📙 MODULE 4: [L2_AI_LGAA] LangGraph and Agentic AI (v1.0)

> **Thời gian:** 01/06/2026 → 09/06/2026 (7 ngày)

---

### Ngày 52 (01/06/2026): LangGraph Foundations & State Management

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **LangGraph Foundations & State Management** | Self Learning | 3h | HuuTN | Lecture |
| 2 | **Quiz** | Offline | 30m | HienVT61 | Test/Quiz |
| 3 | **Workshop (Guided Lab):** Guiding how to create the first graph. Step-by-step: Create "Hello World" Graph (single node), Visualizing Graph (get_graph().print_ascii() / Mermaid diagrams), Hands-On: Tracing execution with LangSmith. Sharing: Best practices on state schema design (keeping state minimal) | Virtual training | 2h | HuuTN | Lecture |
| 4 | **Self review / Cross review** | Offline | 2h 30m | HuuTN | Support/Guide |

**Kiến thức cần nắm:**
- LangGraph Foundations: StateGraph, Nodes, and Edges
- State Management: Defining and updating the TypedDict state
- Core Workflow: Understanding the compilation process and the invoke vs stream methods

**Tài liệu:**
- `fresher/02-Advanced-AI-Advanced/03-langgraph-and-agentic-ai/01-Knowledge/01-langgraph-foundations-state-management.pdf`
- 🔗 https://docs.langchain.com/oss/javascript/langgraph/quickstart#full-code-example

---

### Ngày 53 (02/06/2026): Agentic Patterns: Multi-Expert Research Agent

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Agentic Patterns: Multi-Expert Research Agent** | Self Learning | 3h | HuuTN | Lecture |
| 2 | **Quiz** | Offline | 30m | HienVT61 | Test/Quiz |
| 3 | **Practice: Multi-Expert ReAct Agent** | Offline | 2h 30m | — | Assignment/Lab |
| 4 | **Self review / Cross review** | Offline | 2h | HuuTN | Support/Guide |

**Kiến thức cần nắm:**
- Agentic Patterns: Multi-Expert Research Agent

**Tài liệu:**
- `fresher/02-Advanced-AI-Advanced/03-langgraph-and-agentic-ai/01-Knowledge/02-agentic-patterns-reflection-planning.pdf`

**Lab files:**
- `fresher/02-Advanced-AI-Advanced/03-langgraph-and-agentic-ai/03-Assignments/02-agentic-patterns-assignment.pdf`

---

### Ngày 54 (03/06/2026): Tool Calling & Tavily Search

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Tool Calling & Tavily Search** | Self Learning | 3h | HuuTN | Lecture |
| 2 | **Quiz** | Offline | 30m | HienVT61 | Test/Quiz |
| 3 | **Practice: Tool Calling** | Offline | 2h 30m | — | Assignment/Lab |
| 4 | **Self review / Cross review** | Offline | 2h | HuuTN | Support/Guide |

**Kiến thức cần nắm:**
- Tool Calling & Tavily Search

**Tài liệu:**
- `fresher/02-Advanced-AI-Advanced/03-langgraph-and-agentic-ai/01-Knowledge/03-tool-calling-tavily-search.pdf`

**Lab files:**
- `fresher/02-Advanced-AI-Advanced/03-langgraph-and-agentic-ai/03-Assignments/03-tool-calling-assignment.pdf`

---

### Ngày 55 (04/06/2026): Multi-Agent Collaboration

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Multi-Agent Collaboration** | Self Learning | 3h | HuuTN | Lecture |
| 2 | **Quiz** | Offline | 30m | HienVT61 | Test/Quiz |
| 3 | **Practice: Multi-Agent System** | Offline | 2h 30m | — | Assignment/Lab |
| 4 | **Self review / Cross review** | Offline | 2h | HuuTN | Support/Guide |

**Kiến thức cần nắm:**
- Multi-Agent Collaboration

**Tài liệu:**
- `fresher/02-Advanced-AI-Advanced/03-langgraph-and-agentic-ai/01-Knowledge/04-multi-agent-collaboration.pdf`

**Lab files:**
- `fresher/02-Advanced-AI-Advanced/03-langgraph-and-agentic-ai/03-Assignments/04-multi-agent-collaboration-assignment.pdf`

---

### Ngày 56 (05/06/2026): Human-in-the-Loop & Persistence

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Human-in-the-Loop & Persistence** | Self Learning | 3h | HuuTN | Lecture |
| 2 | **Quiz** | Offline | 30m | HienVT61 | Test/Quiz |
| 3 | **Practice: Persistent & HITL Agent** | Offline | 2h 30m | — | Assignment/Lab |
| 4 | **Self review / Cross review** | Offline | 2h | HuuTN | Support/Guide |

**Kiến thức cần nắm:**
- Human-in-the-Loop & Persistence

**Tài liệu:**
- `fresher/02-Advanced-AI-Advanced/03-langgraph-and-agentic-ai/01-Knowledge/05-human-in-the-loop-persistence.pdf`

**Lab files:**
- `fresher/02-Advanced-AI-Advanced/03-langgraph-and-agentic-ai/03-Assignments/05-human-in-the-loop-assignment.pdf`

---

### Ngày 57 (08/06/2026): Final Module Evaluation — LangGraph and Agentic AI

| # | Nội dung | Hình thức | Thời lượng | Giảng viên | Loại |
|---|----------|-----------|------------|------------|------|
| 1 | **Workshop (Review & Guided Lab):** Prepare and guiding what is the expect of the final exam | Virtual training | 2h 30m | HuuTN | Lecture |
| 2 | **Part 1: Theory Exam (Quiz Concept)** | Offline | 1h 30m | HienVT61 | Exam |
| 3 | **Part 2: Practical Exam (Coding)** | Offline | 4h | HienVT61 | Exam |

**Exam files:**
- `fresher/02-Advanced-AI-Advanced/03-langgraph-and-agentic-ai/05-Exams/03-langgraph-and-agentic-ai-Project-Theory.xlsx`
- `fresher/02-Advanced-AI-Advanced/03-langgraph-and-agentic-ai/05-Exams/03-langgraph-and-agentic-ai-Project-Exams.pdf`

---

### Ngày 58 (09/06/2026): Audit

| # | Nội dung | Hình thức | Thời lượng |
|---|----------|-----------|------------|
| 1 | **Audit (Sáng)** | Offline | 4h |
| 2 | **Audit (Chiều)** | Offline | 4h |

---
---

## 📊 TỔNG KẾT CÁC CHỦ ĐỀ CẦN HỌC

### Module 1 — AI Fundamentals & RAG
1. Introduction to AI & Generative AI
2. RAG Theoretical Foundations
3. Setup Environment (Python venv/Conda, LangChain, OpenAI/Anthropic SDKs, Bedrock)
4. Explore LLM APIs: First completion call
5. Understanding RAG components (Query → Vector Store → Response)
6. Implementing Indexing & Retrieval techniques
7. Modern RAG Architecture (Indexing, Retrieval, Generation)
8. LangChain Framework & Core Components
9. Generation Strategies & Prompt Engineering

### Module 2 — RAG & Optimization
1. **Advanced Indexing:** Semantic Chunking, HNSW Index (M, ef_construction, ef_search), Recursive vs Semantic Chunking
2. **Hybrid Search:** BM25 (TF Saturation & IDF), Sparse & Dense Retriever, Reciprocal Rank Fusion (RRF)
3. **Query Transformation:** User Intent & Semantic Asymmetry, HyDE Mechanism, Query Decomposition
4. **Post-Retrieval Processing:** Bi-Encoder weaknesses, Cross-Encoder Re-ranking, MMR & Diversity Filtering
5. **GraphRAG Implementation:** Entity Extraction, Neo4j Schema, Cypher Query Generation, NLP Translation

### Module 3 — LLMOps and Evaluation
1. **Ragas Evaluation Metrics:** RAG Triad (Faithfulness, Answer Relevance, Context Relevance), Context Precision & Recall, Component-Wise vs End-to-End Evaluation
2. **Observability:** LangFuse & LangSmith, Tracing, Latency tracking, Token cost monitoring, Prompt Versioning
3. **Experiment Comparison:** Naive vs Graph vs Hybrid RAG architectures

### Module 4 — LangGraph and Agentic AI
1. **LangGraph Foundations:** StateGraph, Nodes, Edges, TypedDict state, invoke vs stream
2. **Agentic Patterns:** Multi-Expert Research Agent, ReAct Agent
3. **Tool Calling:** Tavily Search integration
4. **Multi-Agent Collaboration:** Multi-Agent System design
5. **Human-in-the-Loop & Persistence:** Persistent & HITL Agent patterns
