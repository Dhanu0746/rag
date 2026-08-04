# RAG with Built-In Evaluation, Guardrails & Observability

## The Problem

Most RAG systems are built and shipped without any way to know if they're actually working. Teams eyeball a few outputs, decide "looks good," and ship — then the system hallucinates, retrieves irrelevant context, or silently degrades when a chunking parameter or embedding model changes. There's no feedback loop: no automated way to catch a regression before a user does.

This project treats that gap as the actual problem to solve. Instead of building a naive RAG app, it builds a production-grade **RAG pipeline with an automated measurement and guardrail system attached to it** — every configuration change (chunk size, retrieval strategy, reranker on/off, query expansion) is run through a fixed evaluation set and scored, so decisions are based on numbers instead of vibes.

## Architecture

```
                    ┌─────────────────┐
   Markdown docs →  │  Ingestion       │  LlamaIndex: structure-aware chunking
                    │  (src/ingest.py) │  (Markdown headers → sentence splitter)
                    └────────┬─────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
          Chroma (dense)           BM25 (sparse)
                 │                       │
                 └───────────┬───────────┘
                    Multi-Query Expansion & RRF
                             │
                    Cross-encoder reranker
                             │
                    ┌────────┴─────────┐
                    │  Generation       │  LangChain + Groq / OpenAI / Ollama
                    │  (src/rag_chain)  │  grounded prompt & source citations
                    └────────┬──────────┘
                             │
             ┌───────────────┼───────────────┬────────────────┐
             │               │               │                │
       RAGAS eval      LangSmith trace  Prometheus metrics  Groundedness
     (src/eval.py)     (per-query)      (src/metrics)       Guardrail
             │               │               │                │
    eval_results/                          Grafana          UI Badges
    history.csv                                              
             │
   Streamlit dashboard
   (dashboard/app.py)
```

## Why each piece is there

| Component | Purpose |
|---|---|
| **Structure-aware chunking** (LlamaIndex) | Splits on Markdown headers first, preserving section context, instead of blind fixed-size splitting |
| **Hybrid retrieval** (Chroma + BM25 + RRF) | Dense search misses exact terms (error codes, names); BM25 catches them; RRF fuses both rankings seamlessly |
| **Multi-Query Expansion** | Generates alternative query perspectives using LLM heuristics and fuses candidate sets for higher recall |
| **Cross-encoder reranker** | First-stage retrieval is fast but approximate; reranking re-scores top candidates with a cross-encoder model |
| **Real-Time Groundedness Guardrail** | Evaluates response claims against retrieved source documents to detect potential hallucinations before display |
| **RAGAS evaluation harness** | Quantifies 4 core metrics (faithfulness, answer relevancy, context precision, context recall) against a golden dataset |
| **LangSmith + Prometheus + Grafana** | Per-query tracing + time-series metric tracking (latency breakdown, request counts) for production monitoring |
| **Multi-Provider Support** | Works with local Ollama or cloud providers (Groq, OpenAI) via environment configuration |

## Setup

```bash
python -m venv venv && source venv/bin/activate   # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# (Optional) For cloud deployment, set your API key in .env
echo "GROQ_API_KEY=your_groq_key_here" > .env
```

## Running It

**1. Ingest the corpus:**
```bash
python src/ingest.py --config-name default --chunk-size 512 --chunk-overlap 64
```

**2. Interactive Chat Application (Streamlit):**
```bash
streamlit run chat_app.py
```

**3. Run evaluation across pipeline configurations:**
```bash
python src/eval.py --config-name default --retrieval-mode dense --run-label "dense only"
python src/eval.py --config-name default --retrieval-mode hybrid --run-label "hybrid, no rerank"
python src/eval.py --config-name default --retrieval-mode hybrid --use-reranker --run-label "hybrid + rerank"
python src/eval.py --config-name default --retrieval-mode hybrid --use-reranker --use-query-expansion --run-label "hybrid + rerank + multi-query"
```

**4. View the Evaluation & Benchmark Dashboard:**
```bash
streamlit run dashboard/app.py
```

**5. Containerized Docker / Observability Stack:**
```bash
# Build and run application container
docker build -t rag-eval-observability .
docker run -p 8501:8501 rag-eval-observability

# Prometheus & Grafana stack
cd monitoring && docker compose up -d
```

## Resume & Interview Bullet Point

> **Built a Production-Grade RAG Pipeline with Automated Evaluation & Observability**
> - Architected a hybrid retrieval pipeline using **ChromaDB (dense)**, **BM25 (sparse)**, **Reciprocal Rank Fusion (RRF)**, **Multi-Query Expansion**, and **Cross-Encoder Reranking**.
> - Built an automated **RAGAS evaluation harness** to empirically benchmark pipeline configurations across Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
> - Developed a **real-time Groundedness Guardrail** to detect hallucination risks and integrated **Prometheus + Grafana time-series latency monitoring** alongside **LangSmith tracing**.

---

The strongest interview question this unlocks: **"How do you ensure your RAG application isn't hallucinating and degrading in production?"** — you have an actual, concrete system with empirical metrics and automated guardrails.
