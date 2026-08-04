"""
Wraps RAGChain.query() to also push per-query metrics to Prometheus:
retrieval/generation/total latency histograms and a request counter.
"""
import sys
from pathlib import Path

# Ensure src/ is on the path when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

from rag_chain import RAGChain

app = FastAPI(title="RAG Observability Service")
rag = RAGChain(config_name="default")

REQUEST_COUNT = Counter("rag_requests_total", "Total RAG queries served")
RETRIEVAL_LATENCY = Histogram("rag_retrieval_latency_ms", "Retrieval stage latency (ms)")
GENERATION_LATENCY = Histogram("rag_generation_latency_ms", "Generation stage latency (ms)")
TOTAL_LATENCY = Histogram("rag_total_latency_ms", "End-to-end query latency (ms)")
CONTEXT_CHUNKS_RETURNED = Histogram("rag_context_chunks", "Number of chunks used per query")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    contexts: list[str]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    groundedness_score: float
    is_grounded: bool


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    resp = rag.query(req.question)
    contexts = [d["text"] for d in resp.retrieved_docs]

    REQUEST_COUNT.inc()
    RETRIEVAL_LATENCY.observe(resp.retrieval_latency_ms)
    GENERATION_LATENCY.observe(resp.generation_latency_ms)
    TOTAL_LATENCY.observe(resp.total_latency_ms)
    CONTEXT_CHUNKS_RETURNED.observe(len(contexts))

    return QueryResponse(
        answer=resp.answer,
        contexts=contexts,
        retrieval_latency_ms=resp.retrieval_latency_ms,
        generation_latency_ms=resp.generation_latency_ms,
        total_latency_ms=resp.total_latency_ms,
        groundedness_score=resp.groundedness_score,
        is_grounded=resp.is_grounded,
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok"}
