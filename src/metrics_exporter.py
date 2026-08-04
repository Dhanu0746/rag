"""
Wraps RAGChain.query() to also push per-query metrics to Prometheus:
retrieval/generation/total latency histograms and a request counter.
"""
import sys
from pathlib import Path

# Ensure src/ is on the path when run from project root
sys.path.insert(0, str(Path(__file__).parent))

import os

from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

from rag_chain import RAGChain

app = FastAPI(title="RAG Observability Service")

METRICS_CONFIG_NAME = os.getenv("METRICS_CONFIG_NAME", "default")
METRICS_API_KEY = os.getenv("METRICS_API_KEY")  # if unset, /query is left open (dev mode)

# Lazy-loaded so a fresh deploy (no index built yet for METRICS_CONFIG_NAME)
# doesn't crash the whole container at import time. Built on first request.
_rag = None
_rag_error = None


def get_rag():
    global _rag, _rag_error
    if _rag is None and _rag_error is None:
        try:
            _rag = RAGChain(config_name=METRICS_CONFIG_NAME)
        except FileNotFoundError as e:
            _rag_error = (
                f"No knowledge base found for config '{METRICS_CONFIG_NAME}' ({e}). "
                f"Index documents via the app or `python src/ingest.py "
                f"--config-name {METRICS_CONFIG_NAME}` first."
            )
    if _rag_error:
        raise HTTPException(status_code=503, detail=_rag_error)
    return _rag


def check_api_key(x_api_key: str = Header(default=None)):
    if METRICS_API_KEY and x_api_key != METRICS_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

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


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(check_api_key)])
def query(req: QueryRequest):
    rag = get_rag()
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