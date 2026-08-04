"""
Enterprise RAG Chain with Built-In Groundedness Guardrail & Multi-Query Support.
"""

import time
from dataclasses import dataclass

from retriever import HybridRetriever
from llm import LocalLLM


@dataclass
class RAGResponse:
    answer: str
    retrieved_docs: list
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    groundedness_score: float = 1.0
    is_grounded: bool = True


class RAGChain:

    def __init__(
        self,
        config_name="default",
        model="gemma3:1b",
        retrieval_mode="hybrid",
        use_reranker=True,
        use_query_expansion=False,
        top_k=5,
    ):
        self.retriever = HybridRetriever(
            config_name=config_name,
            use_reranker=use_reranker,
        )

        self.llm = LocalLLM(
            model=model,
            temperature=0,
        )

        self.retrieval_mode = retrieval_mode
        self.use_query_expansion = use_query_expansion
        self.top_k = top_k

    def _format_context(self, retrieved_docs):
        contexts = []
        for i, doc in enumerate(retrieved_docs, start=1):
            contexts.append(
                f"[Source {i}]\n\n{doc['text']}\n"
            )
        return "\n\n".join(contexts)

    def _check_groundedness(self, answer: str, retrieved_docs: list) -> tuple[float, bool]:
        """
        Evaluate token/term overlap between answer claims and retrieved sources
        to detect potential hallucinations.
        """
        if not retrieved_docs or not answer:
            return 1.0, True

        combined_context = " ".join([d["text"].lower() for d in retrieved_docs])
        answer_words = [w.strip(".,!?()[]\"'").lower() for w in answer.split() if len(w) > 3]

        if not answer_words:
            return 1.0, True

        supported_words = sum(1 for w in answer_words if w in combined_context)
        score = round(supported_words / len(answer_words), 2)
        is_grounded = score >= 0.5

        return score, is_grounded

    def query(
        self,
        question: str,
        chat_history: list = None,
    ) -> RAGResponse:

        t0 = time.perf_counter()

        retrieved_docs = self.retriever.retrieve(
            query=question,
            top_k=self.top_k,
            mode=self.retrieval_mode,
            use_query_expansion=self.use_query_expansion,
        )

        t1 = time.perf_counter()

        context = self._format_context(retrieved_docs)

        history = ""
        if chat_history:
            for msg in chat_history:
                role = msg["role"].capitalize()
                history += f"{role}: {msg['content']}\n"

        full_context = f"Conversation History\n\n{history}\n\nKnowledge Base\n\n{context}\n"

        answer = self.llm.invoke(
            context=full_context,
            question=question,
        )

        t2 = time.perf_counter()

        groundedness_score, is_grounded = self._check_groundedness(
            answer, retrieved_docs
        )

        return RAGResponse(
            answer=answer,
            retrieved_docs=retrieved_docs,
            retrieval_latency_ms=(t1 - t0) * 1000,
            generation_latency_ms=(t2 - t1) * 1000,
            total_latency_ms=(t2 - t0) * 1000,
            groundedness_score=groundedness_score,
            is_grounded=is_grounded,
        )


if __name__ == "__main__":
    rag = RAGChain()
    response = rag.query("Explain Reciprocal Rank Fusion")
    print("Answer:", response.answer)
    print("Groundedness Score:", response.groundedness_score)