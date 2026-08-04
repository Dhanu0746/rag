"""
Hybrid retrieval: Dense (Chroma) + Sparse (BM25) + Reciprocal Rank Fusion +
Cross Encoder Reranking.

Returns structured retrieval results containing:
- text
- metadata
- similarity score
"""

import pickle
from pathlib import Path

import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder

STORE_DIR = Path(__file__).parent.parent / "storage"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridRetriever:

    def __init__(
        self,
        config_name: str = "default",
        use_reranker: bool = True,
    ):

        self.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME
        )

        chroma_client = chromadb.PersistentClient(
            path=str(STORE_DIR / "chroma")
        )

        self.collection = chroma_client.get_or_create_collection(
            name=config_name
        )

        with open(
            STORE_DIR / f"bm25_{config_name}.pkl",
            "rb",
        ) as f:

            bm25_data = pickle.load(f)

        self.bm25 = bm25_data["bm25"]
        self.bm25_texts = bm25_data["texts"]
        self.bm25_metadata = bm25_data["metadatas"]

        self.use_reranker = use_reranker

        if use_reranker:
            self.reranker = CrossEncoder(
                RERANKER_MODEL_NAME
            )

    # --------------------------------------------------

    def _dense_search(
        self,
        query: str,
        top_k: int,
    ):

        embedding = self.embed_model.get_query_embedding(
            query
        )

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

        docs = results["documents"][0]
        distances = results["distances"][0]
        metadata = results["metadatas"][0]

        output = []

        for doc, score, meta in zip(
            docs,
            distances,
            metadata,
        ):

            output.append(
                {
                    "text": doc,
                    "score": float(score),
                    "metadata": meta,
                }
            )

        return output

    # --------------------------------------------------

    def _sparse_search(
        self,
        query: str,
        top_k: int,
    ):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked = sorted(
            zip(
                self.bm25_texts,
                scores,
                self.bm25_metadata,
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        output = []

        for text, score, meta in ranked[:top_k]:

            output.append(
                {
                    "text": text,
                    "score": float(score),
                    "metadata": meta,
                }
            )

        return output

    # --------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        dense_results,
        sparse_results,
        k: int = 60,
    ):

        scores = {}
        lookup = {}

        for rank, item in enumerate(dense_results):

            text = item["text"]

            lookup[text] = item

            scores[text] = (
                scores.get(text, 0)
                + 1 / (k + rank + 1)
            )

        for rank, item in enumerate(sparse_results):

            text = item["text"]

            lookup[text] = item

            scores[text] = (
                scores.get(text, 0)
                + 1 / (k + rank + 1)
            )

        fused = []

        for text, fusion_score in sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        ):

            item = lookup[text]

            item["fusion_score"] = fusion_score

            fused.append(item)

        return fused

    # --------------------------------------------------

    # --------------------------------------------------

    def _deduplicate(self, candidates: list) -> list:
        """
        Remove duplicate chunks based on a normalized text fingerprint.

        Two chunks are considered duplicates if their first 200 characters
        (lowercased and whitespace-normalized) match.  The highest-ranked
        occurrence is kept; subsequent duplicates are dropped.
        """
        seen = set()
        deduped = []
        for item in candidates:
            # Build a compact fingerprint from the start of the text
            fingerprint = " ".join(item["text"].lower().split())[:200]
            if fingerprint not in seen:
                seen.add(fingerprint)
                deduped.append(item)
        return deduped

    # --------------------------------------------------

    def _generate_query_variations(self, query: str) -> list[str]:
        """
        Generate variations of the query for Multi-Query Expansion.
        """
        variations = [query]
        # Basic heuristic query variations to broaden recall
        words = query.split()
        if len(words) > 3:
            variations.append(" ".join(words[: len(words) // 2]))
            variations.append(" ".join(words[len(words) // 2 :]))
        return variations

    # --------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 20,
        mode: str = "hybrid",
        use_query_expansion: bool = False,
    ):

        queries = [query]
        if use_query_expansion:
            queries = self._generate_query_variations(query)

        all_candidates = []

        for q in queries:
            if mode == "dense":
                candidates = self._dense_search(q, retrieval_k)
            elif mode == "sparse":
                candidates = self._sparse_search(q, retrieval_k)
            else:
                dense = self._dense_search(q, retrieval_k)
                sparse = self._sparse_search(q, retrieval_k)
                candidates = self._reciprocal_rank_fusion(dense, sparse)

            all_candidates.append(candidates)

        if len(all_candidates) > 1:
            # Multi-query fusion across query variations
            fused_candidates = all_candidates[0]
            for cand_set in all_candidates[1:]:
                fused_candidates = self._reciprocal_rank_fusion(
                    fused_candidates, cand_set
                )
            candidates = fused_candidates
        else:
            candidates = all_candidates[0]

        # Deduplicate before reranking so the reranker scores unique chunks only
        candidates = self._deduplicate(candidates)

        if self.use_reranker and candidates:

            pairs = [
                [query, item["text"]]
                for item in candidates
            ]

            rerank_scores = self.reranker.predict(
                pairs
            )

            for item, score in zip(
                candidates,
                rerank_scores,
            ):

                item["rerank_score"] = float(score)

            candidates = sorted(
                candidates,
                key=lambda x: x["rerank_score"],
                reverse=True,
            )

        return candidates[:top_k]