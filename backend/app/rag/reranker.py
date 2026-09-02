import logging
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

logger = logging.getLogger(__name__)

class LegalCrossEncoderReranker:
    """
    Reranks candidate statutory chunks using cross-attention relevance scoring
    augmented with legal domain clause prioritization.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model: Optional[CrossEncoder] = None
        if HAS_CROSS_ENCODER:
            try:
                logger.info(f"Loading Cross-Encoder model: {model_name}...")
                self._model = CrossEncoder(model_name)
                logger.info("✅ Cross-Encoder loaded.")
            except Exception as e:
                logger.warning(f"Could not load cross-encoder ({e}). Using heuristic reranker.")
                self._model = None

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 4) -> List[Dict[str, Any]]:
        """Reranks candidate statutory chunks and returns top_n items sorted by score."""
        if not candidates:
            return []

        query_lower = query.lower()
        scored_candidates = []

        # Step 1: Compute neural cross-encoder scores with contextual metadata
        if self._model is not None:
            pairs = []
            for c in candidates:
                statute = c.get("statute_title") or c.get("document_name") or ""
                sec = c.get("section_or_clause", "")
                text = c.get("text", "")[:500]
                doc_repr = f"{statute} {sec}: {text}".strip()
                pairs.append([query, doc_repr])

            raw_scores = self._model.predict(pairs)
            for idx, c in enumerate(candidates):
                c_copy = dict(c)
                c_copy["neural_rerank_score"] = float(raw_scores[idx])
                scored_candidates.append(c_copy)
        else:
            for c in candidates:
                c_copy = dict(c)
                c_copy["neural_rerank_score"] = c_copy.get("hybrid_rrf_score", 0.5)
                scored_candidates.append(c_copy)

        # Step 2: Use pure neural cross-encoder / RRF scores without artificial keyword bonuses
        for item in scored_candidates:
            final_score = item["neural_rerank_score"]
            item["rerank_score"] = round(float(final_score), 4)

        # Step 3: Sort in descending order
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_n]
