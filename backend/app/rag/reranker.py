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

        # Step 1: Compute neural cross-encoder scores if available
        if self._model is not None:
            pairs = [[query, c["text"][:600]] for c in candidates]
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

        # Step 2: Apply Legal Priority Domain Bonuses
        for item in scored_candidates:
            bonus = 0.0
            text_lower = item["text"].lower()
            section_lower = item.get("section_or_clause", "").lower()
            doc_lower = item.get("document_name", "").lower()

            # Rule 1: Prioritize Section 3(p) for Traditional Knowledge queries
            if ("patent" in query_lower or "traditional" in query_lower or "classical" in query_lower):
                if "3(p)" in section_lower or "3(p)" in text_lower:
                    bonus += 0.35
                if "3(e)" in section_lower or "synerg" in text_lower or "admixture" in text_lower:
                    bonus += 0.25

            # Rule 2: Prioritize 2024 and 2023 legal amendments
            if "2024" in query_lower or "amendment" in query_lower:
                if "2024" in doc_lower or "2024" in text_lower:
                    bonus += 0.30
                if "2023" in doc_lower or "2023" in text_lower:
                    bonus += 0.25

            # Rule 3: Prioritize ABS and Biodiversity exemptions
            if ("abs" in query_lower or "biodiversity" in query_lower or "ayush practitioner" in query_lower):
                if "biological_diversity" in doc_lower or "section 40" in section_lower or "section 40" in text_lower:
                    bonus += 0.30

            # Rule 4: International WIPO GRATK Treaty prioritization
            if ("wipo" in query_lower or "gratk" in query_lower or "disclosure" in query_lower or "origin" in query_lower):
                if "gratk" in doc_lower or "article 3" in section_lower or "article 3" in text_lower:
                    bonus += 0.35

            final_score = item["neural_rerank_score"] + bonus
            item["rerank_score"] = round(float(final_score), 4)

        # Step 3: Sort in descending order
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_n]
