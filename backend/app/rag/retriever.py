import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.app.core.config import settings
from backend.app.core.qdrant_client_provider import QdrantClientProvider
from backend.app.rag.vector_store import QdrantHybridVectorStore
from backend.app.rag.reranker import LegalCrossEncoderReranker
from backend.app.rag.pdf_loader import PDFStatutoryLoader
from backend.app.rag.indexer import QdrantCorpusIndexer

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Unified legal retrieval interface coordinating:
    1. Dense semantic search in Qdrant collections.
    2. BM25 sparse keyword search for statutory sections.
    3. Cross-Encoder reranker prioritizing legal statutory bars.
    4. Strict jurisdiction isolation ('national' vs 'international').
    """

    def __init__(self):
        self.vector_store = QdrantHybridVectorStore()
        self.reranker = LegalCrossEncoderReranker()
        self._is_initialized = False

    def initialize(self, force_reindex: bool = False):
        """Loads statutory chunks into in-memory BM25 indices and ensures Qdrant collections are populated."""
        if self._is_initialized and not force_reindex:
            return

        raw_dir = settings.DATA_DIR / "raw_documents"
        if raw_dir.exists():
            loader = PDFStatutoryLoader(chunk_size=700, chunk_overlap=100)
            chunks_dict = loader.load_all_raw_documents(raw_dir)
            nat_chunks = chunks_dict["national"]
            intl_chunks = chunks_dict["international"]

            self.vector_store.register_chunks_for_sparse_search(
                national_chunks=nat_chunks,
                international_chunks=intl_chunks
            )

            # Check if Qdrant collections need initial vector population
            client = self.vector_store.client
            try:
                existing_cols = [c.name for c in client.get_collections().collections]
                needs_index = (
                    settings.QDRANT_COLLECTION_NATIONAL not in existing_cols or
                    client.get_collection(settings.QDRANT_COLLECTION_NATIONAL).points_count == 0
                )
                if needs_index and nat_chunks:
                    logger.info("🌿 Initializing Qdrant collections with dense statutory vectors...")
                    indexer = QdrantCorpusIndexer(client=client, embedder=self.vector_store.embedder)
                    indexer.index_chunks(nat_chunks, settings.QDRANT_COLLECTION_NATIONAL, batch_size=64)
                    indexer.index_chunks(intl_chunks, settings.QDRANT_COLLECTION_INTERNATIONAL, batch_size=64)
                    logger.info("✅ Qdrant collections populated with dense vectors.")
            except Exception as e:
                logger.warning(f"Could not connect to Qdrant for initial indexing ({e}). Using in-memory fallback.")

        self._is_initialized = True

    def search(
        self,
        query: str,
        jurisdiction: str = "national",
        top_k: int = 5,
        enable_reranking: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search + cross-encoder reranking.
        jurisdiction options: 'national', 'international', 'comparative'
        """
        if not self._is_initialized:
            self.initialize()

        target_jur = jurisdiction.lower()

        # Handle comparative dual retrieval
        if target_jur in ["comparative", "both"]:
            nat_candidates = self.vector_store.hybrid_search(query, jurisdiction="national", top_k=top_k * 2)
            intl_candidates = self.vector_store.hybrid_search(query, jurisdiction="international", top_k=top_k * 2)

            if enable_reranking:
                nat_top = self.reranker.rerank(query, nat_candidates, top_n=top_k)
                intl_top = self.reranker.rerank(query, intl_candidates, top_n=top_k)
            else:
                nat_top = nat_candidates[:top_k]
                intl_top = intl_candidates[:top_k]

            return {
                "national": nat_top,
                "international": intl_top
            }

        # Single jurisdiction retrieval
        effective_jur = "national" if target_jur in ["national", "india"] else "international"
        raw_candidates = self.vector_store.hybrid_search(query, jurisdiction=effective_jur, top_k=top_k * 2)

        if not raw_candidates:
            return []

        if enable_reranking:
            reranked_results = self.reranker.rerank(query, raw_candidates, top_n=top_k)
            return reranked_results

        return raw_candidates[:top_k]

# Global singleton instance
retriever = HybridRetriever()
