import re
import math
import logging
from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

from backend.app.core.config import settings
from backend.app.core.qdrant_client_provider import QdrantClientProvider
from backend.app.rag.embedder import DenseEmbedder
from backend.app.rag.schemas import LegalChunk

logger = logging.getLogger(__name__)

class QdrantHybridVectorStore:
    """
    Hybrid retriever combining Qdrant dense vector search with in-memory BM25 sparse search
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, client: Optional[QdrantClient] = None, embedder: Optional[DenseEmbedder] = None):
        self.client = client or QdrantClientProvider.get_client()
        self.embedder = embedder or DenseEmbedder.get_instance()
        
        # In-memory document chunk store for BM25 and fallback retrieval
        self.chunks_by_jurisdiction: Dict[str, List[Dict[str, Any]]] = {
            "national": [],
            "international": []
        }
        self.bm25_indices: Dict[str, Optional[BM25Okapi]] = {
            "national": None,
            "international": None
        }
        self.in_memory_embeddings: Dict[str, Optional[np.ndarray]] = {
            "national": None,
            "international": None
        }

    def register_chunks_for_sparse_search(self, national_chunks: List[LegalChunk], international_chunks: List[LegalChunk]):
        """Builds in-memory BM25 keyword indices and in-memory dense embedding matrices."""
        self.chunks_by_jurisdiction["national"] = [c.to_qdrant_payload() for c in national_chunks]
        self.chunks_by_jurisdiction["international"] = [c.to_qdrant_payload() for c in international_chunks]

        for jur in ["national", "international"]:
            chunks = self.chunks_by_jurisdiction[jur]
            if chunks:
                tokenized_corpus = [
                    self._tokenize(f"{c.get('statute_title', '')} {c.get('document_name', '')} {c.get('section_or_clause', '')} {c.get('text', '')}")
                    for c in chunks
                ]
                self.bm25_indices[jur] = BM25Okapi(tokenized_corpus)
                logger.info(f"✅ Built BM25 sparse index for {jur} ({len(chunks)} documents).")

        self._ensure_in_memory_embeddings()

    def compute_cache_fingerprint(
        self,
        corpus_version: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        embedding_model_version: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        preprocessing_version: Optional[str] = None
    ) -> str:
        """
        Computes a deterministic SHA-256 fingerprint covering:
        - Corpus version & document identifiers/versions
        - Embedding model name and version
        - Chunk size & chunk overlap
        - Preprocessing pipeline version
        - All chunk IDs and exact text content
        """
        import hashlib
        hasher = hashlib.sha256()

        c_ver = corpus_version or getattr(settings, "CORPUS_VERSION", "2024.1")
        m_name = embedding_model_name or getattr(self.embedder, "model_name", "sentence-transformers/all-MiniLM-L6-v2")
        m_ver = embedding_model_version or getattr(self.embedder, "model_version", "1.0.0")
        c_size = chunk_size if chunk_size is not None else getattr(settings, "CHUNK_SIZE", 750)
        c_overlap = chunk_overlap if chunk_overlap is not None else getattr(settings, "CHUNK_OVERLAP", 100)
        p_ver = preprocessing_version or getattr(settings, "PREPROCESSING_VERSION", "2.1.0")

        hasher.update(f"corpus_ver={c_ver};model={m_name};model_ver={m_ver};size={c_size};overlap={c_overlap};preproc={p_ver};".encode("utf-8"))

        for jur in ["national", "international"]:
            chunks = self.chunks_by_jurisdiction.get(jur, [])
            hasher.update(f"{jur}_{len(chunks)}".encode("utf-8"))
            for c in chunks:
                doc_name = c.get("document_name", "")
                doc_ver = str(c.get("document_version", "1.0"))
                hasher.update(f"doc={doc_name};ver={doc_ver};".encode("utf-8"))
                hasher.update(c.get("chunk_id", "").encode("utf-8"))
                hasher.update(c.get("text", "").encode("utf-8"))

        return hasher.hexdigest()

    def _ensure_in_memory_embeddings(self):
        """Loads cached dense embeddings from disk or computes and persists them with deterministic fingerprint."""
        cache_path = getattr(settings, "EMBEDDINGS_CACHE_FILE", None)
        nat_len = len(self.chunks_by_jurisdiction["national"])
        intl_len = len(self.chunks_by_jurisdiction["international"])
        expected_fp = self.compute_cache_fingerprint()

        if cache_path and cache_path.exists():
            try:
                data = np.load(cache_path)
                cached_fp = str(data["fingerprint"]) if "fingerprint" in data else None
                if cached_fp == expected_fp and "national" in data and "international" in data:
                    nat_arr = data["national"]
                    intl_arr = data["international"]
                    if nat_arr.shape[0] == nat_len and intl_arr.shape[0] == intl_len:
                        self.in_memory_embeddings["national"] = nat_arr
                        self.in_memory_embeddings["international"] = intl_arr
                        logger.info(f"📦 Validated fingerprint [{expected_fp[:8]}] & loaded embeddings from {cache_path}.")
                        return
                    else:
                        logger.info(f"🔄 Chunk shape mismatch in cache. Recomputing embeddings...")
                else:
                    logger.info(f"🔄 Cache fingerprint mismatch (cached: {cached_fp[:8] if cached_fp else 'None'} vs current: {expected_fp[:8]}). Recomputing embeddings...")
            except Exception as e:
                logger.warning(f"Failed to load embeddings cache: {e}. Recomputing...")

        # Compute embeddings for international
        if intl_len > 0 and self.in_memory_embeddings["international"] is None:
            logger.info(f"Computing in-memory dense embeddings for international corpus ({intl_len} chunks)...")
            texts = [c.get("text", "") for c in self.chunks_by_jurisdiction["international"]]
            emb = self.embedder._model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
            self.in_memory_embeddings["international"] = emb

        # Compute embeddings for national
        if nat_len > 0 and self.in_memory_embeddings["national"] is None:
            logger.info(f"Computing in-memory dense embeddings for national corpus ({nat_len} chunks)...")
            texts = [c.get("text", "") for c in self.chunks_by_jurisdiction["national"]]
            emb = self.embedder._model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
            self.in_memory_embeddings["national"] = emb

        # Save to disk cache with fingerprint
        if cache_path and self.in_memory_embeddings["national"] is not None and self.in_memory_embeddings["international"] is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    national=self.in_memory_embeddings["national"],
                    international=self.in_memory_embeddings["international"],
                    fingerprint=expected_fp
                )
                logger.info(f"💾 Cached dense embeddings with fingerprint [{expected_fp[:8]}] to {cache_path}.")
            except Exception as e:
                logger.warning(f"Failed to cache embeddings: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes text preserving section numbers like 3(p) as 'section 3p', '3p', '3', 'p'."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        subsections = re.findall(r'(\d+)\s*\(([a-zA-Z0-9]+)\)', text.lower())
        for sec, sub in subsections:
            tokens.append(f"section_{sec}_{sub}")
            tokens.append(f"section_{sec}{sub}")
            tokens.append(f"{sec}{sub}")
            tokens.append(f"({sub})")
            tokens.append(sec)
            tokens.append(sub)
        
        sec_matches = re.findall(r'section\s+(\d+[a-zA-Z]*)', text.lower())
        for sm in sec_matches:
            tokens.append(f"section_{sm}")
            tokens.append(sm)
            
        return tokens

    def dense_search(self, query: str, jurisdiction: str = "national", top_k: int = 10) -> List[Dict[str, Any]]:
        """Searches Qdrant using dense vector similarity, falling back to genuine in-memory cosine search."""
        collection_name = (
            settings.QDRANT_COLLECTION_NATIONAL 
            if jurisdiction == "national" 
            else settings.QDRANT_COLLECTION_INTERNATIONAL
        )
        
        query_vector = self.embedder.embed_text(query)

        try:
            # Check if collection exists in Qdrant and has points
            collections = [c.name for c in self.client.get_collections().collections]
            if collection_name not in collections:
                return self._fallback_in_memory_dense_search(query_vector, jurisdiction, top_k)

            if hasattr(self.client, 'query_points'):
                response = self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True
                )
                search_results = response.points
            elif hasattr(self.client, 'search'):
                search_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True
                )
            else:
                return self._fallback_in_memory_dense_search(query_vector, jurisdiction, top_k)

            results = []
            for hit in search_results:
                payload = hit.payload or {}
                results.append({
                    "chunk_id": payload.get("chunk_id", str(hit.id)),
                    "text": payload.get("text", ""),
                    "document_name": payload.get("document_name", ""),
                    "statute_title": payload.get("statute_title", ""),
                    "section_or_clause": payload.get("section_or_clause", "General"),
                    "jurisdiction": payload.get("jurisdiction", jurisdiction),
                    "page_numbers": payload.get("page_numbers", []),
                    "source_url": payload.get("official_source_url", ""),
                    "is_statutory_bar": payload.get("is_statutory_bar", False),
                    "dense_score": hit.score if hasattr(hit, 'score') else 0.5
                })
            return results
        except Exception as e:
            return self._fallback_in_memory_dense_search(query_vector, jurisdiction, top_k)

    def _fallback_in_memory_dense_search(self, query_vector: List[float], jurisdiction: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback in-memory dense cosine similarity search across registered chunks using numpy."""
        chunks = self.chunks_by_jurisdiction.get(jurisdiction, [])
        if not chunks:
            return []

        emb_matrix = self.in_memory_embeddings.get(jurisdiction)
        if emb_matrix is None or emb_matrix.shape[0] != len(chunks):
            self._ensure_in_memory_embeddings()
            emb_matrix = self.in_memory_embeddings.get(jurisdiction)

        if emb_matrix is not None:
            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm
            sims = np.dot(emb_matrix, q_vec)
            top_indices = np.argsort(-sims)[:top_k]

            results = []
            for idx in top_indices:
                score = float(sims[idx])
                if score <= 0.0:
                    continue
                c = chunks[idx]
                results.append({
                    "chunk_id": c.get("chunk_id", ""),
                    "text": c.get("text", ""),
                    "document_name": c.get("document_name", ""),
                    "statute_title": c.get("statute_title", ""),
                    "section_or_clause": c.get("section_or_clause", "General"),
                    "jurisdiction": c.get("jurisdiction", jurisdiction),
                    "page_numbers": c.get("page_numbers", []),
                    "source_url": c.get("official_source_url", ""),
                    "is_statutory_bar": c.get("is_statutory_bar", False),
                    "dense_score": round(score, 4)
                })
            return results

        return []

    def sparse_search(self, query: str, jurisdiction: str = "national", top_k: int = 10) -> List[Dict[str, Any]]:
        """Searches in-memory BM25 sparse index using genuine token frequencies without artificial boosts."""
        bm25 = self.bm25_indices.get(jurisdiction)
        chunks = self.chunks_by_jurisdiction.get(jurisdiction, [])

        if not bm25 or not chunks:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        raw_scores = bm25.get_scores(tokens)
        max_score = max(raw_scores) if len(raw_scores) > 0 and max(raw_scores) > 0 else 1.0

        indexed_scores = list(enumerate(raw_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            if score <= 0:
                continue
            c = chunks[idx]
            results.append({
                "chunk_id": c.get("chunk_id", ""),
                "text": c.get("text", ""),
                "document_name": c.get("document_name", ""),
                "statute_title": c.get("statute_title", ""),
                "section_or_clause": c.get("section_or_clause", "General"),
                "jurisdiction": c.get("jurisdiction", jurisdiction),
                "page_numbers": c.get("page_numbers", []),
                "source_url": c.get("official_source_url", ""),
                "is_statutory_bar": c.get("is_statutory_bar", False),
                "sparse_score": round(float(score / max_score), 4),
                "raw_bm25_score": round(float(score), 4)
            })
        return results

    def hybrid_search(self, query: str, jurisdiction: str = "national", top_k: int = 8, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Combines Dense and Sparse search results using Reciprocal Rank Fusion (RRF).
        RRF_Score = sum(1 / (rrf_k + rank))
        """
        dense_results = self.dense_search(query, jurisdiction=jurisdiction, top_k=top_k * 2)
        sparse_results = self.sparse_search(query, jurisdiction=jurisdiction, top_k=top_k * 2)

        rrf_scores: Dict[str, float] = Counter()
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(dense_results, start=1):
            doc_id = doc["chunk_id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank)

        for rank, doc in enumerate(sparse_results, start=1):
            doc_id = doc["chunk_id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            # Exact BM25 matches receive high RRF weighting
            rrf_scores[doc_id] += 3.0 / (rrf_k + rank)

        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        # Enforce statutory diversity: do not let secondary studies monopolize the candidate pool!
        final_results = []
        doc_counts: Dict[str, int] = Counter()
        max_per_doc = max(2, top_k // 3)

        for doc_id in sorted_doc_ids:
            doc = doc_map[doc_id]
            d_name = doc.get("document_name", "")
            doc_cap = 2 if "Traditional_Knowledge_Guidelines" in d_name else max_per_doc
            if doc_counts[d_name] < doc_cap:
                doc["hybrid_rrf_score"] = round(rrf_scores[doc_id], 4)
                final_results.append(doc)
                doc_counts[d_name] += 1
                if len(final_results) >= top_k:
                    break

        # Secondary pass: if slots remain, fill with highest remaining
        if len(final_results) < top_k:
            for doc_id in sorted_doc_ids:
                doc = doc_map[doc_id]
                if doc not in final_results:
                    doc["hybrid_rrf_score"] = round(rrf_scores[doc_id], 4)
                    final_results.append(doc)
                    if len(final_results) >= top_k:
                        break

        return final_results
