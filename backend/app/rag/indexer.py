import uuid
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from backend.app.core.config import settings
from backend.app.core.qdrant_client_provider import QdrantClientProvider
from backend.app.rag.schemas import LegalChunk
from backend.app.rag.embedder import DenseEmbedder

logger = logging.getLogger(__name__)

class QdrantCorpusIndexer:
    """
    Manages collection creation, embedding conversion, and point upserting into Qdrant.
    Maintains separate collections for National statutes and International treaties.
    """

    def __init__(self, client: Optional[QdrantClient] = None, embedder: Optional[DenseEmbedder] = None):
        self.client = client or QdrantClientProvider.get_client()
        self.embedder = embedder or DenseEmbedder.get_instance()
        self.vector_dim = self.embedder.dimension

    def initialize_collections(self, recreate: bool = False):
        """Creates the national and international collections in Qdrant if they do not exist."""
        collections = [
            settings.QDRANT_COLLECTION_NATIONAL,
            settings.QDRANT_COLLECTION_INTERNATIONAL
        ]

        for col_name in collections:
            existing = [c.name for c in self.client.get_collections().collections]
            if col_name in existing and recreate:
                print(f"Recreating existing collection: {col_name}", flush=True)
                self.client.delete_collection(col_name)
                existing.remove(col_name)

            if col_name not in existing:
                print(f"Creating Qdrant collection '{col_name}' (dim={self.vector_dim}, distance=Cosine)...", flush=True)
                self.client.create_collection(
                    collection_name=col_name,
                    vectors_config=VectorParams(
                        size=self.vector_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"[OK] Collection '{col_name}' created.", flush=True)

    def index_chunks(self, chunks: List[LegalChunk], collection_name: str, batch_size: int = 64) -> int:
        """
        Embeds chunks in batches and upserts them into the specified Qdrant collection.
        Returns the total number of points successfully indexed.
        """
        if not chunks:
            return 0

        self.initialize_collections(recreate=False)
        total_indexed = 0

        print(f"  -> Generating dense vectors for {len(chunks)} chunks into '{collection_name}' (batch_size={batch_size})...", flush=True)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.embedder.embed_documents(texts, batch_size=batch_size)

            points = []
            for chunk, vec in zip(batch, embeddings):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
                payload = chunk.to_qdrant_payload()

                points.append(PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload
                ))

            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            total_indexed += len(points)
            if total_indexed % 256 == 0 or total_indexed == len(chunks):
                print(f"     [Progress] Indexed {total_indexed}/{len(chunks)} points...", flush=True)

        print(f"  [OK] Completed indexing {total_indexed} points into '{collection_name}'.", flush=True)
        return total_indexed
