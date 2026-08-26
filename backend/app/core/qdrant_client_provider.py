import logging
from typing import Optional
from qdrant_client import QdrantClient
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class QdrantClientProvider:
    """
    Factory for managing Qdrant vector database connections.
    Connects to the Qdrant Docker service or falls back to local in-memory mode for unit tests.
    """
    _instance: Optional[QdrantClient] = None

    @classmethod
    def get_client(cls, prefer_in_memory: bool = False) -> QdrantClient:
        if cls._instance is not None and not prefer_in_memory:
            return cls._instance

        if prefer_in_memory:
            logger.info("Initializing in-memory Qdrant client for local testing...")
            return QdrantClient(":memory:")

        try:
            logger.info(f"Connecting to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}...")
            client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=5.0)
            # Test connection
            client.get_collections()
            cls._instance = client
            logger.info("✅ Connected to Qdrant Vector Database container successfully.")
            return client
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant container ({e}). Initializing local in-memory Qdrant client...")
            cls._instance = QdrantClient(":memory:")
            return cls._instance
