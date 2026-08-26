import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class DenseEmbedder:
    """
    Manages local dense embedding generation using SentenceTransformers.
    Default model: 'all-MiniLM-L6-v2' / 'BAAI/bge-small-en-v1.5' (384 dimensions).
    """
    _instance: "DenseEmbedder" = None
    _model: SentenceTransformer = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading dense embedding model: {model_name}...")
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"✅ Embedding model loaded. Vector dimension: {self.dimension}")

    @classmethod
    def get_instance(cls) -> "DenseEmbedder":
        if cls._instance is None:
            # Fallback to all-MiniLM-L6-v2 for fast local execution if needed
            model_to_load = "all-MiniLM-L6-v2"
            cls._instance = cls(model_name=model_to_load)
        return cls._instance

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string into a float vector."""
        embedding = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embeds a batch of strings into float vectors."""
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()
