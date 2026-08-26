import os
from pathlib import Path

class Settings:
    PROJECT_NAME: str = "IP-SAKTI Sahayak"
    VERSION: str = "1.0.0 (MVP)"
    DESCRIPTION: str = "Multilingual, Source-Grounded AI Assistant for Intellectual Property and Regulatory Guidance in Ayurveda"
    
    # Filesystem Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CORPUS_DIR: Path = DATA_DIR / "corpus"
    NATIONAL_CORPUS_DIR: Path = CORPUS_DIR / "national"
    INTL_CORPUS_DIR: Path = CORPUS_DIR / "international"
    
    # Qdrant Vector Database Settings
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NATIONAL: str = "ayurveda_national_statutes"
    QDRANT_COLLECTION_INTERNATIONAL: str = "ayurveda_international_treaties"
    
    # Embedding Model Settings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    VECTOR_DIMENSION: int = 384
    
    # Guardrails & Confidence Threshold
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
    
    # Legal Disclaimer
    LEGAL_DISCLAIMER: str = (
        "Disclaimer: IP-SAKTI Sahayak provides informational guidance grounded in official statutes, "
        "rules, treaties, and pharmacopoeial standards. It does not constitute formal legal advice."
    )

settings = Settings()
