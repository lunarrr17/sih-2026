import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file from project root or current working dir
env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

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
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    VECTOR_DIMENSION: int = 384
    
    # LLM API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    
    # Guardrails & Confidence Threshold
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
    
    # Legal Disclaimer
    LEGAL_DISCLAIMER: str = (
        "Disclaimer: IP-SAKTI Sahayak provides informational guidance grounded in official statutes, "
        "rules, treaties, and pharmacopoeial standards. It does not constitute formal legal advice."
    )

settings = Settings()
