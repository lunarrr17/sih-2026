from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class StatutoryMetadata(BaseModel):
    """
    Structured metadata associated with an ingested statutory chunk.
    """
    document_name: str
    statute_title: str
    section_or_clause: str = "General Provision"
    jurisdiction: str = Field(description="'national' or 'international'")
    category: str = "classical"
    page_numbers: List[int] = []
    official_source_url: str = "https://ipindia.gov.in"
    is_statutory_bar: bool = False

class LegalChunk(BaseModel):
    """
    Unit of legal text ingested from official PDF documents,
    containing text content, identifier, and structured metadata.
    """
    chunk_id: str
    text: str
    metadata: StatutoryMetadata
    embedding: Optional[List[float]] = None

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Converts the chunk to a Qdrant point payload dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document_name": self.metadata.document_name,
            "statute_title": self.metadata.statute_title,
            "section_or_clause": self.metadata.section_or_clause,
            "jurisdiction": self.metadata.jurisdiction,
            "category": self.metadata.category,
            "page_numbers": self.metadata.page_numbers,
            "official_source_url": self.metadata.official_source_url,
            "is_statutory_bar": self.metadata.is_statutory_bar,
        }
