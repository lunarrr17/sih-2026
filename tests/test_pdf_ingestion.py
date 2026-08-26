import pytest
from pathlib import Path
from qdrant_client import QdrantClient

from backend.app.core.config import settings
from backend.app.core.qdrant_client_provider import QdrantClientProvider
from backend.app.rag.schemas import LegalChunk, StatutoryMetadata
from backend.app.rag.pdf_loader import PDFStatutoryLoader
from backend.app.rag.indexer import QdrantCorpusIndexer
from backend.app.rag.embedder import DenseEmbedder

def test_statutory_metadata_schema():
    meta = StatutoryMetadata(
        document_name="Patents_Act_1970.PDF",
        statute_title="The Patents Act, 1970",
        section_or_clause="Section 3(p)",
        jurisdiction="national",
        category="classical",
        page_numbers=[3],
        official_source_url="https://ipindia.gov.in",
        is_statutory_bar=True
    )
    assert meta.is_statutory_bar is True
    assert meta.jurisdiction == "national"
    assert meta.section_or_clause == "Section 3(p)"

def test_legal_chunk_payload_conversion():
    meta = StatutoryMetadata(
        document_name="WIPO_GRATK_Treaty_2024.pdf",
        statute_title="WIPO GRATK Treaty 2024",
        section_or_clause="Article 3",
        jurisdiction="international",
        category="classical",
        page_numbers=[2],
        official_source_url="https://www.wipo.int",
        is_statutory_bar=False
    )
    chunk = LegalChunk(
        chunk_id="wipo_gratk_art_3_test",
        text="Where a claimed invention in a patent application is based on genetic resources...",
        metadata=meta
    )
    payload = chunk.to_qdrant_payload()
    assert payload["chunk_id"] == "wipo_gratk_art_3_test"
    assert payload["statute_title"] == "WIPO GRATK Treaty 2024"
    assert payload["jurisdiction"] == "international"
    assert "genetic resources" in payload["text"]

def test_pdf_loader_single_document():
    loader = PDFStatutoryLoader(chunk_size=700, chunk_overlap=100)
    pdf_path = settings.DATA_DIR / "raw_documents" / "international" / "WIPO_GRATK_Treaty_2024.pdf"
    if pdf_path.exists():
        chunks = loader.load_pdf(pdf_path, jurisdiction="international")
        assert len(chunks) > 0
        for c in chunks:
            assert c.metadata.jurisdiction == "international"
            assert len(c.text) > 20
