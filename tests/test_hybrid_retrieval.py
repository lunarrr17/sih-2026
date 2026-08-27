import pytest
from backend.app.rag.schemas import LegalChunk, StatutoryMetadata
from backend.app.rag.retriever import HybridRetriever
from backend.app.rag.reranker import LegalCrossEncoderReranker

@pytest.fixture(scope="module")
def retriever():
    retriever_instance = HybridRetriever()
    retriever_instance.initialize()
    return retriever_instance

def test_jurisdiction_isolation_national(retriever):
    """Verifies that national queries return only Indian statutory provisions."""
    query = "traditional knowledge patent bar Section 3(p)"
    results = retriever.search(query=query, jurisdiction="national", top_k=5)
    
    assert len(results) > 0
    for r in results:
        assert r["jurisdiction"] == "national"
        assert r["source_url"] != ""

def test_jurisdiction_isolation_international(retriever):
    """Verifies that international queries return only International treaties."""
    query = "WIPO GRATK Treaty mandatory disclosure of origin genetic resources"
    results = retriever.search(query=query, jurisdiction="international", top_k=5)
    
    assert len(results) > 0
    for r in results:
        assert r["jurisdiction"] == "international"
        assert "wipo" in r["document_name"].lower() or "nagoya" in r["document_name"].lower() or "trips" in r["document_name"].lower()

def test_exact_keyword_bm25_retrieval(retriever):
    """Verifies that exact statutory queries like Section 3(p) retrieve the exact section."""
    query = "Section 3(p)"
    results = retriever.search(query=query, jurisdiction="national", top_k=3)
    
    assert len(results) > 0
    # At least one result in the top 3 must explicitly match Section 3(p) or patent act
    found_sec_3p = any("3(p)" in r.get("section_or_clause", "").lower() or "3(p)" in r["text"].lower() or "patents_act" in r["document_name"].lower() for r in results)
    assert found_sec_3p

def test_conceptual_semantic_dense_retrieval(retriever):
    """Verifies that conceptual queries retrieve traditional knowledge rules even without typing Section 3(p)."""
    query = "ancient Ayurvedic medicinal formulas cannot be granted exclusive patent monopolies"
    results = retriever.search(query=query, jurisdiction="national", top_k=5)
    
    assert len(results) > 0
    # Should retrieve Patents Act or IPO TK Guidelines
    found_tk_doc = any("patent" in r["document_name"].lower() or "traditional_knowledge" in r["document_name"].lower() for r in results)
    assert found_tk_doc

def test_cross_encoder_reranker(retriever):
    """Verifies that the cross-encoder reranker scores and sorts candidate chunks."""
    query = "mandatory disclosure of biological source origin in patent application"
    raw_candidates = retriever.search(query=query, jurisdiction="national", top_k=8)
    
    reranker = LegalCrossEncoderReranker()
    reranked = reranker.rerank(query=query, candidates=raw_candidates, top_n=4)
    
    assert len(reranked) <= 4
    assert len(reranked) > 0
    # Every reranked candidate should have a rerank_score
    for item in reranked:
        assert "rerank_score" in item
        assert isinstance(item["rerank_score"], float)
    
    # Assert descending order of scores
    scores = [item["rerank_score"] for item in reranked]
    assert scores == sorted(scores, reverse=True)
