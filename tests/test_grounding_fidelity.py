import pytest
from backend.app.rag.schemas import LegalChunk, StatutoryMetadata
from backend.app.rag.retriever import retriever
from backend.app.rag.chat_agent import chat_agent, ChatAgentRequest
from backend.app.rag.generator import GroundedLLMSynthesizer
from backend.app.core.guardrails import GuardrailsEngine

@pytest.fixture(scope="module", autouse=True)
def init_retriever():
    retriever.initialize()

# =====================================================================
# A. POSITIVE RETRIEVAL TESTS
# =====================================================================

def test_positive_retrieval_section_3p():
    """Verifies that Section 3(p) query retrieves the Patents Act chunk with page numbers."""
    query = "What does Section 3(p) of the Patents Act, 1970 say about traditional knowledge?"
    results = retriever.search(query=query, jurisdiction="national", top_k=3)

    assert len(results) > 0, "Should retrieve chunks for Section 3(p)"
    found_3p = any("3(p)" in r.get("section_or_clause", "") or "traditional knowledge" in r.get("text", "").lower() for r in results)
    assert found_3p, "Should retrieve Section 3(p) or traditional knowledge provision"
    for r in results:
        assert r["jurisdiction"] == "national"
        assert len(r.get("page_numbers", [])) > 0, "Page numbers must be preserved"

def test_positive_retrieval_biodiversity_abs():
    """Verifies retrieval of Biological Diversity Act provisions for ABS queries."""
    query = "Biological Diversity Act exemptions for AYUSH practitioners from ABS access benefit sharing"
    results = retriever.search(query=query, jurisdiction="national", top_k=4)

    assert len(results) > 0
    found_bd = any("biological_diversity" in r.get("document_name", "").lower() or "biodiversity" in r.get("statute_title", "").lower() for r in results)
    assert found_bd, "Must retrieve Biological Diversity Act chunks"

def test_positive_retrieval_rule_161_labelling():
    """Verifies retrieval of Rule 161 labelling requirements."""
    query = "Drugs and Cosmetics Rules Rule 161 labelling requirements for Ayurvedic medicines"
    results = retriever.search(query=query, jurisdiction="national", top_k=4)

    assert len(results) > 0
    found_rule = any("161" in r.get("section_or_clause", "") or "161" in r.get("text", "") or "label" in r.get("text", "").lower() for r in results)
    assert found_rule, "Must retrieve Rule 161 labelling chunks"

def test_positive_retrieval_wipo_gratk():
    """Verifies retrieval of WIPO GRATK Treaty in the international collection."""
    query = "WIPO Treaty on Intellectual Property, Genetic Resources and Associated Traditional Knowledge 2024"
    results = retriever.search(query=query, jurisdiction="international", top_k=3)

    assert len(results) > 0
    for r in results:
        assert r["jurisdiction"] == "international"
    found_gratk = any("gratk" in r.get("document_name", "").lower() or "wipo" in r.get("statute_title", "").lower() for r in results)
    assert found_gratk, "Must retrieve WIPO GRATK chunks from international collection"

# =====================================================================
# B. NEGATIVE RETRIEVAL & SAFE ABSTENTION TESTS
# =====================================================================

def test_negative_abstention_cryptocurrency():
    """Verifies system rejects and abstains on cryptocurrency questions."""
    req = ChatAgentRequest(
        query="Can I issue cryptocurrency tokens or smart contracts backed by Ayurvedic IP?",
        jurisdiction="national",
        session_id="test_neg_crypto"
    )
    res = chat_agent.process_message(req)

    assert res.abstain is True, "System must abstain on cryptocurrency query"
    assert res.is_grounded is False
    assert len(res.citations) == 0, "No citations may be attached to an abstention"
    assert "crypto" in res.answer.lower() or "out of scope" in res.answer.lower() or "abstain" in res.answer.lower()

def test_negative_abstention_quantum_computing():
    """Verifies system abstains on quantum computing questions."""
    req = ChatAgentRequest(
        query="How does quantum computing algorithm optimization affect patent claims?",
        jurisdiction="national",
        session_id="test_neg_quantum"
    )
    res = chat_agent.process_message(req)

    assert res.abstain is True, "System must abstain on quantum computing"
    assert res.is_grounded is False
    assert len(res.citations) == 0

def test_negative_abstention_unrelated_tax():
    """Verifies system abstains on unrelated tax law questions."""
    req = ChatAgentRequest(
        query="What is the capital gains tax rate for corporate real estate profits?",
        jurisdiction="national",
        session_id="test_neg_tax"
    )
    res = chat_agent.process_message(req)

    assert res.abstain is True
    assert res.is_grounded is False
    assert len(res.citations) == 0

def test_negative_abstention_criminal_law():
    """Verifies system abstains on unrelated criminal law questions."""
    req = ChatAgentRequest(
        query="What are the criminal penalties for armed robbery and grand theft?",
        jurisdiction="national",
        session_id="test_neg_criminal"
    )
    res = chat_agent.process_message(req)

    assert res.abstain is True
    assert res.is_grounded is False
    assert len(res.citations) == 0

# =====================================================================
# C. HALLUCINATION RESISTANCE TEST
# =====================================================================

def test_hallucination_resistance_fictitious_act():
    """
    Asks about the non-existent 'Ayurveda Intellectual Property Protection Act, 2021'.
    The system MUST NOT accept the premise, must NOT invent provisions, and must abstain.
    """
    req = ChatAgentRequest(
        query="What does the Ayurveda Intellectual Property Protection Act 2021 say about cryptocurrency licensing?",
        jurisdiction="national",
        session_id="test_hallucination_probe"
    )
    res = chat_agent.process_message(req)

    assert res.abstain is True, "Must abstain on fictitious Act inquiry"
    assert res.is_grounded is False
    assert len(res.citations) == 0
    # Must NOT invent the fictitious act name in the answer as if it existed
    assert "Ayurveda Intellectual Property Protection Act, 2021" not in res.answer or "out of scope" in res.answer.lower() or "not contain" in res.answer.lower()

# =====================================================================
# D. CITATION & EVIDENCE-ID FIDELITY TEST
# =====================================================================

def test_evidence_id_citation_fidelity():
    """
    Verifies that citations generated by the synthesizer resolve only to actual
    retrieved chunks and preserve reference IDs.
    """
    query = "What is the traditional knowledge patent bar under Section 3(p)?"
    chunks = retriever.search(query=query, jurisdiction="national", top_k=3)
    assert len(chunks) > 0

    answer, citations = GroundedLLMSynthesizer.synthesize(query=query, chunks=chunks, jurisdiction="national")

    assert len(citations) > 0, "Grounded answer must include citations"
    for c in citations:
        assert c.statute != "", "Statute name must not be empty"
        assert c.section != "", "Section must not be empty"
        assert c.source_url.startswith("http"), "Source URL must be a valid HTTP URL"
        assert c.ref_id is not None, "CitationItem must have a resolved ref_id"
        assert c.ref_id.startswith("[REF-"), "ref_id must match [REF-X] pattern"

# =====================================================================
# E. JURISDICTION ISOLATION TESTS
# =====================================================================

def test_jurisdiction_strict_isolation():
    """Verifies that international search never retrieves national documents."""
    query = "Section 3(p) patent bar"
    intl_results = retriever.search(query=query, jurisdiction="international", top_k=5)
    for r in intl_results:
        assert r["jurisdiction"] == "international"
        assert "patents_act_1970" not in r.get("document_name", "").lower()

def test_comparative_mode_separate_regimes():
    """Verifies comparative synthesis maintains distinct national and international sections."""
    query = "traditional knowledge patent disclosure requirements"
    req = ChatAgentRequest(
        query=query,
        jurisdiction="comparative",
        session_id="test_comp_isolation"
    )
    res = chat_agent.process_message(req)

    assert "National Regime" in res.answer
    assert "International Regime" in res.answer
    assert res.jurisdiction == "comparative"

# =====================================================================
# F. REAL IN-MEMORY VECTOR SIMILARITY TEST (NO FIRST-K FALLBACK)
# =====================================================================

def test_in_memory_dense_search_ranks_by_real_similarity():
    """
    Verifies that in-memory fallback computes real cosine similarity
    and does NOT return sequential first-k chunks with fake scores.
    """
    vs = retriever.vector_store
    q_vec = vs.embedder.embed_text("biological diversity benefit sharing NBA SBB")

    results = vs._fallback_in_memory_dense_search(q_vec, jurisdiction="national", top_k=5)
    assert len(results) > 0

    # Check that scores are distinct real floats, not all identical 0.75
    scores = [r["dense_score"] for r in results]
    assert all(isinstance(s, float) for s in scores)
    assert not all(s == 0.75 for s in scores), "Must not return hardcoded 0.75 score"
    assert scores == sorted(scores, reverse=True), "Results must be sorted descending by similarity"

    # Check that top result is about biodiversity, not the first alphabetical PDF (Patents Act)
    top_doc = results[0]["document_name"].lower()
    assert "biological_diversity" in top_doc or "biodiversity" in results[0]["statute_title"].lower(), "Semantic search must retrieve relevant BD chunks first"

# =====================================================================
# G. EMPTY EVIDENCE ABSTENTION
# =====================================================================

def test_empty_evidence_graceful_abstention():
    """Verifies that passing empty chunks to synthesizer triggers safe abstention."""
    answer, citations = GroundedLLMSynthesizer.synthesize(
        query="Any legal query",
        chunks=[],
        jurisdiction="national"
    )
    assert "Safe Abstention" in answer or "abstains" in answer
    assert len(citations) == 0
