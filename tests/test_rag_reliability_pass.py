import pytest
from typing import List, Dict, Any
from backend.app.rag.schemas import (
    VerificationStatus,
    EvidenceStrength,
    AuthorityLevel,
    LegalStatus,
    GroundedClaim
)
from backend.app.core.guardrails import GuardrailsEngine
from backend.app.rag.chat_agent import chat_agent, ChatAgentRequest
from backend.app.rag.retriever import retriever


# =====================================================================
# SECTION 6: CHARAKA HIGH-RISK REGRESSION (A, B, C, D DIFFERENTIATION)
# =====================================================================

def test_charaka_query_a_classical_formulation():
    """Query A: I have a classical Charaka formulation. Can I patent it?"""
    q = "I have a classical Charaka formulation. Can I patent it?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national", session_id="charaka_a"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    statutes = [c.statute for c in resp.citations]
    # Must retrieve both D&C (classical status) and Patents Act (patentability)
    assert any("Drugs and Cosmetics" in s for s in statutes)
    assert any("Patents Act" in s for s in statutes)
    # Must contain factual scope assessment indicating specific formulation facts cannot be decided by statute alone
    assert "Legal Scope & Material Premise Assessment" in resp.answer
    assert "Classical Formulation Factual Scope" in resp.answer

def test_charaka_query_b_pure_statutory_tk():
    """Query B: Is traditional Ayurvedic knowledge patentable in India?"""
    q = "Is traditional Ayurvedic knowledge patentable in India?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national", session_id="charaka_b"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    # Must directly cite Section 3(p) of Patents Act
    assert any("Patents Act" in c.statute for c in resp.citations)
    assert any("3(p)" in c.section.lower() or 3 in c.page_numbers for c in resp.citations)

def test_charaka_query_c_modified_formulation():
    """Query C: I modified a traditional Ayurvedic formulation. Can the modified formulation be patented?"""
    q = "I modified a traditional Ayurvedic formulation. Can the modified formulation be patented?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national", session_id="charaka_c"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    # Must cite Patents Act and mention Section 3(d) enhanced efficacy requirement in scope assessment
    assert any("Patents Act" in c.statute for c in resp.citations)
    assert "Modified Formulation Factual Scope" in resp.answer
    assert "3(d)" in resp.answer

def test_charaka_query_d_combined_known_ingredients():
    """Query D: I combined known Ayurvedic ingredients into a new mixture. Can I patent it?"""
    q = "I combined known Ayurvedic ingredients into a new mixture. Can I patent it?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national", session_id="charaka_d"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    # Must cite Patents Act and mention Section 3(e) mere admixture bar in scope assessment
    assert any("Patents Act" in c.statute for c in resp.citations)
    assert "Admixture & Combination Factual Scope" in resp.answer
    assert "3(e)" in resp.answer

# =====================================================================
# SECTION 11: UNSEEN EVALUATION SET (PARAPHRASED NOVEL WORDING)
# =====================================================================

def test_unseen_classical_patent_protection():
    """Could I obtain patent protection for knowledge documented in classical Ayurvedic texts?"""
    q = "Could I obtain patent protection for knowledge documented in classical Ayurvedic texts?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national", session_id="unseen_1"))
    assert not resp.abstain
    assert any("Patents Act" in c.statute for c in resp.citations)
    assert any("3(p)" in c.section.lower() or 3 in c.page_numbers for c in resp.citations)

def test_unseen_combining_known_ingredients():
    """Does merely combining known Ayurvedic ingredients create a patentable invention?"""
    q = "Does merely combining known Ayurvedic ingredients create a patentable invention?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national", session_id="unseen_2"))
    assert not resp.abstain
    assert any("Patents Act" in c.statute for c in resp.citations)
    assert any("3(e)" in c.section.lower() or 3 in c.page_numbers for c in resp.citations)

def test_unseen_medicinal_plants_ip_biodiversity():
    """If my formulation uses Indian medicinal plants, what IP and biodiversity checks should I consider?"""
    q = "If my formulation uses Indian medicinal plants, what IP and biodiversity checks should I consider?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national", session_id="unseen_3"))
    assert not resp.abstain
    statutes = [c.statute for c in resp.citations]
    assert any("Biological Diversity" in s for s in statutes)
    assert any("Patents Act" in s for s in statutes)

def test_unseen_wipo_gratk_current_binding_status():
    """Does the WIPO GRATK treaty currently impose a patent disclosure requirement on India?"""
    q = "Does the WIPO GRATK treaty currently impose a patent disclosure requirement on India?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="international", session_id="unseen_4"))
    assert not resp.abstain
    assert any("WIPO" in c.statute for c in resp.citations)
    # Check that citation accurately reflects adopted / pending entry into force status
    wipo_cites = [c for c in resp.citations if "WIPO" in c.statute]
    assert any("adopted" in c.legal_status.lower() or "pending" in c.legal_status.lower() for c in wipo_cites)

def test_unseen_trips_article_27_override_analysis():
    """Does TRIPS Article 27 override India's Section 3(p)?"""
    q = "Does TRIPS Article 27 override India's Section 3(p)?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="comparative", session_id="unseen_5"))
    assert not resp.abstain
    statutes = [c.statute for c in resp.citations]
    assert any("Patents Act" in s for s in statutes)
    assert any("TRIPS" in s for s in statutes)

def test_unseen_rule_158b_not_patent_eligibility():
    """Can a person rely on Rule 158B to establish patent eligibility?"""
    q = "Can a person rely on Rule 158B to establish patent eligibility?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national", session_id="unseen_6"))
    assert not resp.abstain
    # Rule 158B governs drug licensing under D&C, while patent eligibility is under Patents Act 3(p)
    statutes = [c.statute for c in resp.citations]
    assert any("Drugs and Cosmetics" in s for s in statutes)
    assert any("Patents Act" in s for s in statutes)

# =====================================================================
# SECTION 12: ADVERSARIAL CLAIM TESTS
# =====================================================================

def test_adversarial_citation_substitution():
    """Citation substitution: Correct legal claim attached to irrelevant evidence chunk -> REJECT."""
    claim = "Under Section 3(p) of the Patents Act, 1970, traditional knowledge is an excluded category of invention."
    # Assigned chunk is Rule 161 (Labelling) from Drugs & Cosmetics Act
    fake_chunk = {
        "ref_id": "[REF-1]",
        "document_name": "Drugs_and_Cosmetics_Act_Ayurveda.pdf",
        "statute_title": "The Drugs and Cosmetics Act, 1940 & Rules (Ayurveda Provisions)",
        "section_or_clause": "Rule 161",
        "page_numbers": [190],
        "text": "Manner of labelling of Ayurvedic medicines. The container of a medicine for internal use shall be conspicuously labelled with the true list of all ingredients."
    }
    result = GuardrailsEngine.verify_claim_against_evidence(claim, [fake_chunk])
    assert result.status == VerificationStatus.UNSUPPORTED.value
    assert "Citation mismatch" in result.notes or "not present in assigned evidence" in result.notes

def test_adversarial_unsupported_qualifier_always():
    """Unsupported qualifier: Evidence has exemptions, but claim asserts unconditional universality -> PARTIALLY_SUPPORTED."""
    claim = "Under Section 7 of the Biological Diversity Act, prior approval is always required without exception for all citizens."
    chunk = {
        "ref_id": "[REF-1]",
        "document_name": "Biological_Diversity_Amendment_Act_2023.pdf",
        "statute_title": "The Biological Diversity (Amendment) Act, 2023",
        "section_or_clause": "Section 7",
        "page_numbers": [4],
        "text": "Provided that the provisions of this section shall not apply to the codified traditional knowledge, cultivated medicinal plants and to vaids, hakims and registered AYUSH practitioners."
    }
    result = GuardrailsEngine.verify_claim_against_evidence(claim, [chunk])
    assert result.status != VerificationStatus.SUPPORTED.value
    assert "always" in result.unsupported_qualifiers or "without exception" in result.unsupported_qualifiers

def test_adversarial_temporal_inflation_wipo_gratk():
    """Temporal inflation: Adopted treaty asserted as currently binding law in India -> REJECT."""
    claim = "The WIPO GRATK Treaty is currently binding in India and immediately enforceable."
    chunk = {
        "ref_id": "[REF-1]",
        "document_name": "WIPO_GRATK_Treaty_2024.pdf",
        "statute_title": "WIPO Treaty on Intellectual Property, Genetic Resources and Associated Traditional Knowledge (2024)",
        "section_or_clause": "Article 3",
        "legal_status": "adopted (May 2024, pending entry into force under Article 17)",
        "binding_on_jurisdiction": "Not currently binding on India (adopted, pending ratification & entry into force)",
        "page_numbers": [3],
        "text": "Contracting parties shall provide in their national law a mandatory disclosure requirement for patent applications based on genetic resources."
    }
    result = GuardrailsEngine.verify_claim_against_evidence(claim, [chunk])
    assert result.status == VerificationStatus.UNSUPPORTED.value
    assert "currently binding" in result.unsupported_qualifiers or "unsupported legal status" in result.notes.lower()

def test_adversarial_source_authority_inflation():
    """Source-authority inflation: Academic study claimed as a binding primary statute -> REJECT."""
    claim = "Under this binding statute, traditional knowledge must be registered within 30 days."
    chunk = {
        "ref_id": "[REF-1]",
        "document_name": "IPO_Traditional_Knowledge_Guidelines.pdf",
        "statute_title": "Academic Legal Study: Traditional Knowledge Protection & Patent Guidelines (IOSR-JHSS)",
        "authority_level": "secondary_academic_study",
        "legal_status": "academic_publication (non-binding scholarship 2012)",
        "section_or_clause": "General Provision",
        "page_numbers": [35],
        "text": "Academic scholarship discusses traditional knowledge protection regimes and documentation in developing nations."
    }
    result = GuardrailsEngine.verify_claim_against_evidence(claim, [chunk])
    assert result.status == VerificationStatus.UNSUPPORTED.value

# =====================================================================
# SECTION 14: CORPUS VERSION & CACHE INTEGRITY
# =====================================================================

def test_cache_fingerprint_invalidation_coverage():
    """Verify changing corpus version, model, model version, chunk size, overlap, or text invalidates cache."""
    fp1 = retriever.vector_store.compute_cache_fingerprint(
        corpus_version="2024.1",
        embedding_model_name="all-MiniLM-L6-v2",
        embedding_model_version="1.0.0",
        chunk_size=750,
        chunk_overlap=100,
        preprocessing_version="2.1.0"
    )
    # 1. Corpus version change
    fp_corpus = retriever.vector_store.compute_cache_fingerprint(
        corpus_version="2024.2",
        embedding_model_name="all-MiniLM-L6-v2",
        embedding_model_version="1.0.0",
        chunk_size=750,
        chunk_overlap=100,
        preprocessing_version="2.1.0"
    )
    assert fp1 != fp_corpus

    # 2. Embedding model change
    fp_model = retriever.vector_store.compute_cache_fingerprint(
        corpus_version="2024.1",
        embedding_model_name="bge-small-en-v1.5",
        embedding_model_version="1.0.0",
        chunk_size=750,
        chunk_overlap=100,
        preprocessing_version="2.1.0"
    )
    assert fp1 != fp_model

    # 3. Chunk size change
    fp_size = retriever.vector_store.compute_cache_fingerprint(
        corpus_version="2024.1",
        embedding_model_name="all-MiniLM-L6-v2",
        embedding_model_version="1.0.0",
        chunk_size=500,
        chunk_overlap=100,
        preprocessing_version="2.1.0"
    )
    assert fp1 != fp_size

    # 4. Chunk overlap change
    fp_overlap = retriever.vector_store.compute_cache_fingerprint(
        corpus_version="2024.1",
        embedding_model_name="all-MiniLM-L6-v2",
        embedding_model_version="1.0.0",
        chunk_size=750,
        chunk_overlap=150,
        preprocessing_version="2.1.0"
    )
    assert fp1 != fp_overlap

    # 5. Preprocessing version change
    fp_preproc = retriever.vector_store.compute_cache_fingerprint(
        corpus_version="2024.1",
        embedding_model_name="all-MiniLM-L6-v2",
        embedding_model_version="1.0.0",
        chunk_size=750,
        chunk_overlap=100,
        preprocessing_version="2.2.0"
    )
    assert fp1 != fp_preproc

# =====================================================================
# SECTION 17: RETRIEVAL QUALITY METRICS (Recall@k, Precision@k, MRR)
# =====================================================================

def test_retrieval_quality_metrics():
    """Computes Recall@5, Precision@5, and MRR across golden benchmark queries."""
    golden_test_cases = [
        ("Section 3(p) traditional knowledge exclusion", "national", "Patents_Act_1970.PDF"),
        ("Rule 161 labelling requirements Ayurvedic medicines", "national", "Drugs_and_Cosmetics_Act_Ayurveda.pdf"),
        ("Section 6 prior approval National Biodiversity Authority", "national", "Biological_Diversity_Amendment_Act_2023.pdf"),
        ("Regulation 4 disease risk reduction claims Ayurveda Aahar", "national", "FSSAI_Ayurveda_Aahar_Regulations_2022.pdf"),
        ("Article 3 mandatory disclosure traditional knowledge", "international", "WIPO_GRATK_Treaty_2024.pdf"),
        ("Nagoya Protocol access and fair equitable benefit-sharing", "international", "Nagoya_Protocol_ABS.pdf"),
        ("TRIPS Article 27 patentable subject matter flexibilities", "international", "WTO_TRIPS_Agreement.pdf")
    ]

    total_mrr = 0.0
    recalls_at_5 = []
    precisions_at_5 = []

    for query, jur, target_doc in golden_test_cases:
        results = retriever.search(query, jurisdiction=jur, top_k=5)
        docs = [r.get("document_name") for r in results]

        # Recall@5: Did target_doc appear in top 5?
        hit = target_doc in docs
        recalls_at_5.append(1.0 if hit else 0.0)

        # Precision@5: Fraction of retrieved chunks matching target_doc
        p = sum(1 for d in docs if d == target_doc) / len(docs) if docs else 0.0
        precisions_at_5.append(p)

        # MRR: Reciprocal rank of first hit
        if hit:
            rank = docs.index(target_doc) + 1
            total_mrr += (1.0 / rank)

    mean_recall = sum(recalls_at_5) / len(recalls_at_5)
    mean_precision = sum(precisions_at_5) / len(precisions_at_5)
    mrr = total_mrr / len(golden_test_cases)

    print(f"\n[Retrieval Metrics] Recall@5: {mean_recall:.2f}, Precision@5: {mean_precision:.2f}, MRR: {mrr:.2f}")
    assert mean_recall >= 0.85, f"Recall@5 {mean_recall} below target 0.85"
    assert mrr >= 0.70, f"MRR {mrr} below target 0.70"
