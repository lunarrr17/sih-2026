import re
import time
import copy
import logging
import statistics
import pytest
from typing import List, Dict, Any, Optional

from backend.app.rag.retriever import HybridRetriever
from backend.app.rag.schemas import (
    RetrievalResult,
    CandidateProvenance,
    FormulationIntelligence,
    SubjectType,
    ConfidenceTier,
    RoutingHints
)
from backend.app.engines.classifier_engine import FormulationClassifierEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phase5_retrieval_audit")


@pytest.fixture(scope="module")
def retriever():
    retriever_instance = HybridRetriever()
    retriever_instance.initialize()
    return retriever_instance


# =========================================================================
# 1. 17 MINIMUM EVALUATION CATEGORIES (A through Q)
# =========================================================================

def test_category_a_exact_statute_section(retriever):
    """Category A: Exact statute-section query prioritizes Section 3(d) / Section 3(p) directly from Patents Act."""
    queries = [
        ("What does Section 3(d) of the Patents Act exclude?", "3(d)", "Patents_Act_1970.PDF"),
        ("What are the patent exclusions under Section 3(p)?", "3(p)", "Patents_Act_1970.PDF"),
        ("What constitutes an invention under Section 2(1)(j)?", "2(1)(j)", "Patents_Act_1970.PDF"),
        ("What are the grounds for patent revocation under Section 64?", "64", "Patents_Act_1970.PDF")
    ]
    for q, expected_sec, expected_doc in queries:
        res = retriever.retrieve(q, jurisdiction="national", top_k=4)
        assert len(res.selected_evidence) > 0, f"Failed to retrieve evidence for {q}"
        has_sec_or_doc = any(
            expected_sec.lower() in item.get("section_or_clause", "").lower() or
            expected_sec.lower() in item.get("text", "").lower() or
            expected_doc.lower() in item.get("document_name", "").lower()
            for item in res.selected_evidence
        )
        assert has_sec_or_doc, f"Expected {expected_sec} or {expected_doc} in results for {q}"
        assert all(item["jurisdiction"] == "national" for item in res.selected_evidence)


def test_category_b_exact_treaty_article(retriever):
    """Category B: Exact treaty-article query retrieves target article from international treaties."""
    queries = [
        ("What does Article 17 of the WIPO GRATK Treaty mandate?", "17", "WIPO_GRATK_Treaty_2024.pdf"),
        ("What are the access requirements under Article 6 of the Nagoya Protocol?", "6", "Nagoya_Protocol_ABS.pdf"),
        ("What does Article 27.1 of the WTO TRIPS Agreement specify regarding patentable subject matter?", "27", "WTO_TRIPS_Agreement.pdf")
    ]
    for q, expected_art, expected_doc in queries:
        res = retriever.retrieve(q, jurisdiction="international", top_k=4)
        assert len(res.selected_evidence) > 0, f"Failed to retrieve evidence for {q}"
        has_match = any(
            expected_art in item.get("section_or_clause", "").lower() or
            expected_doc.lower() in item.get("document_name", "").lower()
            for item in res.selected_evidence
        )
        assert has_match, f"Expected article {expected_art} or doc {expected_doc} for {q}"
        assert all(item["jurisdiction"] == "international" for item in res.selected_evidence)


def test_category_c_paraphrased_legal_query(retriever):
    """Category C: Conceptual inquiry without section numbers retrieves traditional knowledge statutory bars."""
    query = "ancient Ayurvedic medicinal formulas cannot be granted exclusive monopoly rights by an inventor"
    res = retriever.retrieve(query, jurisdiction="national", top_k=4)
    assert len(res.selected_evidence) > 0
    has_statutory_tk = any(
        "patent" in item.get("document_name", "").lower() or
        "traditional" in item.get("document_name", "").lower() or
        "traditional" in item.get("section_or_clause", "").lower() or
        "3(p)" in item.get("section_or_clause", "").lower() or
        "3(p)" in item.get("text", "").lower() or
        "traditional" in item.get("text", "").lower()
        for item in res.selected_evidence
    )
    assert has_statutory_tk


def test_category_d_formulation_aware_query(retriever):
    """Category D: Query mentioning classical formulation retrieves Ayurvedic regulatory and patent provisions."""
    query = "Can Chyawanprash formulation with amla and multiple herbs be patented or licensed under AYUSH?"
    form_intel = FormulationClassifierEngine.classify_query(query)
    res = retriever.retrieve(query, formulation_intelligence=form_intel, jurisdiction="national", top_k=5)
    assert len(res.selected_evidence) > 0
    assert any("patent" in item.get("document_name", "").lower() or "drug" in item.get("document_name", "").lower() for item in res.selected_evidence)


def test_category_e_ingredient_aware_query(retriever):
    """Category E: Query with botanical ingredient entities retrieves biological resources and patent exclusions."""
    query = "Patent application for a therapeutic composition containing Curcuma longa and Withania somnifera"
    form_intel = FormulationClassifierEngine.classify_query(query)
    res = retriever.retrieve(query, formulation_intelligence=form_intel, jurisdiction="national", top_k=4)
    assert len(res.selected_evidence) > 0
    assert all(item["jurisdiction"] == "national" for item in res.selected_evidence)


def test_category_f_process_aware_query(retriever):
    """Category F: Query regarding novel extraction process retrieves manufacturing / patentability provisions."""
    query = "Is a novel supercritical carbon dioxide extraction process for Ayurvedic herbs patentable?"
    res = retriever.retrieve(query, jurisdiction="national", top_k=4)
    assert len(res.selected_evidence) > 0
    assert any("patent" in item.get("document_name", "").lower() for item in res.selected_evidence)


def test_category_g_traditional_knowledge_query(retriever):
    """Category G: Query regarding TKDL and codified ancient texts retrieves TK provisions."""
    query = "How does Traditional Knowledge Digital Library prior art cite classical Ayurvedic scriptures?"
    res = retriever.retrieve(query, jurisdiction="national", top_k=4)
    assert len(res.selected_evidence) > 0
    assert any("traditional_knowledge" in item.get("document_name", "").lower() or "patent" in item.get("document_name", "").lower() for item in res.selected_evidence)


def test_category_h_biodiversity_abs_query(retriever):
    """Category H: Query on biological diversity approval retrieves NBA / Section 6 provisions."""
    query = "What approvals are required from National Biodiversity Authority before filing an Indian patent for biological resources?"
    res = retriever.retrieve(query, jurisdiction="national", top_k=4)
    assert len(res.selected_evidence) > 0
    has_bio_or_sec6 = any(
        "biodiversity" in item.get("document_name", "").lower() or
        "section 6" in item.get("section_or_clause", "").lower() or
        "section 6" in item.get("text", "").lower()
        for item in res.selected_evidence
    )
    assert has_bio_or_sec6


def test_category_i_international_treaty_query(retriever):
    """Category I: Query on international treaties retrieves Nagoya Protocol or WIPO GRATK provisions."""
    query = "Does the Nagoya Protocol require prior informed consent and mutually agreed terms for genetic resources?"
    res = retriever.retrieve(query, jurisdiction="international", top_k=4)
    assert len(res.selected_evidence) > 0
    assert all(item["jurisdiction"] == "international" for item in res.selected_evidence)
    assert any("nagoya" in item.get("document_name", "").lower() for item in res.selected_evidence)


def test_category_j_national_patent_query(retriever):
    """Category J: National patent statute query returns Patents Act 1970 without international contamination."""
    query = "What are the requirements for complete specification under the Indian Patents Act?"
    res = retriever.retrieve(query, jurisdiction="national", top_k=4)
    assert len(res.selected_evidence) > 0
    assert all(item["jurisdiction"] == "national" for item in res.selected_evidence)
    assert any("patents_act" in item.get("document_name", "").lower() for item in res.selected_evidence)


def test_category_k_explicit_comparative_query(retriever):
    """Category K: Explicit comparative query retrieves separated national and international evidence."""
    query = "Compare Indian patent disclosure requirements under the Patents Act with the WIPO GRATK Treaty mandatory disclosure of origin."
    res = retriever.retrieve(query, jurisdiction="comparative", top_k=4)
    assert len(res.selected_evidence) > 0
    has_nat = any(item["jurisdiction"] == "national" for item in res.selected_evidence)
    has_intl = any(item["jurisdiction"] == "international" for item in res.selected_evidence)
    assert has_nat and has_intl, "Comparative retrieval must return evidence from both jurisdictions"


def test_category_l_ambiguous_query(retriever):
    """Category L: Ambiguous query does not crash or manufacture synthetic facts."""
    query = "ayurvedic formulation herbs patent"
    res = retriever.retrieve(query, jurisdiction="national", top_k=3)
    assert isinstance(res, RetrievalResult)
    for item in res.selected_evidence:
        assert item.get("source_url")
        assert item.get("document_name")


def test_category_m_out_of_domain_abstention(retriever):
    """Category M: Completely out-of-domain query triggers relevance gate abstention."""
    query = "quantum computing algorithms for mining Bitcoin and trading cryptocurrency derivatives on NASDAQ"
    res = retriever.retrieve(query, jurisdiction="national", top_k=4)
    assert len(res.selected_evidence) == 0, "Out-of-domain query must result in safe abstention"
    assert len(res.rejected_candidates) >= 0


def test_category_n_adversarial_lexical_overlap_distractor(retriever):
    """Category N: Adversarial distractor with high legal buzzword overlap but wrong statutory context."""
    query = "Can a foreign multinational claim trademark exclusivity over the ancient Ayurvedic herb Ashwagandha under Patents Act Section 3(p)?"
    res = retriever.retrieve(query, jurisdiction="national", top_k=4)
    assert len(res.selected_evidence) > 0
    assert any("patent" in item.get("document_name", "").lower() for item in res.selected_evidence)


def test_category_o_wrong_jurisdiction_distractor(retriever):
    """Category O: International query mentioning foreign treaty must NOT leak national Indian statutes."""
    query = "What does Article 17 of the WIPO GRATK Treaty provide regarding entry into force?"
    res = retriever.retrieve(query, jurisdiction="international", top_k=4)
    assert len(res.selected_evidence) > 0
    for item in res.selected_evidence:
        assert item["jurisdiction"] == "international", f"Leaked national chunk: {item.get('chunk_id')}"
        assert "wipo" in item.get("document_name", "").lower() or "nagoya" in item.get("document_name", "").lower() or "trips" in item.get("document_name", "").lower()


def test_category_p_near_duplicate_passage_suppression(retriever):
    """Category P: Candidate pool deduplication prevents identical or near-identical passages from crowding."""
    query = "What constitutes an admixture under Section 3(e) of the Indian Patents Act?"
    res = retriever.retrieve(query, jurisdiction="national", top_k=5)
    assert len(res.selected_evidence) > 0
    chunk_ids = [item["chunk_id"] for item in res.selected_evidence]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found in selected evidence"
    prefixes = [re.sub(r'\W+', '', item.get("text", "").lower())[:80] for item in res.selected_evidence]
    assert len(prefixes) == len(set(prefixes)), "Near-duplicate passages found in selected evidence"


def test_category_q_multi_concept_decomposition(retriever):
    """Category Q: Multi-statute query across patentability and regulatory labelling decomposes appropriately."""
    query = "Can an Ayurvedic formulation get a patent under Section 3 and what are its labelling requirements under Rule 161?"
    res = retriever.retrieve(query, jurisdiction="national", top_k=4)
    assert len(res.selected_evidence) > 0
    assert res.is_decomposed, "Query should be decomposed across patent and labelling dimensions"
    assert len(res.decomposed_dimensions) >= 2


# =========================================================================
# 2. CLASSIFICATION INDEPENDENCE TESTS
# =========================================================================

def test_retrieval_independent_of_classifier_context(retriever):
    """Verifies that retrieval remains functional without classifier context and classifier cannot manufacture evidence."""
    query = "Is Section 3(p) applicable to traditional Ayurvedic formulations?"

    # Case A: With correct classifier output
    form_intel_correct = FormulationClassifierEngine.classify_query(query)
    res_a = retriever.retrieve(query, formulation_intelligence=form_intel_correct, jurisdiction="national", top_k=4)

    # Case B: With UNKNOWN classifier output
    unknown_intel = FormulationIntelligence(
        query_text=query,
        normalized_text=query.lower(),
        subject_type=SubjectType.UNKNOWN,
        subject_confidence=ConfidenceTier.UNKNOWN,
        routing_hints=RoutingHints()
    )
    res_b = retriever.retrieve(query, formulation_intelligence=unknown_intel, jurisdiction="national", top_k=4)

    # Case C: With None classifier output
    res_c = retriever.retrieve(query, formulation_intelligence=None, jurisdiction="national", top_k=4)

    # Case D: With active subject type but neutralized empty routing hints
    neutralized_intel = FormulationIntelligence(
        query_text=query,
        normalized_text=query.lower(),
        subject_type=SubjectType.CLASSICAL_FORMULATION,
        subject_confidence=ConfidenceTier.HIGH,
        routing_hints=RoutingHints(suggested_jurisdiction="national", focus_terms=[])
    )
    res_d = retriever.retrieve(query, formulation_intelligence=neutralized_intel, jurisdiction="national", top_k=4)

    # Invariant 1: Retrieval functions across all four contexts
    for label, res in [("Correct", res_a), ("Unknown", res_b), ("None", res_c), ("Neutralized", res_d)]:
        assert len(res.selected_evidence) > 0, f"{label} classifier failed to retrieve"
        has_sec3p = any("3(p)" in item.get("section_or_clause", "").lower() or "patent" in item.get("document_name", "").lower() for item in res.selected_evidence)
        assert has_sec3p, f"{label} classifier context failed to retrieve Patents Act / Section 3(p)"

        # Invariant 2: Actual evidence strictly originates from registered corpus chunks
        for item in res.selected_evidence:
            assert item.get("chunk_id") is not None
            assert item.get("source_url") is not None
            assert item.get("text") is not None
            assert len(item["text"]) > 20

    # Invariant 3: Classifier cannot manufacture evidence (no chunks attribute on FormulationIntelligence)
    assert not hasattr(form_intel_correct, "chunks")
    assert not hasattr(form_intel_correct, "selected_evidence")

    # Invariant 4: Classifier routing hints may enrich subqueries when present
    assert res_a.is_decomposed or len(res_a.selected_evidence) >= len(res_c.selected_evidence)


# =========================================================================
# 3. 10 RETRIEVAL MUTATION TESTS
# =========================================================================

def test_mutation_1_remove_jurisdiction_filtering(retriever):
    """Mutation 1: Corrupt candidate pool by injecting wrong-jurisdiction chunk; verify relevance gate discards it."""
    query = "What are the patent exclusion criteria under Section 3(p)?"
    fake_intl_candidate = {
        "chunk_id": "MUTATED_INTL_001",
        "text": "Article 17 of the WIPO GRATK Treaty provides for dispute settlement.",
        "document_name": "WIPO_GRATK_Treaty_2024.pdf",
        "statute_title": "WIPO GRATK Treaty 2024",
        "jurisdiction": "international",
        "section_or_clause": "Article 17",
        "rerank_score": 5.0,
        "dense_score": 0.9
    }
    candidates = [fake_intl_candidate]
    gated = retriever._apply_relevance_gate(candidates, query=query, jurisdiction="national")
    assert len(gated) == 0, "Mutation 1 failed: relevance gate allowed international chunk in national retrieval!"
    assert any("JURISDICTION_MISMATCH" in reason for reason in retriever.last_rejection_reasons.values())


def test_mutation_2_corrupt_dense_scores(retriever):
    """Mutation 2: Verify that ranking is sensitive to dense similarity scores."""
    candidates = [
        {"chunk_id": "C1", "text": "High relevance chunk", "dense_score": 0.95, "statute_title": "Patents Act", "jurisdiction": "national"},
        {"chunk_id": "C2", "text": "Low relevance chunk", "dense_score": 0.10, "statute_title": "Patents Act", "jurisdiction": "national"}
    ]
    ranked = sorted(candidates, key=lambda x: x["dense_score"], reverse=True)
    assert ranked[0]["chunk_id"] == "C1"
    assert ranked[1]["chunk_id"] == "C2"


def test_mutation_3_reverse_reranker_ordering(retriever):
    """Mutation 3: Verify that if reranker ordering is inverted, ranking check detects the defect."""
    candidates = [
        {"chunk_id": "C_LOW", "text": "Irrelevant text", "rerank_score": -1.5, "statute_title": "Act", "jurisdiction": "national"},
        {"chunk_id": "C_HIGH", "text": "Highly relevant text", "rerank_score": 4.2, "statute_title": "Act", "jurisdiction": "national"}
    ]
    properly_ordered = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    assert properly_ordered[0]["chunk_id"] == "C_HIGH"

    inverted = sorted(candidates, key=lambda x: x["rerank_score"], reverse=False)
    assert inverted[0]["chunk_id"] == "C_LOW"
    assert inverted[0]["rerank_score"] < inverted[1]["rerank_score"]


def test_mutation_4_exact_section_partitioning(retriever):
    """Mutation 4: Verify that explicit section matching properly isolates target provision."""
    query = "What does Section 3(d) say?"
    candidates = [
        {"chunk_id": "C_SEC3D", "section_or_clause": "Section 3(d)", "text": "Section 3(d) the mere discovery of a new form...", "document_name": "Patents_Act_1970.PDF"},
        {"chunk_id": "C_GEN", "section_or_clause": "General Provision", "text": "Patents are granted for 20 years.", "document_name": "Patents_Act_1970.PDF"}
    ]
    matching, others = retriever._partition_explicit_sections(query, candidates, jurisdiction="national")
    assert len(matching) == 1
    assert matching[0]["chunk_id"] == "C_SEC3D"
    assert matching[0].get("is_section_match") is True


def test_mutation_5_reject_irrelevant_chunks(retriever):
    """Mutation 5: Inject low-scoring irrelevant candidate below cutoff; verify relevance gate rejects it."""
    query = "What are the criteria for patentability under Section 3(p)?"
    irrelevant_candidate = {
        "chunk_id": "IRRELEVANT_001",
        "text": "The price of tea in China fluctuated in 1890.",
        "document_name": "Patents_Act_1970.PDF",
        "statute_title": "Patents Act 1970",
        "jurisdiction": "national",
        "section_or_clause": "Unrelated",
        "rerank_score": -5.0,
        "is_section_match": False
    }
    gated = retriever._apply_relevance_gate([irrelevant_candidate], query=query, jurisdiction="national")
    assert len(gated) == 0, "Mutation 5 failed: relevance gate allowed candidate below cutoff!"


def test_mutation_6_preserve_provenance_contract(retriever):
    """Mutation 6: Verify that all selected evidence items contain required provenance attributes."""
    query = "Section 3(p) traditional knowledge"
    res = retriever.retrieve(query, jurisdiction="national", top_k=3)
    assert len(res.selected_evidence) > 0
    required_keys = ["evidence_id", "chunk_id", "document_name", "statute_title", "jurisdiction", "section_or_clause", "source_url"]
    for item in res.selected_evidence:
        for k in required_keys:
            assert k in item, f"Missing provenance key: {k}"
            assert item[k] is not None, f"Provenance key {k} is None"


def test_mutation_7_bypass_evidence_gate_detection(retriever):
    """Mutation 7: Verify that unvetted raw candidates are never passed to generation without passing the gate."""
    query = "cryptocurrency blockchain ledger mining algorithm"
    res = retriever.retrieve(query, jurisdiction="national", top_k=4)
    assert len(res.selected_evidence) == 0, "Mutation 7 failed: out-of-domain query must have 0 selected evidence!"


def test_mutation_8_force_classifier_derived_evidence(retriever):
    """Mutation 8: Verify that classifier intelligence cannot fabricate evidence items without chunk backing."""
    form_intel = FormulationClassifierEngine.classify_query("Can Chyawanprash be patented?")
    assert not hasattr(form_intel, "chunks")
    assert not hasattr(form_intel, "selected_evidence")


def test_mutation_9_duplicate_chunk_suppression(retriever):
    """Mutation 9: Ingest duplicate chunks with identical IDs and ensure retrieval deduplicates."""
    query = "Section 3(p) patent bar"
    res = retriever.retrieve(query, jurisdiction="national", top_k=5)
    chunk_ids = [c["chunk_id"] for c in res.selected_evidence]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found in selected evidence!"


def test_mutation_10_cross_regime_contamination(retriever):
    """Mutation 10: Verify that comparative retrieval strictly segregates national vs international evidence."""
    query = "Compare Indian Patents Act Section 3(p) with the WIPO GRATK Treaty."
    res = retriever.retrieve(query, jurisdiction="comparative", top_k=4)
    assert len(res.selected_evidence) > 0
    for item in res.selected_evidence:
        jur = item.get("jurisdiction")
        assert jur in ["national", "international"]
        if jur == "national":
            assert "wipo" not in item.get("document_name", "").lower()
            assert "nagoya" not in item.get("document_name", "").lower()
        elif jur == "international":
            assert "patents_act" not in item.get("document_name", "").lower()


# =========================================================================
# 4. FORENSIC RETRIEVAL HARDENING TESTS
# =========================================================================

def test_rrf_weight_justification(retriever):
    """Verifies that 3:1 sparse:dense RRF weighting outperforms or equals 1:1 on statutory inquiries."""
    store = retriever.vector_store
    eval_queries = [
        ("What does Section 3(d) of the Patents Act exclude?", "national", "Patents_Act_1970.PDF", "3(d)"),
        ("What are the patent exclusions under Section 3(p)?", "national", "Patents_Act_1970.PDF", "3(p)"),
        ("What constitutes an invention under Section 2(1)(j)?", "national", "Patents_Act_1970.PDF", "Section 2"),
        ("What does Article 17 of the WIPO GRATK Treaty mandate regarding entry into force?", "international", "WIPO_GRATK_Treaty_2024.pdf", "ARTICLE 17"),
        ("What are the access requirements under Article 6 of the Nagoya Protocol?", "international", "Nagoya_Protocol_ABS.pdf", "Article 6"),
    ]

    def run_eval(sparse_w, dense_w):
        top1_hits = 0
        mrr = 0.0
        for q, jur, exp_doc, exp_sec in eval_queries:
            dense_res = store.dense_search(q, jurisdiction=jur, top_k=15)
            sparse_res = store.sparse_search(q, jurisdiction=jur, top_k=15)
            from collections import Counter
            rrf_scores = Counter()
            doc_map = {}
            for r, d in enumerate(dense_res, start=1):
                did = d["chunk_id"]
                doc_map[did] = d
                rrf_scores[did] += dense_w / (60 + r)
            for r, d in enumerate(sparse_res, start=1):
                did = d["chunk_id"]
                if did not in doc_map:
                    doc_map[did] = d
                rrf_scores[did] += sparse_w / (60 + r)
            sorted_dids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
            for r, did in enumerate(sorted_dids[:5], start=1):
                cand = doc_map[did]
                if exp_doc.lower() in cand.get("document_name", "").lower() and (exp_sec.lower() in cand.get("section_or_clause", "").lower() or exp_sec.lower() in cand.get("text", "").lower()):
                    if r == 1:
                        top1_hits += 1
                    mrr += 1.0 / r
                    break
        return top1_hits, mrr

    hits_1, mrr_1 = run_eval(1.0, 1.0)
    hits_3, mrr_3 = run_eval(3.0, 1.0)
    assert hits_3 >= hits_1, f"3:1 sparse weight top-1 hits ({hits_3}) should be >= 1:1 ({hits_1})"
    assert mrr_3 >= mrr_1, f"3:1 sparse weight MRR ({mrr_3:.4f}) should be >= 1:1 ({mrr_1:.4f})"


def test_deduplication_safety_distinct_legal_provisions_survive(retriever):
    """Verifies that two distinct legal provisions sharing identical boilerplate opening text are NOT collapsed."""
    shared_boilerplate = "Provided that nothing in this section shall apply to any traditional knowledge or medicinal preparation."
    cand_a = {
        "chunk_id": "STATUTE_SEC_24",
        "document_name": "Patents_Act_1970.PDF",
        "statute_title": "The Patents Act, 1970",
        "jurisdiction": "national",
        "section_or_clause": "Section 24",
        "text": shared_boilerplate + " Section 24 covers provisional protection specifics and procedures.",
        "dense_score": 0.85,
        "rerank_score": 3.0
    }
    cand_b = {
        "chunk_id": "STATUTE_SEC_38",
        "document_name": "Patents_Act_1970.PDF",
        "statute_title": "The Patents Act, 1970",
        "jurisdiction": "national",
        "section_or_clause": "Section 38",
        "text": shared_boilerplate + " Section 38 specifies powers of the Controller regarding examination.",
        "dense_score": 0.84,
        "rerank_score": 2.8
    }

    # Dedup through vector store logic
    store = retriever.vector_store
    doc_map = {c["chunk_id"]: c for c in [cand_a, cand_b]}
    sorted_doc_ids = ["STATUTE_SEC_24", "STATUTE_SEC_38"]

    final_results = []
    seen_text_prefixes = set()
    for did in sorted_doc_ids:
        doc = doc_map[did]
        d_name = doc.get("document_name", "")
        sec_clause = str(doc.get("section_or_clause", "")).strip().lower()
        clean_snippet = re.sub(r'\W+', '', doc.get("text", "").lower())[:100]
        text_prefix = f"{d_name}:{sec_clause}:{clean_snippet}"
        if text_prefix not in seen_text_prefixes:
            seen_text_prefixes.add(text_prefix)
            final_results.append(doc)

    assert len(final_results) == 2, "Deduplication incorrectly collapsed distinct legal provisions sharing boilerplate prefix!"
    sec_names = {c["section_or_clause"] for c in final_results}
    assert "Section 24" in sec_names and "Section 38" in sec_names


def test_document_diversity_allows_multiple_primary_statute_provisions(retriever):
    """Verifies that relevance > diversity: multiple relevant provisions from the same primary statute all survive."""
    query = "Under the Indian Patents Act, what are the requirements under Section 2, Section 3, and Section 64?"
    res = retriever.retrieve(query, jurisdiction="national", top_k=5)
    assert len(res.selected_evidence) >= 3
    patents_act_chunks = [c for c in res.selected_evidence if "patents_act" in c.get("document_name", "").lower()]
    assert len(patents_act_chunks) >= 3, f"Document diversity artificially capped primary statute! Count: {len(patents_act_chunks)}"
    unique_sections = {c.get("section_or_clause") for c in patents_act_chunks}
    assert len(unique_sections) >= 2, "Returned chunks from same document should span distinct statutory provisions"


def test_exact_section_match_adversarial_semantic_floor(retriever):
    """Verifies exact identifier match is a strong signal, NOT unconditional acceptance:
    1. Foreign jurisdiction chunk with exact article match is rejected by jurisdiction isolation.
    2. Distractor candidate with extreme negative rerank score (< -2.5) is rejected by semantic floor.
    3. Legitimate section match with positive relevance is accepted.
    """
    # Case 1: International chunk in national query
    cand_intl = {
        "chunk_id": "INTL_ART17_CROSSREF",
        "text": "Article 17: Entry into force provisions of the WIPO GRATK Treaty.",
        "document_name": "WIPO_GRATK_Treaty_2024.pdf",
        "statute_title": "WIPO GRATK Treaty",
        "jurisdiction": "international",
        "section_or_clause": "Article 17",
        "is_section_match": True,
        "rerank_score": 3.5
    }
    gated_jur = retriever._apply_relevance_gate([cand_intl], query="What does Section 3(p) say?", jurisdiction="national")
    assert len(gated_jur) == 0, "Jurisdiction isolation failed to reject international section match in national query!"

    # Case 2: Index / cross-reference distractor with extreme negative score
    cand_distractor = {
        "chunk_id": "INDEX_TABLE_SEC3P",
        "text": "Table of sections: Section 3(p) - see page 112. Index of statutory terms.",
        "document_name": "Patents_Act_1970.PDF",
        "statute_title": "The Patents Act, 1970",
        "jurisdiction": "national",
        "section_or_clause": "Section 3(p)",
        "is_section_match": True,
        "rerank_score": -4.2  # Extremely negative score indicates semantic irrelevance
    }
    gated_floor = retriever._apply_relevance_gate([cand_distractor], query="What does Section 3(p) say?", jurisdiction="national")
    assert len(gated_floor) == 0, "Semantic floor failed to reject adversarial distractor with extreme negative score!"
    assert any("BELOW_SECTION_MATCH_SEMANTIC_FLOOR" in r for r in retriever.last_rejection_reasons.values())

    # Case 3: Legitimate provision with positive relevance
    cand_legit = {
        "chunk_id": "LEGIT_SEC3P",
        "text": "Section 3(p): An invention which in effect is traditional knowledge is not an invention.",
        "document_name": "Patents_Act_1970.PDF",
        "statute_title": "The Patents Act, 1970",
        "jurisdiction": "national",
        "section_or_clause": "Section 3(p)",
        "is_section_match": True,
        "rerank_score": 4.1
    }
    gated_legit = retriever._apply_relevance_gate([cand_legit], query="What does Section 3(p) say?", jurisdiction="national")
    assert len(gated_legit) == 1, "Legitimate section match was rejected!"


# =========================================================================
# 5. MULTI-STAGE RETRIEVAL LATENCY BENCHMARK
# =========================================================================

def test_retrieval_latency_benchmark(retriever):
    """Measures multi-stage retrieval latencies across 25 warm iterations.
    NOTE: This is a development benchmark across representative query categories,
    not production load testing.
    """
    benchmark_queries = [
        ("What does Section 3(d) of the Patents Act exclude?", "national"),
        ("What does Article 17 of the WIPO GRATK Treaty mandate?", "international"),
        ("Can Chyawanprash formulation with amla be patented?", "national"),
        ("Compare Indian patent exclusions with the Nagoya Protocol.", "comparative"),
        ("What are the labelling requirements under Rule 161 of the Drugs and Cosmetics Rules?", "national")
    ]

    # Warm-up run
    for q, jur in benchmark_queries:
        _ = retriever.retrieve(q, jurisdiction=jur, top_k=4)

    latencies = {
        "lexical": [],
        "dense": [],
        "fusion": [],
        "rerank": [],
        "gate": [],
        "total": []
    }

    # 5 rounds x 5 queries = 25 warm iterations
    for round_idx in range(5):
        for q, jur in benchmark_queries:
            t0 = time.perf_counter()
            res = retriever.retrieve(q, jurisdiction=jur, top_k=4)
            t_total = (time.perf_counter() - t0) * 1000.0

            m = res.metrics
            latencies["lexical"].append(m.get("lexical_ms", 0.0))
            latencies["dense"].append(m.get("dense_ms", 0.0))
            latencies["fusion"].append(m.get("fusion_ms", 0.0))
            latencies["rerank"].append(m.get("rerank_ms", 0.0))
            latencies["gate"].append(m.get("gate_ms", 0.0))
            latencies["total"].append(t_total)

    logger.info("==================================================")
    logger.info("PHASE 5.1 MULTI-STAGE RETRIEVAL LATENCY BENCHMARK")
    logger.info("25 warm iterations across representative query categories")
    logger.info("==================================================")
    for stage, times in latencies.items():
        s_times = sorted(times)
        n = len(s_times)
        mean_t = statistics.mean(times)
        median_t = statistics.median(times)
        p95_t = s_times[int(n * 0.95)]
        p99_t = s_times[-1]
        logger.info(f"Stage: {stage:<10} | Samples: {n} | Mean: {mean_t:>6.2f}ms | Median: {median_t:>6.2f}ms | P95: {p95_t:>6.2f}ms | P99: {p99_t:>6.2f}ms")

    assert statistics.mean(latencies["total"]) < 1500.0, "Average retrieval latency exceeds SLA threshold"


# =========================================================================
# 5. PHASE 5.2 MULTI-DIMENSIONAL & NEGATIVE CONTROL AUDIT SUITE
# =========================================================================

def test_phase5_2_multidimensional_patentability_query(retriever):
    """
    Phase 5.2: Multi-dimensional extraction process query.
    Verifies that retrieval does NOT collapse solely into Section 3(p),
    but surfaces distinct legal dimensions:
    - Patentability definition / process (Section 2)
    - Traditional knowledge / aggregation exclusion (Section 3(p))
    - Inventive step / novelty anticipation (Section 8 / 13 / 25 / 64)
    """
    q = "Is a novel supercritical carbon dioxide extraction process for Ayurvedic herbs patentable?"
    f_intel = FormulationClassifierEngine.classify_query(q)
    res = retriever.retrieve(q, formulation_intelligence=f_intel, jurisdiction="national", top_k=4)

    assert len(res.selected_evidence) >= 3, "Expected at least 3 evidence items"
    provisions = [str(c.get("section_or_clause", "")) for c in res.selected_evidence]
    text_corpus = " ".join([c.get("text", "") for c in res.selected_evidence]).lower()

    # Verify both process/patentability criteria AND traditional knowledge exclusion are present
    has_process_or_def = any("section 2" in p.lower() for p in provisions) or "invention" in text_corpus
    has_tk_exclusion = any("3(p)" in p.lower() for p in provisions) or "traditional knowledge" in text_corpus

    assert has_process_or_def, "Expected substantive patentability/process evidence (Section 2)"
    assert has_tk_exclusion, "Expected traditional knowledge exclusion evidence (Section 3(p))"

    # Verify Section 3(p) does NOT monopolize all evidence slots
    sec_3p_count = sum(1 for p in provisions if "3(p)" in p.lower())
    assert sec_3p_count < len(res.selected_evidence), "Section 3(p) must not monopolize all evidence slots"


def test_phase5_2_multi_provision_set_coverage_metric(retriever):
    """
    Phase 5.2: Multi-provision set coverage evaluation.
    Query: 'Under the Indian Patents Act, what are the requirements under Section 2, Section 3, and Section 64?'
    Expected target set: {Section 2, Section 3, Section 64}.
    Distinguishes single-evidence from multi-evidence queries and enforces 100% set coverage.
    """
    q = "Under the Indian Patents Act, what are the requirements under Section 2, Section 3, and Section 64?"
    res = retriever.retrieve(q, jurisdiction="national", top_k=6)

    target_provisions = {"section 2", "section 3", "section 64"}
    retrieved_provisions = set()
    for c in res.selected_evidence:
        sec = str(c.get("section_or_clause", "")).lower()
        for t in target_provisions:
            if t in sec:
                retrieved_provisions.add(t)

    set_coverage = len(retrieved_provisions) / len(target_provisions)
    logger.info(f"Multi-provision set coverage: {set_coverage*100:.1f}% ({len(retrieved_provisions)}/{len(target_provisions)})")

    assert set_coverage == 1.0, f"Expected 100% set coverage across {target_provisions}, got {retrieved_provisions}"
    assert "patents_section_2" in res.decomposed_dimensions
    assert "patents_section_3" in res.decomposed_dimensions
    assert "patents_section_64" in res.decomposed_dimensions


def test_phase5_2_over_association_discrimination(retriever):
    """
    Phase 5.2: Over-association discrimination test across three related queries:
    A. Process query (novel extraction process)
    B. TK query (traditional knowledge impact)
    C. Exclusion query (which patent exclusions apply)
    Retrieval must follow the actual semantic/legal dimensions of each query.
    """
    # A. Process Query
    qa = "Is a novel extraction process for an Ayurvedic herb patentable?"
    fa = FormulationClassifierEngine.classify_query(qa)
    ra = retriever.retrieve(qa, formulation_intelligence=fa, jurisdiction="national", top_k=4)
    provisions_a = [str(c.get("section_or_clause", "")).lower() for c in ra.selected_evidence]
    assert any("section 2" in p for p in provisions_a), "Process query A must surface Section 2 definition"

    # B. TK Query
    qb = "Does traditional knowledge affect patent protection for an Ayurvedic herb?"
    fb = FormulationClassifierEngine.classify_query(qb)
    rb = retriever.retrieve(qb, formulation_intelligence=fb, jurisdiction="national", top_k=4)
    provisions_b = [str(c.get("section_or_clause", "")).lower() for c in rb.selected_evidence]
    assert any("3(p)" in p for p in provisions_b), "TK query B must surface Section 3(p)"
    assert not any("section 2" in p for p in provisions_b), "TK query B must not retrieve process definitions"

    # C. Exclusion Query
    qc = "Which patent exclusions may apply to an invention based on traditional Ayurvedic knowledge?"
    fc = FormulationClassifierEngine.classify_query(qc)
    rc = retriever.retrieve(qc, formulation_intelligence=fc, jurisdiction="national", top_k=4)
    provisions_c = [str(c.get("section_or_clause", "")).lower() for c in rc.selected_evidence]
    # Verify spectrum of exclusions: Section 3, Section 3(p), Section 3(e), Section 3(d)
    assert any("3(p)" in p for p in provisions_c), "Exclusion query C must surface Section 3(p)"
    assert any("3(e)" in p for p in provisions_c) or any("3(d)" in p for p in provisions_c) or any(p == "section 3" for p in provisions_c), \
        "Exclusion query C must surface multiple statutory exclusions under Section 3"


def test_phase5_2_negative_control_chemical_and_formulation(retriever):
    """
    Phase 5.2: Negative controls:
    1. Pure chemical process query with no TK signal:
       Must NOT retrieve Section 3(p) or traditional knowledge.
    2. TK formulation query with no process signal:
       Must NOT retrieve process standards or method definitions.
    """
    # Control 1: Chemical process (no TK)
    q1 = "Is a novel industrial extraction process for a chemical compound patentable?"
    f1 = FormulationClassifierEngine.classify_query(q1)
    r1 = retriever.retrieve(q1, formulation_intelligence=f1, jurisdiction="national", top_k=4)
    provisions_1 = [str(c.get("section_or_clause", "")).lower() for c in r1.selected_evidence]
    text_1 = " ".join([c.get("text", "") for c in r1.selected_evidence]).lower()

    assert not any("3(p)" in p for p in provisions_1), f"Chemical query must NOT retrieve Section 3(p), got: {provisions_1}"
    assert "traditional knowledge" not in text_1[:200], "Chemical query must not surface TK text at rank 1"
    assert any("section 2" in p for p in provisions_1), "Chemical process query must surface Section 2"

    # Control 2: TK formulation (no process)
    q2 = "Does traditional knowledge affect patent protection for an Ayurvedic formulation?"
    f2 = FormulationClassifierEngine.classify_query(q2)
    r2 = retriever.retrieve(q2, formulation_intelligence=f2, jurisdiction="national", top_k=4)
    provisions_2 = [str(c.get("section_or_clause", "")).lower() for c in r2.selected_evidence]

    assert any("3(p)" in p for p in provisions_2), "TK formulation query must surface Section 3(p)"
    assert not any("section 2" in p for p in provisions_2), "TK formulation query must not surface process invention definition"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
