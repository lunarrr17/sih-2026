import pytest
from backend.app.rag.chat_agent import chat_agent, ChatAgentRequest

# =========================================================================
# PHASE 5: THE 14 GOLDEN EVALUATION TEST CASES
# =========================================================================

def test_q01_patents_act_section_3p_traditional_knowledge():
    """Q1: Patents Act §3(p) traditional knowledge bar."""
    query = "What does Section 3(p) of the Patents Act, 1970 say about traditional knowledge?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="national", session_id="test_q1"))

    assert not resp.abstain, "Q1 should not abstain"
    assert resp.is_grounded, "Q1 must be grounded"
    assert resp.confidence_score >= 0.70, "Q1 must have high confidence"
    assert len(resp.citations) > 0, "Q1 must include citations"

    # Must cite Patents Act
    statutes = [c.statute for c in resp.citations]
    assert any("Patents Act" in s for s in statutes), "Q1 must cite Patents Act, 1970"

    # Must cite Section 3(p) or Section 3
    sections = [c.section for c in resp.citations]
    assert any("3(p)" in sec or "3" in sec for sec in sections), "Q1 must cite Section 3(p)"

    # Text must reference traditional knowledge exclusion
    assert "traditional knowledge" in resp.answer.lower(), "Answer must discuss traditional knowledge"


def test_q02_aggregation_of_traditionally_known_properties():
    """Q2: Mere aggregation of known properties under Patents Act."""
    query = "What is the patentability issue when an invention is merely an aggregation of traditionally known properties?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="national", session_id="test_q2"))

    assert not resp.abstain, "Q2 should not abstain"
    assert resp.is_grounded, "Q2 must be grounded"
    assert len(resp.citations) > 0, "Q2 must provide citations"

    statutes = [c.statute for c in resp.citations]
    assert any("Patents Act" in s for s in statutes), "Q2 must cite Patents Act"
    assert "aggregation" in resp.answer.lower() or "duplication" in resp.answer.lower()


def test_q03_drugs_and_cosmetics_classical_formulations():
    """Q3: Drugs and Cosmetics Act provisions on classical formulations."""
    query = "What does the Drugs and Cosmetics framework say about classical Ayurvedic formulations?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="national", session_id="test_q3"))

    assert not resp.abstain, "Q3 should not abstain"
    assert resp.is_grounded, "Q3 must be grounded"
    statutes = [c.statute for c in resp.citations]
    assert any("Drugs and Cosmetics" in s for s in statutes), "Q3 must cite Drugs & Cosmetics Act"


def test_q04_charaka_samhita_relevance():
    """Q4: Relevance of Charaka Samhita to Ayurvedic formulation classification."""
    query = "What is the relevance of Charaka Samhita to Ayurvedic formulation classification?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="national", session_id="test_q4"))

    assert not resp.abstain, "Q4 should not abstain"
    assert resp.is_grounded, "Q4 must be grounded"
    assert len(resp.citations) > 0, "Q4 must include citations"
    # Should cite D&C First Schedule texts or Rule 158B
    statutes = [c.statute for c in resp.citations]
    assert any("Drugs and Cosmetics" in s or "Ayurveda Aahar" in s for s in statutes)


def test_q05_biological_diversity_framework_abs():
    """Q5: Biological Diversity Act access and benefit sharing."""
    query = "What does the Biological Diversity framework say about access and benefit sharing?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="national", session_id="test_q5"))

    assert not resp.abstain, "Q5 should not abstain"
    assert resp.is_grounded, "Q5 must be grounded"
    statutes = [c.statute for c in resp.citations]
    assert any("Biological Diversity" in s for s in statutes), "Q5 must cite Biological Diversity Act"


def test_q06_nagoya_protocol_access_and_benefit_sharing():
    """Q6: Nagoya Protocol on ABS under international jurisdiction."""
    query = "What does the Nagoya Protocol say about access and benefit-sharing?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="international", session_id="test_q6"))

    assert not resp.abstain, "Q6 should not abstain"
    assert resp.is_grounded, "Q6 must be grounded"
    statutes = [c.statute for c in resp.citations]
    assert all("Nagoya Protocol" in s for s in statutes), "Q6 must strictly cite Nagoya Protocol with zero national leakage"


def test_q07_wipo_gratk_treaty_requirements():
    """Q7: WIPO GRATK Treaty mandatory disclosure requirements."""
    query = "What does the WIPO treaty on genetic resources and associated traditional knowledge require?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="international", session_id="test_q7"))

    assert not resp.abstain, "Q7 should not abstain"
    assert resp.is_grounded, "Q7 must be grounded"
    statutes = [c.statute for c in resp.citations]
    assert any("WIPO" in s or "GRATK" in s for s in statutes), "Q7 must cite WIPO GRATK Treaty"


def test_q08_negative_tax_treatment_abstention():
    """Q8: Tax treatment must safely abstain (outside corpus)."""
    query = "What is the tax treatment of Ayurvedic formulations?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="national", session_id="test_q8"))

    assert resp.abstain, "Q8 must trigger safe abstention"
    assert not resp.is_grounded
    assert len(resp.citations) == 0, "No citations allowed on abstention"


def test_q09_negative_fictitious_law_abstention():
    """Q9: Fictitious law must safely abstain."""
    query = "What does the Ayurveda Intellectual Property Protection Act, 2021 say about cryptocurrency licensing?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="national", session_id="test_q9"))

    assert resp.abstain, "Q9 must trigger safe abstention"
    assert len(resp.citations) == 0


def test_q10_negative_general_knowledge_abstention():
    """Q10: General knowledge out of scope."""
    query = "Who is the current Prime Minister of India?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="national", session_id="test_q10"))

    assert resp.abstain, "Q10 must trigger safe abstention"
    assert len(resp.citations) == 0


def test_q11_negative_medical_advice_abstention():
    """Q11: Medical prescribing advice out of scope."""
    query = "What is the best Ayurvedic medicine for diabetes?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="national", session_id="test_q11"))

    assert resp.abstain, "Q11 must trigger safe abstention"
    assert len(resp.citations) == 0


def test_q12_negative_gst_registration_abstention():
    """Q12: Company GST registration out of scope."""
    query = "How should I register my company under GST?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="national", session_id="test_q12"))

    assert resp.abstain, "Q12 must trigger safe abstention"
    assert len(resp.citations) == 0


def test_q13_negative_section_3p_cryptocurrency_abstention():
    """Q13: Wrong concept attached to statutory section."""
    query = "What does Section 3(p) say about cryptocurrency?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="national", session_id="test_q13"))

    assert resp.abstain, "Q13 must trigger safe abstention"
    assert len(resp.citations) == 0


def test_q14_critical_regression_nagoya_indian_patent_filing_fees_abstention():
    """
    Q14: Critical regression test.
    The Nagoya Protocol contains licence fees in its Annex, but does NOT contain
    Indian patent filing fees. Must strictly abstain.
    """
    query = "What does the Nagoya Protocol say about Indian patent filing fees?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="international", session_id="test_q14"))

    assert resp.abstain, "Q14 MUST safely abstain"
    assert not resp.is_grounded, "Q14 must not claim to be grounded"
    assert resp.confidence_score == 0.0, "Q14 confidence must be 0.0 on abstention"
    assert len(resp.citations) == 0, "Q14 must have zero citations"


# =========================================================================
# PHASE 6: ADVERSARIAL TEST CASES
# =========================================================================

def test_adv_a_keyword_collision_nagoya_patent_filing_fees():
    """Adversarial A: Keyword collision — Nagoya Protocol + patent filing fees."""
    query = "Does the Nagoya Protocol discuss patent filing fees?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="international", session_id="test_adv_a"))

    assert resp.abstain, "Nagoya Protocol does not contain patent filing fees and must abstain"
    assert len(resp.citations) == 0


def test_adv_b_same_word_valid_meaning_nagoya_commercialization_licence_fees():
    """Adversarial B: Same word 'fees' with legitimate meaning in Nagoya Annex."""
    query = "Does the Nagoya Protocol discuss commercialization licence fees?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="international", session_id="test_adv_b"))

    assert not resp.abstain, "Nagoya Protocol explicitly mentions commercialization licence fees in its Annex and should answer"
    assert resp.is_grounded
    assert len(resp.citations) > 0
    assert all("Nagoya Protocol" in c.statute for c in resp.citations)


def test_adv_c_wrong_jurisdiction_indian_question_under_international():
    """Adversarial C: Domestic Indian question asked under international jurisdiction."""
    query = "What are the labelling rules under Rule 161 of the Drugs and Cosmetics Act?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="international", session_id="test_adv_c"))

    assert resp.abstain, "Domestic Indian rule asked under international jurisdiction must abstain"
    assert len(resp.citations) == 0


def test_adv_d_fake_statute_ayurvedic_herbal_export_act():
    """Adversarial D: Plausible sounding fake statute."""
    query = "What are the compliance requirements under the National Ayurvedic Herbal Export Promotion Act, 2019?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="national", session_id="test_adv_d"))

    assert resp.abstain, "Fake statute must trigger safe abstention"
    assert len(resp.citations) == 0


def test_adv_e_mixed_jurisdiction_unsupported_unified_claim():
    """Adversarial E: Mixed Indian and Nagoya question with unsupported premise."""
    query = "Under Indian law and the Nagoya Protocol, what are the patent filing fees?"
    resp = chat_agent.process_message(ChatAgentRequest(query=query, jurisdiction="national", session_id="test_adv_e"))

    # Nagoya does not have patent filing fees, so full unified claim cannot be grounded
    assert resp.abstain or "insufficient evidence" in resp.answer.lower()
