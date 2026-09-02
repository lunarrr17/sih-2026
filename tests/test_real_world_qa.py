"""
Unit and Integration Tests for Real-World Legal QA Validation
Covering Categories A through H (31 Total Scenarios) across all 9 PDFs in backend/data.
"""

import pytest
from backend.app.rag.chat_agent import chat_agent, ChatAgentRequest
from backend.app.rag.retriever import retriever


@pytest.fixture(scope="module", autouse=True)
def init_agent():
    retriever.initialize()


# =========================================================================
# Category A: Direct Statutory Questions
# =========================================================================

def test_q_a1_patents_act_section_3_categories():
    """Verify Section 3 non-patentable subject matter in Patents Act 1970."""
    q = "What are the general categories of inventions that are not patentable under Section 3 of the Patents Act, 1970?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Patents Act" in c.statute for c in resp.citations)
    assert any(3 in c.page_numbers for c in resp.citations)


def test_q_a2_patents_act_section_3e_admixture():
    """Verify Section 3(e) mere admixture bar on patentability."""
    q = "Under Section 3(e) of the Patents Act, 1970, when is an admixture of substances excluded from patentability?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Patents Act" in c.statute for c in resp.citations)
    assert any("3(e)" in c.section.lower() or 3 in c.page_numbers for c in resp.citations)


def test_q_a3_patents_act_section_3d_known_substance():
    """Verify Section 3(d) enhancement of efficacy requirement."""
    q = "What does Section 3(d) of the Patents Act say regarding the patentability of a new form of a known substance?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Patents Act" in c.statute for c in resp.citations)
    assert any(3 in c.page_numbers for c in resp.citations)


def test_q_a4_bd_act_section_6_ipr_approval():
    """Verify Section 6 requirement for NBA approval before applying for IPR."""
    q = "Under the Biological Diversity Act, what is the mandatory requirement for seeking intellectual property rights for inventions based on biological resources from India?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Biological Diversity" in c.statute for c in resp.citations)


# =========================================================================
# Category B: Section-Specific Questions
# =========================================================================

def test_q_b1_patents_act_section_3p_traditional_knowledge():
    """Verify Section 3(p) traditional knowledge bar."""
    q = "What specific bar does Section 3(p) of the Patents Act, 1970 impose on Ayurvedic formulations?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Patents Act" in c.statute for c in resp.citations)
    assert any("3(p)" in c.section.lower() or 3 in c.page_numbers for c in resp.citations)


def test_q_b2_drugs_cosmetics_rule_161_labelling():
    """Verify Rule 161 labelling requirements for Ayurvedic medicines."""
    q = "What are the labelling requirements for Ayurvedic medicines under Rule 161 of the Drugs and Cosmetics Rules?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Drugs and Cosmetics" in c.statute for c in resp.citations)
    assert any("161" in c.section.lower() or 190 in c.page_numbers or 198 in c.page_numbers for c in resp.citations)


def test_q_b3_drugs_cosmetics_rule_158b_licensing():
    """Verify Rule 158B guidelines for licensing Ayurvedic drugs."""
    q = "What are the guidelines for issuing manufacturing licenses for Ayurvedic drugs under Rule 158B?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Drugs and Cosmetics" in c.statute for c in resp.citations)
    assert any("158b" in c.section.lower() or 182 in c.page_numbers for c in resp.citations)


def test_q_b4_bd_act_section_7_proviso_exemption():
    """Verify Section 7(1) Proviso exemption for vaids, hakims, and registered AYUSH practitioners."""
    q = "What is the statutory exemption granted to vaids, hakims, and registered AYUSH practitioners under the Proviso to Section 7(1) of the Biological Diversity Act?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Biological Diversity" in c.statute for c in resp.citations)
    assert any(4 in c.page_numbers for c in resp.citations)


def test_q_b5_bd_act_section_40_normally_traded_commodities():
    """Verify Section 40 normally traded commodities exclusion."""
    q = "What does Section 40 of the Biological Diversity Act provide regarding Normally Traded Commodities?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Biological Diversity" in c.statute for c in resp.citations)
    assert any("40" in c.section.lower() or 14 in c.page_numbers for c in resp.citations)


# =========================================================================
# Category C: Regulatory Questions
# =========================================================================

def test_q_c1_ayurveda_aahar_disease_claims():
    """Verify FSSAI Ayurveda Aahar Regulation 4 prohibition on therapeutic claims."""
    q = "Can an Ayurveda Aahar product make therapeutic or medicinal claims under the FSSAI Regulations 2022?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Ayurveda Aahar" in c.statute for c in resp.citations)


def test_q_c2_ayurveda_aahar_recognized_books():
    """Verify Schedule A recognized texts for Ayurveda Aahar."""
    q = "What authoritative books are recognized for preparing Ayurveda Aahar formulations under the 2022 Regulations?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Ayurveda Aahar" in c.statute for c in resp.citations)


def test_q_c3_patents_rules_2024_form_27():
    """Verify triennial commercial working reporting under Patent Amendment Rules 2024."""
    q = "What is the new filing frequency for Form 27 regarding the commercial working of patents under the Patent Amendment Rules 2024?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Patents (Amendment) Rules, 2024" in c.statute for c in resp.citations)
    assert any(16 in c.page_numbers for c in resp.citations)


def test_q_c4_drugs_cosmetics_excipients():
    """Verify permissible excipients and additives under Drugs and Cosmetics Rules."""
    q = "What are the permissible excipients and additives for Ayurvedic medicines under the Drugs and Cosmetics Rules?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Drugs and Cosmetics" in c.statute for c in resp.citations)


# =========================================================================
# Category D: Traditional Knowledge Questions
# =========================================================================

def test_q_d1_first_schedule_classical_ayurveda():
    """Verify role of First Schedule authoritative books in defining classical drugs."""
    q = "What role does the First Schedule to the Drugs and Cosmetics Act play in establishing whether an Ayurvedic formulation is classical?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Drugs and Cosmetics" in c.statute for c in resp.citations)


def test_q_d2_academic_study_provenance():
    """Verify academic study is cited with secondary_academic_study source type."""
    q = "How does an academic legal study characterize the defensive protection of Indian traditional knowledge in patent examination?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any(c.source_type == "secondary_academic_study" for c in resp.citations)


def test_q_d3_proprietary_vs_classical_medicine():
    """Verify requirement for proving innovation over classical texts for proprietary medicine."""
    q = "Can an Ayurvedic formulation mentioned in Charaka Samhita be registered as a proprietary medicine without proving modification or novelty?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Drugs and Cosmetics" in c.statute for c in resp.citations)


# =========================================================================
# Category E: Biodiversity & ABS Questions
# =========================================================================

def test_q_e1_nba_prior_approval_for_ipr():
    """Verify who must obtain prior approval from NBA for IPR under Section 6."""
    q = "Who must obtain prior approval from the National Biodiversity Authority before applying for any intellectual property right involving Indian biological resources?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Biological Diversity" in c.statute for c in resp.citations)


def test_q_e2_ayush_practitioner_abs_exemption():
    """Verify exemption from ABS for Indian citizens, AYUSH practitioners, and cultivators."""
    q = "Are Indian citizens and AYUSH practitioners required to pay access and benefit-sharing fees for cultivating medicinal plants for local health traditions?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Biological Diversity" in c.statute for c in resp.citations)
    assert any(4 in c.page_numbers for c in resp.citations)


def test_q_e3_bd_act_2023_penalties():
    """Verify decriminalized penalties under Biological Diversity Amendment Act 2023."""
    q = "What does the Biological Diversity Amendment Act 2023 specify regarding penalties for accessing biological resources without approval?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Biological Diversity" in c.statute for c in resp.citations)


# =========================================================================
# Category F: International Treaty Questions
# =========================================================================

def test_q_f1_wipo_gratk_article_3_mandatory_disclosure():
    """Verify Article 3 mandatory disclosure of origin for GR and TK."""
    q = "What mandatory disclosure requirements does Article 3 of the WIPO GRATK Treaty 2024 impose on patent applicants?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="international"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("WIPO" in c.statute for c in resp.citations)
    assert any("3" in c.section for c in resp.citations)


def test_q_f2_wipo_gratk_article_4_non_retroactivity():
    """Verify Article 4 non-retroactivity of WIPO GRATK Treaty."""
    q = "Does the WIPO GRATK Treaty apply retroactively to patent applications filed before its entry into force?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="international"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("WIPO" in c.statute for c in resp.citations)
    assert any("4" in c.section for c in resp.citations)


def test_q_f3_trips_article_27_patentability_flexibility():
    """Verify TRIPS Article 27 patentability criteria and permissible exclusions."""
    q = "Under Article 27 of the TRIPS Agreement, what patentability criteria must inventions satisfy and what subject matter exclusions are permitted?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="international"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("TRIPS" in c.statute for c in resp.citations)


def test_q_f4_nagoya_protocol_annex_benefits():
    """Verify monetary and non-monetary benefit sharing in Nagoya Protocol Annex."""
    q = "What types of monetary and non-monetary benefits are recognized in the Annex to the Nagoya Protocol?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="international"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.70
    assert any("Nagoya" in c.statute for c in resp.citations)
    assert any(15 in c.page_numbers for c in resp.citations)


# =========================================================================
# Category G: Cross-Document & Comparative Questions
# =========================================================================

def test_q_g1_comparative_section_3p_vs_trips_article_27():
    """Verify comparative synthesis between Patents Act §3(p) and TRIPS Article 27."""
    q = "How does Indian Section 3(p) exclusion compare with the TRIPS Article 27 patentability flexibility?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="comparative"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.60
    assert "## 🇮🇳 National Regime" in resp.answer
    assert "## 🌐 International Regime" in resp.answer
    assert len(resp.citations) >= 2


def test_q_g2_comparative_bd_act_section_6_vs_wipo_gratk():
    """Verify comparative synthesis between BD Act §6 prior approval and WIPO GRATK disclosure."""
    q = "Compare the domestic ABS requirement under Section 6 of the Biological Diversity Act with the WIPO GRATK Treaty disclosure requirements."
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="comparative"))
    assert not resp.abstain
    assert resp.confidence_score >= 0.60
    assert "## 🇮🇳 National Regime" in resp.answer
    assert "## 🌐 International Regime" in resp.answer
    assert any("Biological Diversity" in c.statute for c in resp.citations)
    assert any("WIPO" in c.statute for c in resp.citations)


# =========================================================================
# Category H: Unsupported Questions (Must Safely Abstain)
# =========================================================================

def test_q_h1_abstain_nagoya_patent_fees():
    """Nagoya Protocol contains no patent filing fees."""
    q = "What does the Nagoya Protocol say about Indian patent filing fees?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="international"))
    assert resp.abstain
    assert resp.confidence_score == 0.0
    assert len(resp.citations) == 0


def test_q_h2_abstain_corporate_tax_rate():
    """Corporate tax rate is out of scope and absent from corpus."""
    q = "What is the corporate tax rate for Ayurvedic manufacturing companies under the Income Tax Act?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert resp.abstain
    assert resp.confidence_score == 0.0
    assert len(resp.citations) == 0


def test_q_h3_abstain_fictitious_ayurveda_act_software():
    """Fictitious statute and software algorithms must safely abstain."""
    q = "What does the Ayurveda Intellectual Property Protection Act, 2021 specify about software algorithms?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert resp.abstain
    assert resp.confidence_score == 0.0
    assert len(resp.citations) == 0


def test_q_h4_abstain_patent_application_inr_fee():
    """Exact INR fee table is absent from corpus (in 2003 Principal Rules First Schedule)."""
    q = "What is the exact official government fee in Indian Rupees for filing an ordinary patent application under Form 1?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert resp.abstain
    assert resp.confidence_score == 0.0
    assert len(resp.citations) == 0


def test_q_h5_abstain_section_3p_crypto():
    """Crypto blockchain contracts are out of scope and absent."""
    q = "What does Section 3(p) say about cryptocurrency blockchain contracts?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert resp.abstain
    assert resp.confidence_score == 0.0
    assert len(resp.citations) == 0


def test_q_h6_abstain_ipc_420_sentencing():
    """IPC 420 criminal sentencing is completely absent and out of scope."""
    q = "What are the criminal sentencing guidelines under Section 420 of the Indian Penal Code?"
    resp = chat_agent.process_message(ChatAgentRequest(query=q, jurisdiction="national"))
    assert resp.abstain
    assert resp.confidence_score == 0.0
    assert len(resp.citations) == 0
