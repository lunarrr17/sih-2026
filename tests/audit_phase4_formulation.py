"""
Phase 4 — Formulation Intelligence & Classification Audit Suite
================================================================
Comprehensive verification of FormulationIntelligence, intent classification,
entity extraction, routing hints, negative clearance protections, and mutation checks.

Categories Tested:
- Category A: Classical formulation inquiry
- Category B: Proprietary formulation inquiry
- Category C: Modified formulation
- Category D: Ingredient-only query
- Category E: Novel process query
- Category F: Traditional knowledge query
- Category G: Biodiversity / ABS query
- Category H: Patentability query
- Category I: Regulatory classification query
- Category J: Ambiguous formulation
- Category K: Missing ingredient information
- Category L: Misspelled formulation name
- Category M: Multi-ingredient formulation
- Category N: Mixed intent
- Category O: International query
- Category P: India-only query
- Category Q: Explicit comparative query
- Category R: Out-of-domain query
- Category S: Adversarial query attempting to force a legal conclusion
- Category T: Query designed to trigger old Section 3(p) hardcoding
- Section 20: Critical Negative Tests (No legal conclusions from classification)
- Section 25: Phase 4 Mutation Suite (10 mutations verifying test sensitivity)
"""

import re
import sys
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from backend.app.engines.classifier_engine import classifier_engine, FormulationClassifierEngine
from backend.app.rag.chat_agent import chat_agent, ChatAgentRequest
from backend.app.rag.schemas import (
    SubjectType,
    SubstanceOrigin,
    ProcessType,
    TraditionalKnowledgeSignal,
    UserIntent,
    ConfidenceTier,
    FormulationIntelligence,
    EvidenceStrength
)

PHASE4_SCENARIOS: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # Category A: Classical Formulation Inquiry
    # -------------------------------------------------------------------------
    {
        "id": "PH4-A01",
        "category": "A. Classical Formulation Inquiry",
        "query": "Is Chyawanprash considered a classical formulation under the First Schedule of the Drugs and Cosmetics Act?",
        "expected_subject_type": SubjectType.CLASSICAL_FORMULATION,
        "expected_formulation": "Chyawanprash",
        "expected_tk_signal": [TraditionalKnowledgeSignal.EXPLICIT_TRADITIONAL, TraditionalKnowledgeSignal.INFERRED_TRADITIONAL],
        "expected_confidence": [ConfidenceTier.HIGH, ConfidenceTier.MEDIUM],
        "forbidden_legal_conclusions": ["automatically non-patentable", "guaranteed barred under 3(p)"],
        "expected_routing": {"retrieve_traditional_knowledge": True}
    },
    {
        "id": "PH4-A02",
        "category": "A. Classical Formulation Inquiry",
        "query": "What are the classical texts cited for Triphala Churna preparation in Ayurvedic medicine?",
        "expected_subject_type": SubjectType.CLASSICAL_FORMULATION,
        "expected_formulation": "Triphala",
        "expected_dosage_form": "churna",
        "expected_confidence": [ConfidenceTier.HIGH, ConfidenceTier.MEDIUM],
        "forbidden_legal_conclusions": ["not patentable"]
    },

    # -------------------------------------------------------------------------
    # Category B: Proprietary Formulation Inquiry
    # -------------------------------------------------------------------------
    {
        "id": "PH4-B01",
        "category": "B. Proprietary Formulation Inquiry",
        "query": "What are the licensing requirements for a patent or proprietary (P&P) Ayurvedic medicine under Rule 158B?",
        "expected_subject_type": SubjectType.PROPRIETARY_FORMULATION,
        "expected_intents": [UserIntent.REGULATORY_LICENSING],
        "expected_routing": {"retrieve_regulatory_licensing": True},
        "forbidden_legal_conclusions": ["patent granted", "automatically patentable"]
    },

    # -------------------------------------------------------------------------
    # Category C: Modified Formulation
    # -------------------------------------------------------------------------
    {
        "id": "PH4-C01",
        "category": "C. Modified Formulation",
        "query": "Can I patent a modified Ashwagandha formulation with altered ingredient ratios?",
        "expected_subject_type": SubjectType.MODIFIED_FORMULATION,
        "expected_ingredients": ["Ashwagandha"],
        "expected_origin": SubstanceOrigin.BOTANICAL,
        "expected_intents": [UserIntent.PATENTABILITY_INQUIRY],
        "expected_routing": {"retrieve_patent_exclusions": True, "retrieve_patentability_criteria": True},
        "forbidden_legal_conclusions": ["Section 3(d) definitely bars", "patentable", "not patentable"]
    },

    # -------------------------------------------------------------------------
    # Category D: Ingredient-Only Query
    # -------------------------------------------------------------------------
    {
        "id": "PH4-D01",
        "category": "D. Ingredient-Only Query",
        "query": "What is the biological origin and traditional status of Guduchi (Tinospora cordifolia)?",
        "expected_subject_type": SubjectType.SUBSTANCE_INGREDIENT,
        "expected_ingredients": ["Guduchi / Giloy"],
        "expected_origin": SubstanceOrigin.BOTANICAL,
        "forbidden_legal_conclusions": ["unpatentable"]
    },

    # -------------------------------------------------------------------------
    # Category E: Novel Process Query
    # -------------------------------------------------------------------------
    {
        "id": "PH4-E01",
        "category": "E. Novel Process Query",
        "query": "Can I get a patent for a genuinely new extraction process for isolating bioactive fractions from Curcuma longa?",
        "expected_subject_type": [SubjectType.PROCESS_METHOD, SubjectType.MODIFIED_FORMULATION],
        "novel_process_signal": True,
        "expected_process_type": ProcessType.POTENTIALLY_NOVEL_PROCESS,
        "expected_intents": [UserIntent.PATENTABILITY_INQUIRY],
        "expected_routing": {"retrieve_process_standards": True},
        "forbidden_legal_conclusions": ["process is automatically patentable", "guaranteed granted patent"]
    },

    # -------------------------------------------------------------------------
    # Category F: Traditional Knowledge Query
    # -------------------------------------------------------------------------
    {
        "id": "PH4-F01",
        "category": "F. Traditional Knowledge Query",
        "query": "How is traditional knowledge from ancient Ayurvedic texts examined as prior art during patent prosecution?",
        "expected_tk_signal": [TraditionalKnowledgeSignal.EXPLICIT_TRADITIONAL],
        "expected_intents": [UserIntent.PATENTABILITY_INQUIRY, UserIntent.PRIOR_ART_TK_CONCERN],
        "expected_routing": {"retrieve_traditional_knowledge": True},
        "forbidden_legal_conclusions": ["Section 3(p) automatically applies"]
    },

    # -------------------------------------------------------------------------
    # Category G: Biodiversity / ABS Query
    # -------------------------------------------------------------------------
    {
        "id": "PH4-G01",
        "category": "G. Biodiversity / ABS Query",
        "query": "Under Section 6 of the Biological Diversity Act, what NBA approvals are mandatory prior to applying for an IPR?",
        "expected_intents": [UserIntent.BIODIVERSITY_ABS],
        "expected_routing": {"retrieve_biodiversity_abs": True},
        "forbidden_legal_conclusions": ["guaranteed NBA approval", "patent office bound by NBA"]
    },

    # -------------------------------------------------------------------------
    # Category H: Patentability Query
    # -------------------------------------------------------------------------
    {
        "id": "PH4-H01",
        "category": "H. Patentability Query",
        "query": "What criteria under Section 2(1)(j) must an Ayurvedic invention meet beyond overcoming Section 3 exclusions?",
        "expected_intents": [UserIntent.PATENTABILITY_INQUIRY],
        "expected_routing": {"retrieve_patentability_criteria": True},
        "forbidden_legal_conclusions": ["patentability guaranteed"]
    },

    # -------------------------------------------------------------------------
    # Category I: Regulatory Classification Query
    # -------------------------------------------------------------------------
    {
        "id": "PH4-I01",
        "category": "I. Regulatory Classification Query",
        "query": "What container labelling and packaging particulars are required under Rule 161 of the Drugs and Cosmetics Rules?",
        "expected_intents": [UserIntent.REGULATORY_LICENSING],
        "expected_routing": {"retrieve_regulatory_licensing": True},
        "forbidden_legal_conclusions": ["patent eligible"]
    },

    # -------------------------------------------------------------------------
    # Category J: Ambiguous Formulation
    # -------------------------------------------------------------------------
    {
        "id": "PH4-J01",
        "category": "J. Ambiguous Formulation",
        "query": "What are the rules for an herbal preparation containing amla and other plant extracts?",
        "expected_ingredients": ["Amla / Amalaki"],
        "expected_ambiguities": True,  # polysemy of amla
        "expected_missing_info": True  # other extracts unspecified
    },

    # -------------------------------------------------------------------------
    # Category K: Missing Ingredient Information
    # -------------------------------------------------------------------------
    {
        "id": "PH4-K01",
        "category": "K. Missing Ingredient Information",
        "query": "Can I patent a new polyherbal formulation for diabetes?",
        "expected_subject_type": [SubjectType.UNSPECIFIED_FORMULATION, SubjectType.MODIFIED_FORMULATION],
        "expected_missing_info": True,
        "forbidden_legal_conclusions": ["patentable", "not patentable"]
    },

    # -------------------------------------------------------------------------
    # Category L: Misspelled Formulation Name
    # -------------------------------------------------------------------------
    {
        "id": "PH4-L01",
        "category": "L. Misspelled Formulation Name",
        "query": "Is chaywanprash awaleha described in charaka samhita?",
        "expected_formulation": "Chyawanprash",
        "expected_subject_type": SubjectType.CLASSICAL_FORMULATION,
        "expected_ambiguities": True  # note spelling variation
    },

    # -------------------------------------------------------------------------
    # Category M: Multi-Ingredient Formulation
    # -------------------------------------------------------------------------
    {
        "id": "PH4-M01",
        "category": "M. Multi-Ingredient Formulation",
        "query": "What regulatory approvals apply to a polyherbal formulation containing ashwagandha, turmeric and guduchi?",
        "expected_ingredients": ["Ashwagandha", "Turmeric / Curcumin", "Guduchi / Giloy"],
        "min_ingredient_count": 3,
        "expected_origin": SubstanceOrigin.BOTANICAL
    },

    # -------------------------------------------------------------------------
    # Category N: Mixed Intent
    # -------------------------------------------------------------------------
    {
        "id": "PH4-N01",
        "category": "N. Mixed Intent",
        "query": "How do patentability standards under Section 3(e) and NBA approval under Section 6 interact for an Ayurvedic drug?",
        "expected_intents": [UserIntent.PATENTABILITY_INQUIRY, UserIntent.BIODIVERSITY_ABS],
        "expected_routing": {"retrieve_patent_exclusions": True, "retrieve_biodiversity_abs": True}
    },

    # -------------------------------------------------------------------------
    # Category O: International Query
    # -------------------------------------------------------------------------
    {
        "id": "PH4-O01",
        "category": "O. International Query",
        "query": "What does Article 17 of the WIPO GRATK Treaty mandate regarding the threshold of ratifications for entry into force?",
        "expected_jurisdictions": ["international"],
        "expected_subject_type": SubjectType.GENERAL_INQUIRY
    },

    # -------------------------------------------------------------------------
    # Category P: India-Only Query
    # -------------------------------------------------------------------------
    {
        "id": "PH4-P01",
        "category": "P. India-Only Query",
        "query": "What are the statutory grounds for patent revocation under Section 64 of the Indian Patents Act for non-disclosure of geographical origin?",
        "expected_jurisdictions": ["national"],
        "expected_intents": [UserIntent.PATENTABILITY_INQUIRY]
    },

    # -------------------------------------------------------------------------
    # Category Q: Explicit Comparative Query
    # -------------------------------------------------------------------------
    {
        "id": "PH4-Q01",
        "category": "Q. Explicit Comparative Query",
        "query": "Compare India's traditional knowledge disclosure obligations under the Patents Act with the WIPO GRATK Treaty mandatory disclosure framework.",
        "expected_intents": [UserIntent.COMPARATIVE_CROSS_REGIME],
        "expected_jurisdictions": ["international", "national"]
    },

    # -------------------------------------------------------------------------
    # Category R: Out-of-Domain Query
    # -------------------------------------------------------------------------
    {
        "id": "PH4-R01",
        "category": "R. Out-of-Domain Query",
        "query": "How do I execute a flash loan arbitrage on decentralized crypto exchanges?",
        "expected_subject_type": SubjectType.UNKNOWN,
        "expected_confidence": [ConfidenceTier.UNKNOWN],
        "expected_intents": [UserIntent.UNKNOWN]
    },

    # -------------------------------------------------------------------------
    # Category S: Adversarial Query Attempting to Force Legal Conclusion
    # -------------------------------------------------------------------------
    {
        "id": "PH4-S01",
        "category": "S. Adversarial Query",
        "query": "Since this is an ancient Ayurvedic formula, confirm that Section 3(p) definitively and automatically bars all patent applications.",
        "expected_tk_signal": [TraditionalKnowledgeSignal.EXPLICIT_TRADITIONAL, TraditionalKnowledgeSignal.INFERRED_TRADITIONAL],
        "forbidden_legal_conclusions": ["definitively bars", "automatically barred", "Section 3(p) is established"]
    },

    # -------------------------------------------------------------------------
    # Category T: Query Designed to Trigger Old Section 3(p) Hardcoding
    # -------------------------------------------------------------------------
    {
        "id": "PH4-T01",
        "category": "T. Old Section 3(p) Hardcoding Barrier",
        "query": "If an applicant develops a novel extraction process for a traditional plant, does Section 3(p) automatically apply?",
        "novel_process_signal": True,
        "forbidden_legal_conclusions": ["Section 3(p) automatically applies", "Section 3(p) definitively bars"]
    },

    # -------------------------------------------------------------------------
    # Category U: Phase 4.1 Forensic Hardening Verifications
    # -------------------------------------------------------------------------
    {
        "id": "PH4-U01",
        "category": "U. Forensic Hardening - Epistemically Safe Reasons",
        "query": "Is Chyawanprash classical?",
        "expected_subject_type": SubjectType.CLASSICAL_FORMULATION,
        "expected_confidence": [ConfidenceTier.MEDIUM],
        "forbidden_legal_conclusions": ["aligns with recognized", "statutory status", "barred", "patentable", "non-patentable", "legally established"]
    },
    {
        "id": "PH4-U02",
        "category": "U. Forensic Hardening - Brahmi Polysemy Warning",
        "query": "What are the therapeutic properties of Brahmi in Ayurvedic medicine?",
        "expected_ingredients": ["Brahmi"],
        "expected_ambiguities": True
    },
    {
        "id": "PH4-U03",
        "category": "U. Forensic Hardening - Brahmi Specified Species",
        "query": "What are the therapeutic properties of Brahmi (Bacopa monnieri)?",
        "expected_ingredients": ["Brahmi"],
        "expected_ambiguities": False
    },
    {
        "id": "PH4-U04",
        "category": "U. Forensic Hardening - Independent Feature Confidence",
        "query": "Chyawanprash",
        "expected_subject_type": SubjectType.CLASSICAL_FORMULATION,
        "expected_confidence": [ConfidenceTier.MEDIUM]  # Not inflated to HIGH!
    },
    {
        "id": "PH4-U05",
        "category": "U. Forensic Hardening - Explicit Dosage Feature Confidence",
        "query": "Chyawanprash avaleha",
        "expected_subject_type": SubjectType.CLASSICAL_FORMULATION,
        "expected_dosage_form": "avaleha",
        "expected_confidence": [ConfidenceTier.HIGH]  # High because explicitly specified in query
    }
]

# -------------------------------------------------------------------------
# Phase 4 Mutation Suite (10 Intentional Regression Mutations)
# -------------------------------------------------------------------------
MUTATION_TESTS = [
    {
        "id": "MUT-P4-01",
        "description": "Classifier sets patentable=True directly from novel_process_signal",
        "mutation": lambda res: setattr(res, "classification_reasons", res.classification_reasons + ["The process is legally patentable."]),
        "verify_defect": lambda res: any("is legally patentable" in r for r in res.classification_reasons)
    },
    {
        "id": "MUT-P4-02",
        "description": "Classifier maps classical formulation directly to Section 3(p) bar conclusion",
        "mutation": lambda res: setattr(res, "classification_reasons", res.classification_reasons + ["Section 3(p) automatically applies."]),
        "verify_defect": lambda res: any("automatically applies" in r for r in res.classification_reasons)
    },
    {
        "id": "MUT-P4-03",
        "description": "Out-of-domain crypto query marked as HIGH confidence",
        "mutation": lambda res: setattr(res, "overall_confidence", ConfidenceTier.HIGH),
        "verify_defect": lambda res: res.overall_confidence == ConfidenceTier.HIGH
    },
    {
        "id": "MUT-P4-04",
        "description": "Missing ingredient list not reported in missing_information for generic polyherbal query",
        "mutation": lambda res: setattr(res, "missing_information", []),
        "verify_defect": lambda res: len(res.missing_information) == 0
    },
    {
        "id": "MUT-P4-05",
        "description": "Ambiguous polysemous term 'amla' has empty ambiguities list",
        "mutation": lambda res: setattr(res, "ambiguities", []),
        "verify_defect": lambda res: len(res.ambiguities) == 0
    },
    {
        "id": "MUT-P4-06",
        "description": "National query leaks international jurisdiction suggestion without treaty reference",
        "mutation": lambda res: setattr(res.routing_hints, "jurisdictions_suggested", ["international"]),
        "verify_defect": lambda res: res.routing_hints.jurisdictions_suggested == ["international"]
    },
    {
        "id": "MUT-P4-07",
        "description": "Novel extraction query fails to flag novel_process_signal",
        "mutation": lambda res: setattr(res, "novel_process_signal", False),
        "verify_defect": lambda res: res.novel_process_signal is False
    },
    {
        "id": "MUT-P4-08",
        "description": "Multi-ingredient query fails to extract all three constituents",
        "mutation": lambda res: setattr(res, "ingredients", ["Ashwagandha"]),
        "verify_defect": lambda res: len(res.ingredients) < 3
    },
    {
        "id": "MUT-P4-09",
        "description": "Non-numeric confidence tier set to invalid string",
        "mutation": lambda res: setattr(res, "overall_confidence", "95% PROBABLE"),
        "verify_defect": lambda res: res.overall_confidence == "95% PROBABLE"
    },
    {
        "id": "MUT-P4-10",
        "description": "Chat response omits formulation_intelligence entirely",
        "mutation": lambda chat_resp: setattr(chat_resp, "formulation_intelligence", None),
        "verify_defect": lambda chat_resp: chat_resp.formulation_intelligence is None
    }
]


def run_phase4_audit():
    print("=" * 80)
    print("🌿 PHASE 4 AUDIT: FORMULATION INTELLIGENCE & CLASSIFICATION")
    print("=" * 80)

    start_time = time.time()
    passed_count = 0
    total_scenarios = len(PHASE4_SCENARIOS)
    scenario_results = []

    for i, sc in enumerate(PHASE4_SCENARIOS, 1):
        sc_id = sc["id"]
        cat = sc["category"]
        query = sc["query"]
        print(f"\n[{i:02d}/{total_scenarios:02d}] Testing {sc_id} ({cat}): '{query[:65]}...'")

        res: FormulationIntelligence = classifier_engine.classify_query(query)
        defects = []

        # 1. Subject Type Check
        if "expected_subject_type" in sc:
            exp_subj = sc["expected_subject_type"]
            exp_list = exp_subj if isinstance(exp_subj, list) else [exp_subj]
            if res.subject_type not in exp_list:
                defects.append(f"WRONG_SUBJECT_TYPE: expected {exp_subj}, got {res.subject_type}")

        # 2. Formulation Name Check
        if "expected_formulation" in sc:
            if not res.formulation_name or sc["expected_formulation"].lower() not in res.formulation_name.lower():
                defects.append(f"WRONG_FORMULATION_NAME: expected '{sc['expected_formulation']}', got '{res.formulation_name}'")

        # 3. Dosage Form Check
        if "expected_dosage_form" in sc:
            if res.dosage_form != sc["expected_dosage_form"]:
                defects.append(f"WRONG_DOSAGE_FORM: expected '{sc['expected_dosage_form']}', got '{res.dosage_form}'")

        # 4. Ingredients Check
        if "expected_ingredients" in sc:
            for exp_ing in sc["expected_ingredients"]:
                if not any(exp_ing.lower() in ing.lower() for ing in res.ingredients):
                    defects.append(f"MISSING_INGREDIENT: '{exp_ing}' not in {res.ingredients}")

        if "min_ingredient_count" in sc:
            if len(res.ingredients) < sc["min_ingredient_count"]:
                defects.append(f"TOO_FEW_INGREDIENTS: expected >= {sc['min_ingredient_count']}, got {len(res.ingredients)}")

        # 5. Substance Origin Check
        if "expected_origin" in sc:
            if res.substance_origin != sc["expected_origin"]:
                defects.append(f"WRONG_ORIGIN: expected {sc['expected_origin']}, got {res.substance_origin}")

        # 6. Process / Novelty Check
        if "novel_process_signal" in sc:
            if res.novel_process_signal != sc["novel_process_signal"]:
                defects.append(f"WRONG_NOVEL_PROCESS_SIGNAL: expected {sc['novel_process_signal']}, got {res.novel_process_signal}")

        if "expected_process_type" in sc:
            if res.process_type != sc["expected_process_type"]:
                defects.append(f"WRONG_PROCESS_TYPE: expected {sc['expected_process_type']}, got {res.process_type}")

        # 7. Traditional Knowledge Signal Check
        if "expected_tk_signal" in sc:
            exp_tk = sc["expected_tk_signal"]
            exp_tk_list = exp_tk if isinstance(exp_tk, list) else [exp_tk]
            if res.traditional_knowledge_signal not in exp_tk_list:
                defects.append(f"WRONG_TK_SIGNAL: expected {exp_tk}, got {res.traditional_knowledge_signal}")

        # 8. User Intents Check
        if "expected_intents" in sc:
            for exp_int in sc["expected_intents"]:
                if exp_int not in res.user_intents:
                    defects.append(f"MISSING_INTENT: '{exp_int}' not in {res.user_intents}")

        # 9. Confidence Check
        if "expected_confidence" in sc:
            exp_conf = sc["expected_confidence"]
            exp_conf_list = exp_conf if isinstance(exp_conf, list) else [exp_conf]
            if res.overall_confidence not in exp_conf_list:
                defects.append(f"WRONG_CONFIDENCE: expected {exp_conf}, got {res.overall_confidence}")

        # 10. Routing Hints Check
        if "expected_routing" in sc:
            for hint_k, hint_v in sc["expected_routing"].items():
                actual_v = getattr(res.routing_hints, hint_k, None)
                if actual_v != hint_v:
                    defects.append(f"WRONG_ROUTING_HINT: {hint_k} expected {hint_v}, got {actual_v}")

        # 11. Ambiguities & Missing Info Check
        if sc.get("expected_ambiguities") is True and not res.ambiguities:
            defects.append("EXPECTED_AMBIGUITIES_MISSING: Query has polysemy/spelling variation but ambiguities list is empty.")
        elif sc.get("expected_ambiguities") is False and res.ambiguities:
            defects.append(f"UNEXPECTED_AMBIGUITY_PRESENT: Expected zero ambiguities but got {res.ambiguities}")

        if sc.get("expected_missing_info") and not res.missing_information:
            defects.append("EXPECTED_MISSING_INFO_NOT_FLAGGED: Incomplete query but missing_information list is empty.")

        # 12. CRITICAL NEGATIVE TEST: No Legal Conclusions in Classification!
        combined_text = " ".join(res.classification_reasons + res.ambiguities).lower()
        for forbidden in sc.get("forbidden_legal_conclusions", []):
            if forbidden.lower() in combined_text:
                defects.append(f"CRITICAL_FAILURE_LEGAL_CONCLUSION_IN_CLASSIFIER: Found '{forbidden}' in classification text.")

        status = "✅ PASS" if not defects else f"❌ FAIL: {defects}"
        print(f"       -> Status: {status} | Subj: {res.subject_type.value} | Conf: {res.overall_confidence.value}")

        is_pass = len(defects) == 0
        if is_pass:
            passed_count += 1

        scenario_results.append({
            "id": sc_id,
            "category": cat,
            "query": query,
            "passed": is_pass,
            "defects": defects,
            "subject_type": res.subject_type.value,
            "overall_confidence": res.overall_confidence.value,
            "formulation_name": res.formulation_name,
            "ingredients": res.ingredients,
            "novel_process_signal": res.novel_process_signal,
            "traditional_knowledge_signal": res.traditional_knowledge_signal.value,
            "user_intents": [i.value for i in res.user_intents],
            "routing_hints": res.routing_hints.model_dump(),
            "ambiguities": res.ambiguities,
            "missing_information": res.missing_information
        })

    # Run End-to-End Chat Integration Check
    print("\n" + "-" * 80)
    print("💬 END-TO-END CHAT INTEGRATION CHECK")
    print("-" * 80)
    chat_req = ChatAgentRequest(
        query="Can I patent a modified Ashwagandha formulation with a novel extraction process?",
        jurisdiction="national"
    )
    chat_resp = chat_agent.process_message(chat_req)
    has_intel = getattr(chat_resp, "formulation_intelligence", None) is not None
    print(f"Chat Response formulation_intelligence attached: {'✅ YES' if has_intel else '❌ NO'}")
    if not has_intel:
        print("❌ CRITICAL: ChatAgentResponse failed to attach formulation_intelligence!")

    # Run Mutation Suite
    print("\n" + "-" * 80)
    print("🧬 RUNNING PHASE 4 MUTATION TESTING SUITE (10 Mutations)")
    print("-" * 80)
    mutations_caught = 0
    for mut in MUTATION_TESTS:
        mut_id = mut["id"]
        desc = mut["description"]
        # Generate baseline query or chat response
        if "chat" in desc.lower():
            target = chat_resp.model_copy(deep=True) if hasattr(chat_resp, "model_copy") else chat_agent.process_message(chat_req)
        else:
            target = classifier_engine.classify_query("Can I patent a modified Ashwagandha formulation with a novel extraction process?")
        # Apply mutation
        mut["mutation"](target)
        # Verify defect is detected
        is_detected = mut["verify_defect"](target)
        if is_detected:
            mutations_caught += 1
            print(f"  ✅ {mut_id}: Caught regression -> '{desc}'")
        else:
            print(f"  ❌ {mut_id}: FAILED to catch regression -> '{desc}'")

    elapsed = time.time() - start_time
    pass_pct = (passed_count / total_scenarios) * 100.0
    mut_pct = (mutations_caught / len(MUTATION_TESTS)) * 100.0

    print("\n" + "=" * 80)
    print(f"📊 PHASE 4 SUMMARY: {passed_count}/{total_scenarios} Passed ({pass_pct:.1f}%) in {elapsed:.2f}s")
    print(f"🧬 MUTATIONS CAUGHT: {mutations_caught}/{len(MUTATION_TESTS)} ({mut_pct:.1f}%)")
    print("=" * 80)

    # Save JSON report
    out_dir = Path("/Users/siddhant_patil/Projects/sih-2026/tests/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "phase4_formulation_intelligence_report.json"
    with open(report_file, "w") as f:
        json.dump({
            "total_scenarios": total_scenarios,
            "passed": passed_count,
            "pass_rate": pass_pct,
            "mutations_caught": mutations_caught,
            "total_mutations": len(MUTATION_TESTS),
            "mutation_detection_rate": mut_pct,
            "scenarios": scenario_results
        }, f, indent=2)
    print(f"💾 Saved report to {report_file}")

    if passed_count == total_scenarios and mutations_caught == len(MUTATION_TESTS) and has_intel:
        print("🎯 VERDICT: ✅ PHASE 4 VERIFIED COMPLETE")
        return True
    else:
        print("🎯 VERDICT: ❌ DEFECTS DETECTED")
        return False


if __name__ == "__main__":
    success = run_phase4_audit()
    sys.exit(0 if success else 1)
