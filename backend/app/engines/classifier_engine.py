from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class FormulationInput(BaseModel):
    product_name: str = "Chyawanprash Awaleha"
    is_first_schedule: bool = True
    has_proprietary_formulation: bool = False
    requires_clinical_safety_proof: bool = False
    is_isolated_phytoconstituent: bool = False
    product_use_category: str = "therapeutic"

class ClassificationResult(BaseModel):
    category_id: str
    category_name: str
    statutory_definition: str
    governing_regime: str
    licensing_authority: str
    ip_posture: Dict[str, str]
    abs_posture: Dict[str, str]
    statutory_citations: List[str]

class FormulationClassifierEngine:
    """
    Deterministic triage engine that classifies Ayurvedic products and calculates statutory IP/ABS posture.
    """

    @classmethod
    def classify(cls, inputs: FormulationInput) -> ClassificationResult:
        if inputs.is_first_schedule:
            return ClassificationResult(
                category_id="classical",
                category_name="Classical / Generic Ayurvedic Medicine",
                statutory_definition=(
                    "Ayurvedic drug manufactured strictly in accordance with formulations and methods "
                    "described in the 54 authoritative books of Ayurvedic medicine specified in the First Schedule "
                    "of the Drugs and Cosmetics Act, 1940 (e.g. Charaka Samhita, Sushruta Samhita, AFI, API)."
                ),
                governing_regime="Drugs and Cosmetics Act, 1940 (Chapter IV-A) & Schedule T (GMP)",
                licensing_authority="State Licensing Authority (SLA) - AYUSH Directorate",
                ip_posture={
                    "patent_eligibility": (
                        "Barred under Section 3(p) of the Patents Act, 1970 if claimed as an unmodified classical formulation. "
                        "Listing in the First Schedule of the D&C Act establishes regulatory status for drug licensing, which is distinct from patent-law Section 3(p) analysis. "
                        "Modified formulations or novel processes require separate examination of novelty, inventive step, and statutory exclusions."
                    ),
                    "trademark_protection": "Class 5 (Brand name/logo only; generic classical names cannot be trademarked exclusively).",
                    "geographical_indication": "Applicable if manufactured using demarcated GI raw materials (e.g. Navara rice, Nilambur teak)."
                },
                abs_posture={
                    "status": "EXEMPTED for Registered AYUSH Practitioners & Traditional Vaidyas under BD Act 2023 Section 7(1) Proviso.",
                    "commercial_manufacturing": "Indian entities must submit prior intimation to State Biodiversity Board (SBB) under Section 7."
                },
                statutory_citations=[
                    "Drugs and Cosmetics Act, 1940 - Section 3(a) & First Schedule",
                    "The Patents Act, 1970 - Section 3(p) & Section 10(4)(ii)(D)",
                    "Biological Diversity (Amendment) Act, 2023 - Section 7(1) Proviso & Section 7"
                ]
            )

        if inputs.has_proprietary_formulation:
            return ClassificationResult(
                category_id="proprietary",
                category_name="Patent or Proprietary (P&P) Ayurvedic Medicine",
                statutory_definition="Contains only ingredients described in First Schedule texts but in novel ratios, vehicles, or modern delivery forms.",
                governing_regime="Drugs and Cosmetics Act, 1940 (Rule 158B)",
                licensing_authority="State Licensing Authority (SLA)",
                ip_posture={
                    "patent_eligibility": "Conditional. Synergistic interaction must be demonstrated beyond mere aggregation under Section 3(e), while modified forms or processes require separate analysis of novelty and efficacy under Section 3(d) and Section 3(p)."
                },
                abs_posture={
                    "status": "Prior intimation to SBB required before commercialization."
                },
                statutory_citations=["Drugs and Cosmetics Rules 1945 - Rule 158B", "Patents Act - Section 3(e)"]
            )

        return ClassificationResult(
            category_id="general_ayush",
            category_name="General AYUSH Formulation",
            statutory_definition="Ayurvedic product under regulatory assessment.",
            governing_regime="Ministry of AYUSH Regulatory Directives",
            licensing_authority="State Licensing Authority (SLA)",
            ip_posture={"patent_eligibility": "Subject to Section 3(p) and Section 3(e) patentability guidelines."},
            abs_posture={"status": "Check SBB / NBA compliance requirements."},
            statutory_citations=["Drugs and Cosmetics Act, 1940"]
        )
