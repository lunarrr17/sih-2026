import re
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ProductCategoryEnum(str, Enum):
    CLASSICAL_AYURVEDIC = "Classical / Generic Ayurvedic Medicine"
    PATENT_PROPRIETARY = "Patent or Proprietary (P&P) Ayurvedic Medicine"
    PHYTOPHARMACEUTICAL = "Phytopharmaceutical Drug (CDSCO)"
    AYURVEDA_AAHAR = "Ayurveda Aahar (FSSAI Food Supplement)"
    COSMETIC_AYUSH = "AYUSH Cosmetic / Topical Care"

class TriageFormulationInput(BaseModel):
    product_name: str
    is_first_schedule_text: bool = True
    extraction_method: str = "traditional_aqueous_decoction"
    # options: 'traditional_aqueous_decoction', 'traditional_powder', 'standardized_hydroalcoholic_extract', 'isolated_purified_phytochemical_fraction'
    delivery_format: str = "traditional_ghrita_oil"
    # options: 'traditional_ghrita_oil', 'traditional_powder', 'hard_gelatin_capsule', 'topical_emulgel', 'nano_liposomal_emulsion', 'food_beverage_powder'
    has_comparative_synergy_data: bool = False
    synergy_percentage_increase: Optional[float] = 0.0
    is_registered_ayush_practitioner: bool = False
    is_foreign_entity_or_nri: bool = False
    intended_use: str = "therapeutic_internal"
    # options: 'therapeutic_internal', 'therapeutic_topical', 'food_wellness_supplement', 'cosmetic_external'

class TriageFormulationOutput(BaseModel):
    product_name: str
    category: ProductCategoryEnum
    patent_status: str  # 'BARRED' | 'CONDITIONALLY_ELIGIBLE' | 'HIGHLY_ELIGIBLE'
    patent_rationale: str
    abs_status: str     # 'EXEMPTED' | 'ACTION_REQUIRED' | 'MANDATORY_NBA_APPROVAL'
    abs_rationale: str
    licensing_framework: str
    licensing_forms: List[str]
    compliance_checklist: List[str]
    statutory_citations: List[str]

class FormulationTriageEngine:
    """
    5-Step Statutory Formulation Triage Engine for Indian Ayurvedic Medicine,
    Patentability Posture (IPO Section 3), and Biodiversity ABS Matrix (NBA / BD Act 2023).
    """

    FIRST_SCHEDULE_TEXTS = [
        "Charaka Samhita", "Sushruta Samhita", "Ashtanga Hridaya", "Ashtanga Sangraha",
        "Ayurvedic Formulary of India (AFI)", "Ayurvedic Pharmacopoeia of India (API)",
        "Sahasrayogam", "Bhavaprakasha Nighantu", "Sharangadhara Samhita", "Bhaishajya Ratnavali",
        "Rasa Tarangini", "Rasa Ratna Samucchaya", "Chikitsa Kalika", "Madhava Nidana"
    ]

    def evaluate(self, inp: TriageFormulationInput) -> TriageFormulationOutput:
        # Step 1: Check Food Supplement (Ayurveda Aahar)
        if inp.intended_use == "food_wellness_supplement":
            return TriageFormulationOutput(
                product_name=inp.product_name,
                category=ProductCategoryEnum.AYURVEDA_AAHAR,
                patent_status="BARRED",
                patent_rationale=(
                    "BARRED under Section 3(p) as Traditional Food / Recipe. "
                    "Food supplements formulated from authoritative Ayurvedic ingredients without proprietary therapeutic innovation are in the public domain."
                ),
                abs_status="EXEMPTED" if not inp.is_foreign_entity_or_nri else "ACTION_REQUIRED",
                abs_rationale="Normally covered under normally traded commodities (NTAC) under Section 40 of Biological Diversity Act if sourced within India.",
                licensing_framework="FSSAI (Food Safety and Standards - Ayurveda Aahar Regulations, 2022)",
                licensing_forms=["FSSAI Form B (Central/State License)", "Ayurveda Aahar Logo Mandatory Display"],
                compliance_checklist=[
                    "Display special Ayurveda Aahar logo prominently on front-of-pack.",
                    "Strict prohibition against therapeutic, disease-prevention, or drug claims.",
                    "Ingredients must strictly feature in First Schedule texts or API Part I."
                ],
                statutory_citations=[
                    "FSSAI (Ayurveda Aahar) Regulations, 2022",
                    "The Patents Act, 1970 - Section 3(p)"
                ]
            )

        # Step 2: Check Phytopharmaceutical (Purified Fractions / Modern Drug Trials)
        if inp.extraction_method == "isolated_purified_phytochemical_fraction":
            return TriageFormulationOutput(
                product_name=inp.product_name,
                category=ProductCategoryEnum.PHYTOPHARMACEUTICAL,
                patent_status="HIGHLY_ELIGIBLE",
                patent_rationale=(
                    "HIGHLY ELIGIBLE for Composition / Process Patent under Section 3(d) & 3(e). "
                    "Purified and standardized fractions exhibiting characterized chemical profiles and novel efficacy can overcome Section 3(p) traditional knowledge barriers."
                ),
                abs_status="MANDATORY_NBA_APPROVAL" if inp.is_foreign_entity_or_nri else "ACTION_REQUIRED",
                abs_rationale=(
                    "Mandatory prior approval from National Biodiversity Authority (NBA Form III) "
                    "prior to filing or grant of patent under Section 6 of the Biological Diversity Act."
                ),
                licensing_framework="CDSCO (Drugs and Cosmetics Rules, Schedule Y / Phytopharmaceutical Drug Regulation)",
                licensing_forms=["CDSCO Form 44 (New Drug Application)", "NBA Form III (Patent Prior Approval)"],
                compliance_checklist=[
                    "Standardize fraction to minimum 4 active chemical biomarkers.",
                    "Conduct Pre-clinical toxicology and Phase I-III human clinical trials under CDSCO GCP guidelines.",
                    "File Form III with NBA before commercialization or patent grant."
                ],
                statutory_citations=[
                    "Drugs and Cosmetics (Twelfth Amendment) Rules, 2015 - Phytopharmaceutical Drugs",
                    "The Patents Act, 1970 - Section 3(d), 3(e)",
                    "Biological Diversity Act, 2002/2023 - Section 6, Form III"
                ]
            )

        # Step 3: Pure Classical Ayurvedic Medicine
        if inp.is_first_schedule_text and inp.extraction_method in ["traditional_aqueous_decoction", "traditional_powder"]:
            abs_status = "EXEMPTED" if inp.is_registered_ayush_practitioner else ("ACTION_REQUIRED" if not inp.is_foreign_entity_or_nri else "MANDATORY_NBA_APPROVAL")
            abs_rationale = (
                "STATUTORY EXEMPTION: Registered AYUSH Practitioners, Vaidyas, and community traditional healers "
                "are statutory exempt from prior intimation and ABS fees under Section 40 Proviso of BD (Amendment) Act 2023."
                if inp.is_registered_ayush_practitioner
                else "Indian commercial manufacturers must submit prior intimation to State Biodiversity Board (SBB Form I)."
            )

            return TriageFormulationOutput(
                product_name=inp.product_name,
                category=ProductCategoryEnum.CLASSICAL_AYURVEDIC,
                patent_status="BARRED",
                patent_rationale=(
                    "BARRED under Section 3(p) of the Patents Act, 1970 as absolute Traditional Knowledge. "
                    "Classical formulations codified in the 54 authoritative First Schedule books belong to the public domain and are defended globally by CSIR/AYUSH TKDL."
                ),
                abs_status=abs_status,
                abs_rationale=abs_rationale,
                licensing_framework="Drugs and Cosmetics Act, 1940 (Chapter IV-A) & First Schedule Authoritative Texts",
                licensing_forms=["Form 24D / 25D (State AYUSH SLA Manufacturing License)", "Schedule T GMP Certification"],
                compliance_checklist=[
                    "Must be manufactured verbatim in accordance with specified First Schedule text (e.g. Charaka / AFI).",
                    "Mandatory labelling compliance under Rule 161 (true list of ingredients with classical reference).",
                    "Schedule T GMP compliance in licensed factory premises."
                ],
                statutory_citations=[
                    "Drugs and Cosmetics Act, 1940 - Section 3(a), First Schedule, Rule 161",
                    "The Patents Act, 1970 - Section 3(p)",
                    "Biological Diversity (Amendment) Act, 2023 - Section 40 Proviso"
                ]
            )

        # Step 4: Patent or Proprietary (P&P) Ayurvedic Medicine
        if inp.has_comparative_synergy_data:
            return TriageFormulationOutput(
                product_name=inp.product_name,
                category=ProductCategoryEnum.PATENT_PROPRIETARY,
                patent_status="CONDITIONALLY_ELIGIBLE",
                patent_rationale=(
                    f"CONDITIONALLY ELIGIBLE under Section 3(e) of the Patents Act. "
                    f"Your comparative trial data (showing {inp.synergy_percentage_increase or 30}% enhanced efficacy) "
                    "provides empirical evidence of synergistic interaction overcoming Section 3(p) traditional knowledge admixture bars."
                ),
                abs_status="ACTION_REQUIRED",
                abs_rationale="Prior intimation to State Biodiversity Board (SBB) under Section 7 of the BD Act is mandatory for commercial production.",
                licensing_framework="Drugs and Cosmetics Act, 1940 (Rule 158B - Patent or Proprietary Ayurvedic Medicine)",
                licensing_forms=["Form 24D (SLA Manufacturing License for P&P Drug)", "NBA Form III (for Patent Filing)"],
                compliance_checklist=[
                    "Submit safety/stability trial reports and textual justification under Rule 158B.",
                    "File Form 27 (Commercial Working Statement) annually under Patent Amendment Rules 2024.",
                    "File NBA Form III approval before patent sealing."
                ],
                statutory_citations=[
                    "Drugs and Cosmetics Rules, 1945 - Rule 158B",
                    "The Patents Act, 1970 - Section 3(e), Section 3(p)",
                    "Patent Amendment Rules, 2024 - Form 27",
                    "Biological Diversity Act, 2023 - Section 6, 7"
                ]
            )

        # Step 5: Proprietary without Synergy Proof (Mere Admixture Bar)
        return TriageFormulationOutput(
            product_name=inp.product_name,
            category=ProductCategoryEnum.PATENT_PROPRIETARY,
            patent_status="BARRED",
            patent_rationale=(
                "BARRED under Section 3(e) of the Patents Act as a 'Mere Admixture'. "
                "Combining known Ayurvedic ingredients without comparative clinical/in-vitro proof of unexpected synergistic enhancement "
                "is barred from patent grant under Section 3(e) and 3(p)."
            ),
            abs_status="ACTION_REQUIRED",
            abs_rationale="Commercial manufacture requires prior intimation to State Biodiversity Board (SBB Section 7).",
            licensing_framework="Drugs and Cosmetics Act, 1940 (Rule 158B - Patent or Proprietary Medicine)",
            licensing_forms=["Form 24D (State AYUSH SLA Manufacturing License)"],
            compliance_checklist=[
                "Obtain Rule 158B SLA license for commercial sale without patent claims.",
                "Conduct comparative synergistic in-vitro/animal studies if you wish to file for patent protection.",
                "Mandatory Schedule T GMP compliance."
            ],
            statutory_citations=[
                "Drugs and Cosmetics Rules, 1945 - Rule 158B",
                "The Patents Act, 1970 - Section 3(e), Section 3(p)",
                "Biological Diversity Act, 2023 - Section 7"
            ]
        )

# Global Triage Engine singleton
triage_engine = FormulationTriageEngine()
