import pytest
from backend.app.engines.triage_engine import (
    FormulationTriageEngine,
    TriageFormulationInput,
    TriageFormulationOutput,
    ProductCategoryEnum
)

@pytest.fixture
def engine():
    return FormulationTriageEngine()

def test_pure_classical_formulation_triage(engine):
    """Classical text formula: Barred under §3(p), Exempt under BD Act §7(1) Proviso for Vaidyas."""
    inp = TriageFormulationInput(
        product_name="Maha Triphala Ghrita",
        is_first_schedule_text=True,
        extraction_method="traditional_aqueous_decoction",
        delivery_format="traditional_ghrita_oil",
        has_comparative_synergy_data=False,
        is_registered_ayush_practitioner=True,
        intended_use="therapeutic_internal"
    )
    res = engine.evaluate(inp)
    
    assert res.category == ProductCategoryEnum.CLASSICAL_AYURVEDIC
    assert res.patent_status == "BARRED"
    assert "Section 3(p)" in res.patent_rationale
    assert res.abs_status == "EXEMPTED"
    assert "Section 7(1)" in res.abs_rationale
    assert "First Schedule" in res.licensing_framework

def test_proprietary_formulation_with_synergy(engine):
    """Proprietary with verified lab synergy: Eligible under §3(e), SBB Section 7 intimation required."""
    inp = TriageFormulationInput(
        product_name="Ashwa-Shallaki Synergy Gel",
        is_first_schedule_text=False,
        extraction_method="standardized_hydroalcoholic_extract",
        delivery_format="topical_emulgel",
        has_comparative_synergy_data=True,
        synergy_percentage_increase=45.0,
        is_registered_ayush_practitioner=False,
        intended_use="therapeutic_topical"
    )
    res = engine.evaluate(inp)
    
    assert res.category == ProductCategoryEnum.PATENT_PROPRIETARY
    assert res.patent_status == "CONDITIONALLY_ELIGIBLE"
    assert "Section 3(e)" in res.patent_rationale
    assert res.abs_status == "ACTION_REQUIRED"
    assert "Section 7" in res.abs_rationale
    assert "Rule 158B" in res.licensing_framework

def test_proprietary_formulation_without_synergy(engine):
    """Proprietary mixture without trial proof: Barred under §3(e) mere admixture."""
    inp = TriageFormulationInput(
        product_name="Herbal Vitality Capsules",
        is_first_schedule_text=False,
        extraction_method="traditional_aqueous_decoction",
        delivery_format="hard_gelatin_capsule",
        has_comparative_synergy_data=False,
        is_registered_ayush_practitioner=False,
        intended_use="therapeutic_internal"
    )
    res = engine.evaluate(inp)
    
    assert res.category == ProductCategoryEnum.PATENT_PROPRIETARY
    assert res.patent_status == "BARRED"
    assert "Section 3(e)" in res.patent_rationale
    assert "admixture" in res.patent_rationale.lower()

def test_phytopharmaceutical_drug_triage(engine):
    """Purified fraction with organic solvent: Governed under CDSCO Phytopharmaceutical rules."""
    inp = TriageFormulationInput(
        product_name="Curcuminoid Fraction 95%",
        is_first_schedule_text=False,
        extraction_method="isolated_purified_phytochemical_fraction",
        delivery_format="nano_liposomal_emulsion",
        has_comparative_synergy_data=True,
        is_registered_ayush_practitioner=False,
        intended_use="therapeutic_internal"
    )
    res = engine.evaluate(inp)
    
    assert res.category == ProductCategoryEnum.PHYTOPHARMACEUTICAL
    assert res.patent_status == "HIGHLY_ELIGIBLE"
    assert "CDSCO" in res.licensing_framework
    assert "Form III" in res.abs_rationale

def test_ayurveda_aahar_food_supplement(engine):
    """Health food supplement without therapeutic claims: Regulated under FSSAI 2022."""
    inp = TriageFormulationInput(
        product_name="Ayurvedic Turmeric Latte Mix",
        is_first_schedule_text=True,
        extraction_method="traditional_powder",
        delivery_format="food_beverage_powder",
        has_comparative_synergy_data=False,
        is_registered_ayush_practitioner=False,
        intended_use="food_wellness_supplement"
    )
    res = engine.evaluate(inp)
    
    assert res.category == ProductCategoryEnum.AYURVEDA_AAHAR
    assert "FSSAI" in res.licensing_framework
    assert res.patent_status == "BARRED"
