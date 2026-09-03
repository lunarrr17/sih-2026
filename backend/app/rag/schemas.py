from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

class AuthorityLevel(str, Enum):
    PRIMARY_STATUTE = "primary_statute"
    SUBORDINATE_REGULATION = "subordinate_regulation"
    INTERNATIONAL_TREATY = "international_treaty"
    OFFICIAL_GUIDELINE = "official_guideline"
    SECONDARY_ACADEMIC_STUDY = "secondary_academic_study"

class LegalStatus(str, Enum):
    IN_FORCE = "in_force"
    AMENDED = "amended"
    ADOPTED = "adopted"
    RATIFIED = "ratified"
    REPEALED = "repealed"
    UNKNOWN = "unknown"

class EvidenceStrength(str, Enum):
    STRONG = "Strong Evidence"
    MODERATE = "Moderate Evidence"
    INSUFFICIENT = "Insufficient Evidence"

class VerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"

class DetailedLegalStatus(BaseModel):
    authority_level: str = AuthorityLevel.PRIMARY_STATUTE.value
    canonical_status: str = LegalStatus.IN_FORCE.value
    enacted_date: Optional[str] = None
    effective_date: Optional[str] = None
    amended_date: Optional[str] = None
    adopted_date: Optional[str] = None
    signature_date: Optional[str] = None
    ratified_date: Optional[str] = None
    global_entry_into_force_date: Optional[str] = None
    entry_into_force_date: Optional[str] = None
    entry_into_force_for_india_date: Optional[str] = None
    binding_on_jurisdiction: Optional[str] = None
    status_source: Optional[str] = None
    status_verified_at: Optional[str] = None

class StatutoryMetadata(BaseModel):
    """
    Structured metadata associated with an ingested statutory chunk.
    """
    document_name: str
    statute_title: str
    section_or_clause: str = "General Provision"
    jurisdiction: str = Field(description="'national' or 'international'")
    category: str = "classical"
    source_type: str = "primary_statute"  # primary_statute, subordinate_regulation, international_treaty, secondary_academic_study
    authority_level: str = "primary_statute"
    legal_status: str = "in_force"
    binding_on_jurisdiction: Optional[str] = None
    detailed_legal_status: Optional[DetailedLegalStatus] = None
    page_numbers: List[int] = []
    official_source_url: str = "https://ipindia.gov.in"
    is_statutory_bar: bool = False

class CitationItem(BaseModel):
    statute: str
    section: str
    title: str
    source_url: str
    page_numbers: List[int] = []
    ref_id: Optional[str] = None
    document_name: Optional[str] = None
    source_type: str = "primary_statute"
    authority_level: str = "primary_statute"
    legal_status: str = "in_force"
    binding_on_jurisdiction: Optional[str] = None
    detailed_legal_status: Optional[DetailedLegalStatus] = None

class AtomicProposition(BaseModel):
    """
    Atomic proposition within a compound legal claim.
    Distinguishes statutory rules from applicant product factual classifications.
    """
    proposition_text: str
    proposition_type: str = "statutory_rule"  # statutory_rule, factual_premise, application_conclusion, temporal_status
    assigned_evidence_ids: List[str] = []
    status: str = VerificationStatus.UNSUPPORTED.value
    unsupported_qualifiers: List[str] = []
    missing_premise_type: Optional[str] = None  # material_missing_premise, non_material_missing_detail, None
    notes: str = ""

class ClaimVerificationResult(BaseModel):
    claim_text: str
    status: str = VerificationStatus.UNSUPPORTED.value
    assigned_evidence_ids: List[str] = []
    valid_evidence_ids: List[str] = []
    overlap_score: float = 0.0
    concept_alignment_score: float = 0.0
    notes: str = ""
    # Rigorous diagnostic separation (Section 3)
    retrieval_relevance: float = 0.0
    evidence_relevance: float = 0.0
    claim_entailment: str = VerificationStatus.UNSUPPORTED.value
    legal_conclusion_confidence: float = 0.0
    propositions: List[AtomicProposition] = []
    unsupported_qualifiers: List[str] = []
    has_material_missing_premise: bool = False

class GroundedClaim(BaseModel):
    """
    Intermediate claim-level evidence contract.
    Each substantive legal proposition is tied to explicit evidence IDs and server-verified.
    """
    claim_text: str
    evidence_ids: List[str] = []
    verification_status: str = VerificationStatus.UNSUPPORTED.value
    evidence_strength: str = EvidenceStrength.INSUFFICIENT.value
    verification_notes: Optional[str] = None
    citations: List[CitationItem] = []
    # Proposition and qualifier modeling
    retrieval_relevance: float = 0.0
    evidence_relevance: float = 0.0
    claim_entailment: str = VerificationStatus.UNSUPPORTED.value
    legal_conclusion_confidence: float = 0.0
    propositions: List[AtomicProposition] = []
    unsupported_qualifiers: List[str] = []
    has_material_missing_premise: bool = False

class EvidenceRecord(BaseModel):
    """
    Authoritative, structured internal representation of an accepted piece of legal evidence.
    Provides complete end-to-end traceability from chunk text to source PDF page.
    """
    evidence_id: str  # e.g. "REF-1"
    chunk_id: str
    document_name: str
    statute_title: str
    jurisdiction: str  # "national" or "international"
    source_type: str = "primary_statute"
    authority_level: str = "primary_statute"
    page_numbers: List[int] = []
    section_or_clause: str = "General Provision"
    chunk_text: str
    retrieval_score: float = 0.0
    rerank_score: Optional[float] = None
    acceptance_reason: str = "Relevance gate passed"
    official_source_url: str = "https://ipindia.gov.in"
    is_statutory_bar: bool = False

class ClaimRecord(BaseModel):
    """
    Substantive legal claim generated by the pipeline, linked to exact supporting evidence.
    """
    claim_id: str  # e.g. "CLAIM-1"
    claim_text: str
    evidence_ids: List[str] = []  # e.g. ["REF-1"]
    support_status: str = VerificationStatus.UNSUPPORTED.value  # SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, CONTRADICTED
    support_strength: str = EvidenceStrength.INSUFFICIENT.value  # Strong Evidence, Moderate Evidence, Insufficient Evidence
    legal_scope: str = "statutory_rule"  # statutory_rule, statutory_exclusion, affirmative_patentability, regulatory_mandate, factual_premise
    verification_notes: Optional[str] = None
    unsupported_qualifiers: List[str] = []

class LegalChunk(BaseModel):
    """
    Unit of legal text ingested from official PDF documents,
    containing text content, identifier, and structured metadata.
    """
    chunk_id: str
    text: str
    metadata: StatutoryMetadata
    embedding: Optional[List[float]] = None

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Converts the chunk to a Qdrant point payload dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document_name": self.metadata.document_name,
            "statute_title": self.metadata.statute_title,
            "section_or_clause": self.metadata.section_or_clause,
            "jurisdiction": self.metadata.jurisdiction,
            "category": self.metadata.category,
            "source_type": self.metadata.source_type,
            "authority_level": getattr(self.metadata, "authority_level", self.metadata.source_type),
            "legal_status": getattr(self.metadata, "legal_status", "in_force"),
            "binding_on_jurisdiction": getattr(self.metadata, "binding_on_jurisdiction", None),
            "page_numbers": self.metadata.page_numbers,
            "official_source_url": self.metadata.official_source_url,
            "is_statutory_bar": self.metadata.is_statutory_bar,
        }

# =========================================================================
# PHASE 4: FORMULATION INTELLIGENCE & CLASSIFICATION SCHEMAS
# =========================================================================

class SubjectType(str, Enum):
    """
    Topical inquiry category identified from linguistic cues in the user query.
    NOTE: Represents query focus for retrieval routing; does NOT establish statutory status.
    """
    CLASSICAL_FORMULATION = "classical_formulation"
    PROPRIETARY_FORMULATION = "proprietary_formulation"
    MODIFIED_FORMULATION = "modified_formulation"
    UNSPECIFIED_FORMULATION = "unspecified_formulation"
    SUBSTANCE_INGREDIENT = "substance_ingredient"
    PROCESS_METHOD = "process_method"
    REGULATORY_PRODUCT = "regulatory_product"
    GENERAL_INQUIRY = "general_inquiry"
    UNKNOWN = "unknown"

class SubstanceOrigin(str, Enum):
    BOTANICAL = "botanical"
    ANIMAL_DERIVED = "animal_derived"
    MINERAL_HERBO_MINERAL = "mineral_herbo_mineral"
    MICROBIAL = "microbial"
    SYNTHETIC_CHEMICAL = "synthetic_chemical"
    UNKNOWN_MIXED = "unknown_mixed"

class ProcessType(str, Enum):
    KNOWN_TRADITIONAL_PROCESS = "known_traditional_process"
    MODIFIED_PROCESS = "modified_process"
    POTENTIALLY_NOVEL_PROCESS = "potentially_novel_process"
    UNSPECIFIED_PROCESS = "unspecified_process"

class TraditionalKnowledgeSignal(str, Enum):
    EXPLICIT_TRADITIONAL = "explicit_traditional"
    INFERRED_TRADITIONAL = "inferred_traditional"
    UNCLEAR_OR_MIXED = "unclear_or_mixed"
    NO_TK_SIGNAL = "no_tk_signal"

class UserIntent(str, Enum):
    PATENTABILITY_INQUIRY = "patentability_inquiry"
    PRIOR_ART_TK_CONCERN = "prior_art_tk_concern"
    REGULATORY_LICENSING = "regulatory_licensing"
    BIODIVERSITY_ABS = "biodiversity_abs"
    PROCEDURAL_FILING = "procedural_filing"
    COMPARATIVE_CROSS_REGIME = "comparative_cross_regime"
    GENERAL_INFORMATIONAL = "general_informational"
    UNKNOWN = "unknown"

class ConfidenceTier(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"

class RoutingHints(BaseModel):
    """
    Downstream routing hints derived from linguistic signals in the query.
    NOTE: These are retrieval hints to focus statutory evidence search, NOT legal conclusions.
    """
    retrieve_patent_exclusions: bool = False
    retrieve_patentability_criteria: bool = False
    retrieve_traditional_knowledge: bool = False
    retrieve_biodiversity_abs: bool = False
    retrieve_regulatory_licensing: bool = False
    retrieve_food_safety_aahar: bool = False
    retrieve_process_standards: bool = False
    jurisdictions_suggested: List[str] = ["national"]
    focus_terms: List[str] = []

class FormulationIntelligence(BaseModel):
    """
    Structured, evidence-aware formulation and substance intelligence model.
    Captures extracted query entities, linguistic signals, inquiry taxonomy,
    explicit uncertainty, ambiguities, missing information, and retrieval routing hints.

    CRITICAL ARCHITECTURAL INVARIANT:
    This metadata represents query understanding to focus downstream evidence retrieval.
    It NEVER constitutes statutory proof, evidentiary authority, or a legal conclusion
    (e.g., patentability, exclusion under Section 3, or regulatory approval).
    """
    query_text: str
    normalized_text: str
    subject_type: SubjectType = SubjectType.UNKNOWN
    subject_confidence: ConfidenceTier = ConfidenceTier.UNKNOWN
    formulation_name: Optional[str] = None
    alternative_names: List[str] = []
    ingredients: List[str] = []
    ingredient_count: int = 0
    substance_origin: SubstanceOrigin = SubstanceOrigin.UNKNOWN_MIXED
    dosage_form: Optional[str] = None
    preparation_method: Optional[str] = None
    process_type: ProcessType = ProcessType.UNSPECIFIED_PROCESS
    novel_process_signal: bool = False
    traditional_knowledge_signal: TraditionalKnowledgeSignal = TraditionalKnowledgeSignal.NO_TK_SIGNAL
    user_intents: List[UserIntent] = []
    plant_origin_signal: bool = False
    animal_origin_signal: bool = False
    mineral_origin_signal: bool = False
    microbial_origin_signal: bool = False
    synthetic_chemical_signal: bool = False
    food_or_ayush_product_signal: bool = False
    patent_inquiry_signal: bool = False
    regulatory_inquiry_signal: bool = False
    biodiversity_signal: bool = False
    abs_signal: bool = False
    jurisdictions_relevant: List[str] = []
    classification_reasons: List[str] = []
    ambiguities: List[str] = []
    missing_information: List[str] = []
    routing_hints: RoutingHints = Field(default_factory=RoutingHints)
    overall_confidence: ConfidenceTier = ConfidenceTier.UNKNOWN
    confidence_score: float = 0.0
    classifier_version: str = "v1.0-deterministic"
