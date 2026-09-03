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
