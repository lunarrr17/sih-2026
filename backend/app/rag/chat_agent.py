import logging
from typing import Dict, Any, List, Optional, TypedDict
from pydantic import BaseModel, Field

from backend.app.rag.retriever import retriever
from backend.app.rag.generator import GroundedLLMSynthesizer
from backend.app.rag.schemas import GroundedClaim, CitationItem, EvidenceStrength, EvidenceRecord, ClaimRecord
from backend.app.core.guardrails import GuardrailsEngine
from backend.app.engines.triage_engine import (
    triage_engine,
    TriageFormulationInput,
    TriageFormulationOutput,
    ProductCategoryEnum
)

logger = logging.getLogger(__name__)

class ChatAgentRequest(BaseModel):
    query: str
    jurisdiction: str = "national"
    session_id: str = "default_session"
    classification_context: Optional[Dict[str, Any]] = None
    triage_input: Optional[TriageFormulationInput] = None

class ChatAgentResponse(BaseModel):
    query: str
    jurisdiction: str
    session_id: str
    answer: str
    confidence_score: float
    evidence_strength: str = EvidenceStrength.INSUFFICIENT.value
    is_grounded: bool
    is_safe: bool
    abstain: bool
    partial_support: bool = False
    claims: List[GroundedClaim] = []
    claim_records: List[ClaimRecord] = []
    citations: List[CitationItem] = []
    evidence_records: List[EvidenceRecord] = []
    triage_result: Optional[TriageFormulationOutput] = None
    escalation_available: bool = True

class LegalChatAgent:
    """
    LangGraph-based stateful Chat Agent for Ayurvedic IPR & Regulatory Guidance.
    Manages multi-turn conversation state, interactive 5-step formulation triage,
    sliced hybrid retrieval, query decomposition, and grounded synthesis.
    """

    def __init__(self):
        self.session_memory: Dict[str, List[Dict[str, str]]] = {}

    def process_message(self, request: ChatAgentRequest) -> ChatAgentResponse:
        """Executes the complete state machine workflow for an incoming user message."""
        session_id = request.session_id
        if session_id not in self.session_memory:
            self.session_memory[session_id] = []

        # Step 1: Safety & Out-of-Scope Check on raw incoming query (Prevents history poisoning)
        safety_result = GuardrailsEngine.check_query_safety(request.query)
        if not safety_result["is_safe"]:
            out_of_scope_ans = f"⚠️ **Out of Scope Request**: {safety_result['reason']}\n\n💡 {safety_result['suggestion']}"
            return ChatAgentResponse(
                query=request.query,
                jurisdiction=request.jurisdiction,
                session_id=session_id,
                answer=out_of_scope_ans,
                confidence_score=0.0,
                evidence_strength=EvidenceStrength.INSUFFICIENT.value,
                is_grounded=False,
                is_safe=False,
                abstain=True,
                partial_support=False,
                claims=[],
                citations=[],
                escalation_available=False
            )

        # Step 2: Check if explicit Triage Input was provided or requested
        triage_output: Optional[TriageFormulationOutput] = None
        if request.triage_input:
            triage_output = triage_engine.evaluate(request.triage_input)
            request.classification_context = triage_output.dict()

        # Step 3: Contextual Query Resolution (Only for genuine follow-up / anaphora)
        effective_query = request.query
        history = self.session_memory[session_id]
        q_words = request.query.lower().split()
        q_lower = request.query.lower()

        # Genuine follow-up check: short queries or queries starting with anaphoric pronouns
        starts_follow_up = any(q_lower.startswith(p) for p in ["what about", "how about", "can it", "is it", "does it", "can that", "is that", "does this"])
        has_standalone_domain = any(term in q_lower for term in ["patents act", "drugs and cosmetics", "biological diversity", "fssai", "nagoya", "wipo", "trips", "section 3", "rule 161", "rule 158b"])
        is_short_follow_up = len(q_words) <= 5 and any(w in q_words for w in ["it", "this", "that", "these", "those", "same"])

        is_follow_up = (starts_follow_up or is_short_follow_up) and not has_standalone_domain
        if history and is_follow_up:
            last_user_msg = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
            if last_user_msg:
                effective_query = f"{last_user_msg} {request.query}"

        # Step 4: Hybrid Retrieval with Jurisdiction Routing & Decomposition
        target_jur = request.jurisdiction.lower()
        q_lower = request.query.lower()

        # Dual domestic + international regime detection (Item 6):
        # Queries requiring both Indian statutory regime and international treaty postures
        # must route explicitly through comparative retrieval to keep evidence packs separated.
        has_intl_need = any(w in q_lower for w in ["nagoya", "cbd", "trips", "wipo", "gratk", "international treaty", "international regime", "global treaties"])
        has_nat_need = any(w in q_lower for w in ["india", "indian", "domestic", "biological diversity", "patents act", "drugs and cosmetics", "section 3", "rule 161", "sbb", "nba"])
        is_explicit_comparison = (
            any(c in q_lower for c in ["both national and international", "compare national and international", "compare indian patent", "comparison between", "comparative analysis"])
            or (has_intl_need and has_nat_need and any(c in q_lower for c in ["compare", "comparison", "comparative", "contrast", "versus", " vs ", "differ", "difference", "with the"]))
            or (has_intl_need and any(c in q_lower for c in ["compare", "comparison", "comparative", "contrast", "versus", " vs ", "differ", "difference"]))
        )
        is_india_nagoya_abs = ("nagoya" in q_lower and any(w in q_lower for w in ["india", "indian"]) and any(b in q_lower for b in ["access and benefit", "benefit sharing", "abs", "position"]))

        if target_jur in ["comparative", "both"] or is_explicit_comparison or is_india_nagoya_abs:
            retrieval_results = retriever.search(
                query=effective_query,
                jurisdiction="comparative",
                top_k=4,
                enable_reranking=True
            )
            decomp = getattr(retriever, "last_decomposition", {})
            unsupported_dims = decomp.get("unsupported_dimensions", []) if decomp.get("is_decomposed") else []
            synth_res = GroundedLLMSynthesizer.synthesize_comparative(
                query=request.query,
                national_chunks=retrieval_results["national"],
                intl_chunks=retrieval_results["international"],
                classification_context=request.classification_context,
                unsupported_dimensions=unsupported_dims
            )
            raw_answer, citations = synth_res[0], synth_res[1]
            claims = getattr(synth_res, "claims", [])
            partial_support = getattr(synth_res, "partial_support", False)
            all_chunks = retrieval_results["national"] + retrieval_results["international"]
        else:
            effective_jur = "national" if target_jur in ["national", "india"] else "international"
            retrieval_results = retriever.search(
                query=effective_query,
                jurisdiction=effective_jur,
                top_k=4,
                enable_reranking=True
            )
            decomp = getattr(retriever, "last_decomposition", {})
            unsupported_dims = decomp.get("unsupported_dimensions", []) if decomp.get("is_decomposed") else []
            synth_res = GroundedLLMSynthesizer.synthesize(
                query=request.query,
                chunks=retrieval_results,
                jurisdiction=effective_jur,
                classification_context=request.classification_context,
                unsupported_dimensions=unsupported_dims
            )
            raw_answer, citations = synth_res[0], synth_res[1]
            claims = getattr(synth_res, "claims", [])
            partial_support = getattr(synth_res, "partial_support", False)
            all_chunks = retrieval_results

        # Step 5: Grounding & Confidence Evaluation
        cited_ref_ids = [c.ref_id for c in citations if getattr(c, 'ref_id', None)]
        grounding_eval = GuardrailsEngine.evaluate_grounding(raw_answer, all_chunks, cited_ref_ids)
        evidence_strength = GuardrailsEngine.audit_evidence_strength(
            claims,
            grounding_eval["is_grounded"],
            grounding_eval["confidence_score"]
        )
        final_answer = GuardrailsEngine.inject_disclaimer(raw_answer)

        # Step 6: Update Session Memory (Bounded to last 10 messages)
        self.session_memory[session_id].append({"role": "user", "content": request.query})
        self.session_memory[session_id].append({"role": "assistant", "content": final_answer})
        if len(self.session_memory[session_id]) > 10:
            self.session_memory[session_id] = self.session_memory[session_id][-10:]

        evidence_records = getattr(synth_res, "evidence_records", [])
        claim_records = getattr(synth_res, "claim_records", [])

        return ChatAgentResponse(
            query=request.query,
            jurisdiction=request.jurisdiction,
            session_id=session_id,
            answer=final_answer,
            confidence_score=grounding_eval["confidence_score"],
            evidence_strength=evidence_strength,
            is_grounded=grounding_eval["is_grounded"],
            is_safe=True,
            abstain=grounding_eval["abstain"],
            partial_support=partial_support,
            claims=claims,
            claim_records=claim_records,
            citations=citations,
            evidence_records=evidence_records,
            triage_result=triage_output,
            escalation_available=True
        )

# Global chat agent instance
chat_agent = LegalChatAgent()
