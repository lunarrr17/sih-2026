import logging
from typing import Dict, Any, List, Optional, TypedDict
from pydantic import BaseModel, Field

from backend.app.rag.retriever import retriever
from backend.app.rag.generator import GroundedLLMSynthesizer, CitationItem
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
    is_grounded: bool
    is_safe: bool
    abstain: bool
    citations: List[CitationItem] = []
    triage_result: Optional[TriageFormulationOutput] = None
    escalation_available: bool = True

class LegalChatAgent:
    """
    LangGraph-based stateful Chat Agent for Ayurvedic IPR & Regulatory Guidance.
    Manages multi-turn conversation state, interactive 5-step formulation triage,
    sliced hybrid retrieval, and grounded synthesis.
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
                is_grounded=False,
                is_safe=False,
                abstain=True,
                citations=[],
                escalation_available=False
            )

        # Step 2: Check if explicit Triage Input was provided or requested
        triage_output: Optional[TriageFormulationOutput] = None
        if request.triage_input:
            triage_output = triage_engine.evaluate(request.triage_input)
            request.classification_context = triage_output.dict()

        # Step 3: Contextual Query Resolution (Multi-turn session memory for retrieval)
        effective_query = request.query
        history = self.session_memory[session_id]
        if history:
            last_user_msg = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
            if last_user_msg and len(request.query.split()) < 10:
                effective_query = f"{last_user_msg} {request.query}"

        # Step 4: Hybrid Retrieval with Jurisdiction Routing
        target_jur = request.jurisdiction.lower()

        if target_jur in ["comparative", "both"]:
            retrieval_results = retriever.search(
                query=effective_query,
                jurisdiction="comparative",
                top_k=4,
                enable_reranking=True
            )
            raw_answer, citations = GroundedLLMSynthesizer.synthesize_comparative(
                query=request.query,
                national_chunks=retrieval_results["national"],
                intl_chunks=retrieval_results["international"],
                classification_context=request.classification_context
            )
            all_chunks = retrieval_results["national"] + retrieval_results["international"]
        else:
            effective_jur = "national" if target_jur in ["national", "india"] else "international"
            retrieval_results = retriever.search(
                query=effective_query,
                jurisdiction=effective_jur,
                top_k=4,
                enable_reranking=True
            )
            raw_answer, citations = GroundedLLMSynthesizer.synthesize(
                query=request.query,
                chunks=retrieval_results,
                jurisdiction=effective_jur,
                classification_context=request.classification_context
            )
            all_chunks = retrieval_results

        # Step 5: Grounding & Confidence Evaluation
        grounding_eval = GuardrailsEngine.evaluate_grounding(raw_answer, all_chunks)
        final_answer = GuardrailsEngine.inject_disclaimer(raw_answer)

        # Step 6: Update Session Memory
        self.session_memory[session_id].append({"role": "user", "content": request.query})
        self.session_memory[session_id].append({"role": "assistant", "content": final_answer})

        return ChatAgentResponse(
            query=request.query,
            jurisdiction=request.jurisdiction,
            session_id=session_id,
            answer=final_answer,
            confidence_score=grounding_eval["confidence_score"],
            is_grounded=grounding_eval["is_grounded"],
            is_safe=True,
            abstain=grounding_eval["abstain"],
            citations=citations,
            triage_result=triage_output,
            escalation_available=True
        )

# Global chat agent instance
chat_agent = LegalChatAgent()
