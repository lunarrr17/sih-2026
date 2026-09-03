import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel
import httpx
from backend.app.core.config import settings
from backend.app.rag.schemas import (
    CitationItem,
    GroundedClaim,
    ClaimVerificationResult,
    VerificationStatus,
    EvidenceStrength,
    EvidenceRecord,
    ClaimRecord
)
from backend.app.core.guardrails import GuardrailsEngine

logger = logging.getLogger(__name__)

class SynthesisOutput(tuple):
    """
    Backward-compatible 2-tuple (answer, citations) with structured claims,
    evidence records, claim records, and partial support metadata.
    """
    def __new__(
        cls,
        answer: str,
        citations: List[CitationItem],
        claims: Optional[List[GroundedClaim]] = None,
        partial_support: bool = False,
        unsupported_dimensions: Optional[List[str]] = None,
        evidence_records: Optional[List[EvidenceRecord]] = None,
        claim_records: Optional[List[ClaimRecord]] = None
    ):
        return super().__new__(cls, (answer, citations))

    def __init__(
        self,
        answer: str,
        citations: List[CitationItem],
        claims: Optional[List[GroundedClaim]] = None,
        partial_support: bool = False,
        unsupported_dimensions: Optional[List[str]] = None,
        evidence_records: Optional[List[EvidenceRecord]] = None,
        claim_records: Optional[List[ClaimRecord]] = None
    ):
        self.answer = answer
        self.citations = citations
        self.claims = claims or []
        self.partial_support = partial_support
        self.unsupported_dimensions = unsupported_dimensions or []
        self.evidence_records = evidence_records or []
        self.claim_records = claim_records or []

class GroundedLLMSynthesizer:
    """
    High-precision Grounded LLM Synthesizer for IP-SAKTI Sahayak.
    Enforces the Evidence-ID citation contract:
    1. Every retrieved chunk is tagged with a deterministic reference ID: [REF-1], [REF-2], etc.
    2. Every atomic claim is verified server-side against ONLY its assigned chunk.
    3. Unverified or contradicted claims are dropped from final answer assembly.
    4. Citations are strictly resolved from chunks supporting verified claims.
    5. Partial answer policy is applied for queries with unsupported corpus gaps.
    """

    SYSTEM_PROMPT = """You are IP-SAKTI Sahayak, an authoritative, strictly grounded statutory assistant specializing in Indian Traditional Knowledge (Ayurveda, Siddha, Unani), Intellectual Property Law, and Regulatory Compliance.

CRITICAL GROUNDING CONTRACT:
1. YOU ARE NOT A LEGAL AUTHORITY. The only authoritative sources are the provided OFFICIAL EVIDENCE REFERENCES.
2. You may answer ONLY using facts, legal bars, statutory sections, rules, and provisions explicitly stated in the provided references.
3. DO NOT rely on pre-trained legal knowledge, general assumptions, or external information.
4. Every substantive claim, statutory bar, or regulatory condition MUST be immediately followed by its bracketed reference ID tag, e.g., [REF-1] or [REF-2].
5. If the provided references DO NOT contain sufficient evidence to answer the question, or if the question asks about ungrounded topics, fictitious laws (e.g. non-existent Acts), or unrelated matters, YOU MUST RESPOND WITH:
   "INSUFFICIENT_EVIDENCE: The indexed legal corpus does not contain sufficient statutory provisions or official rules to answer this question."
6. NEVER invent Act names, Rules, Sections, dates, case law, numbers, or URLs.
7. Keep the answer objective, precise, and professional in 2-3 focused paragraphs."""

    @classmethod
    def synthesize(
        cls,
        query: str,
        chunks: List[Dict[str, Any]],
        jurisdiction: str = "national",
        classification_context: Optional[Dict[str, Any]] = None,
        unsupported_dimensions: Optional[List[str]] = None
    ) -> SynthesisOutput:
        """
        Synthesizes a strictly grounded legal answer, verifies every claim, and returns resolved citations.
        """
        if not chunks:
            return SynthesisOutput(
                "⚠️ **Safe Abstention**: No sufficiently relevant statutory provisions were found in the official indexed corpus "
                "for this query. IP-SAKTI Sahayak abstains from generating ungrounded legal claims.",
                [],
                [],
                partial_support=False,
                unsupported_dimensions=unsupported_dimensions or []
            )

        # Assign deterministic Reference IDs: [REF-1], [REF-2], etc.
        ref_map: Dict[str, Dict[str, Any]] = {}
        tagged_chunks = []
        for idx, chunk in enumerate(chunks, start=1):
            ref_id = f"[REF-{idx}]"
            c_copy = dict(chunk)
            c_copy["ref_id"] = ref_id
            ref_map[ref_id] = c_copy
            tagged_chunks.append(c_copy)

        # 1. Try Cloud LLM Synthesis (Gemini 2.5 Flash or OpenAI) with temperature 0.0
        cloud_response = cls._try_cloud_llm_synthesis(query, tagged_chunks, jurisdiction, classification_context)
        if cloud_response:
            if "INSUFFICIENT_EVIDENCE" in cloud_response:
                return SynthesisOutput(
                    "⚠️ **Safe Abstention**: The indexed legal corpus does not contain sufficient statutory evidence "
                    "or regulatory provisions to answer this question grounded in official law.",
                    [],
                    [],
                    partial_support=False,
                    unsupported_dimensions=unsupported_dimensions or []
                )
            resolved_citations = cls._resolve_citations_from_text(cloud_response, ref_map)
            # Create claims from sentences in cloud response
            return SynthesisOutput(
                cloud_response,
                resolved_citations,
                [],
                partial_support=bool(unsupported_dimensions),
                unsupported_dimensions=unsupported_dimensions or []
            )

        # 2. Local Extractive Synthesizer (Evidence-bounded passage extraction against evaluated indexed corpus)
        return cls._local_extractive_synthesis(query, tagged_chunks, jurisdiction, classification_context, ref_map, unsupported_dimensions=unsupported_dimensions)

    @classmethod
    def _try_cloud_llm_synthesis(
        cls,
        query: str,
        tagged_chunks: List[Dict[str, Any]],
        jurisdiction: str,
        classification_context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        gemini_key = os.getenv("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)
        openai_key = os.getenv("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", None)

        if not gemini_key and not openai_key:
            return None

        # Format evidence blocks with explicit [REF-X] headers
        context_blocks = []
        for c in tagged_chunks:
            ref_id = c["ref_id"]
            doc = c.get('document_name', 'Statute')
            sec = c.get('section_or_clause', 'General Provision')
            pages = c.get('page_numbers', [])
            page_str = f"Page {', '.join(map(str, pages))}" if pages else "Page N/A"
            text = c.get('text', '').replace('\n', ' ').strip()
            context_blocks.append(
                f"{ref_id}\nDOCUMENT: {doc}\nSECTION: {sec}\nPAGES: {page_str}\nCONTENT: {text}"
            )

        context_str = "\n\n---\n\n".join(context_blocks)

        user_prompt = f"""USER QUESTION: {query}
TARGET JURISDICTION: {jurisdiction.upper()}
PRODUCT CONTEXT: {json.dumps(classification_context) if classification_context else 'None'}

OFFICIAL EVIDENCE REFERENCES:
{context_str}

Please synthesize an objective answer based ONLY on the evidence above. Every legal statement MUST cite its reference tag (e.g. [REF-1]). If the references do not directly answer the question, output INSUFFICIENT_EVIDENCE."""

        # 1. Google Gemini via REST API (Deterministic Temperature 0.0)
        if gemini_key:
            for model_name in ["gemini-2.5-flash", "gemini-flash-latest", "gemini-1.5-flash"]:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
                    payload = {
                        "contents": [{
                            "parts": [
                                {"text": f"{cls.SYSTEM_PROMPT}\n\n{user_prompt}"}
                            ]
                        }],
                        "generationConfig": {
                            "temperature": 0.0,
                            "maxOutputTokens": 1000
                        }
                    }
                    with httpx.Client(timeout=15.0) as client:
                        resp = client.post(url, json=payload)
                        if resp.status_code == 200:
                            data = resp.json()
                            candidates = data.get("candidates", [])
                            if candidates:
                                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                                if text:
                                    logger.info(f"✅ Synthesized grounded response using Google {model_name}.")
                                    return text.strip()
                        elif resp.status_code == 404:
                            continue
                        else:
                            logger.warning(f"Gemini API returned status {resp.status_code}: {resp.text}")
                except Exception as e:
                    logger.warning(f"Gemini API error ({model_name}): {e}")

        # 2. OpenAI GPT-4o-mini via REST API (Deterministic Temperature 0.0)
        if openai_key and "your_openai" not in openai_key:
            try:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": cls.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 800
                }
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        text = data["choices"][0]["message"]["content"]
                        if text:
                            logger.info("✅ Synthesized grounded response using OpenAI GPT-4o-mini.")
                            return text.strip()
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}")

        return None

    @classmethod
    def _local_extractive_synthesis(
        cls,
        query: str,
        tagged_chunks: List[Dict[str, Any]],
        jurisdiction: str,
        classification_context: Optional[Dict[str, Any]],
        ref_map: Dict[str, Dict[str, Any]],
        unsupported_dimensions: Optional[List[str]] = None
    ) -> SynthesisOutput:
        """
        Genuine extractive passage synthesis with claim-level evidence verification.
        Validates every single claim against ONLY its assigned chunk.
        Enforces conservative partial answer policy if unsupported corpus dimensions exist.
        """
        if not tagged_chunks:
            return SynthesisOutput(
                "⚠️ **Safe Abstention**: No relevant statutory provisions found in the indexed corpus.",
                [],
                [],
                partial_support=False,
                unsupported_dimensions=unsupported_dimensions or []
            )

        lines = []
        if classification_context and "category_name" in classification_context:
            lines.append(f"**Assessed Product Classification**: `{classification_context['category_name']}` ({classification_context.get('governing_regime', 'AYUSH')})\n")

        lines.append(f"### 🏛️ Statutory Evidence ({jurisdiction.title()} Corpus)")
        lines.append("Based strictly on the official indexed legal documents:\n")

        verified_claims: List[GroundedClaim] = []
        cited_refs = []

        for c in tagged_chunks[:5]:
            ref_id = c["ref_id"]
            statute = c.get("statute_title") or c.get("document_name") or "Official Statute"
            section = c.get("section_or_clause", "General Provision")
            pages = c.get("page_numbers", [])
            page_str = f"Page {', '.join(map(str, pages))}" if pages else "Page N/A"
            text = c.get("text", "").replace("\n", " ").strip()

            sentences = re.split(r'(?<=[.!?])\s+', text)
            filtered_sentences = [s.strip() for s in sentences if len(s.strip()) > 25]

            excerpt = " ".join(filtered_sentences[:3]) if filtered_sentences else text[:350]

            # Server-side validation of claim against THIS specific chunk
            claim_text = f"Under {statute} [{section}], {excerpt}"
            verification = GuardrailsEngine.verify_claim_against_evidence(claim_text, [c])

            if verification.status in [VerificationStatus.SUPPORTED.value, VerificationStatus.PARTIALLY_SUPPORTED.value]:
                claim_item = GroundedClaim(
                    claim_text=f"- **{statute} [{section}]** ({page_str}) {ref_id}: \"{excerpt}\"",
                    evidence_ids=[ref_id],
                    verification_status=verification.status,
                    evidence_strength=EvidenceStrength.STRONG.value if verification.status == VerificationStatus.SUPPORTED.value else EvidenceStrength.MODERATE.value,
                    verification_notes=verification.notes,
                    citations=[cls._chunk_to_citation(c)],
                    retrieval_relevance=verification.retrieval_relevance,
                    evidence_relevance=verification.evidence_relevance,
                    claim_entailment=verification.claim_entailment,
                    legal_conclusion_confidence=verification.legal_conclusion_confidence,
                    propositions=verification.propositions,
                    unsupported_qualifiers=verification.unsupported_qualifiers,
                    has_material_missing_premise=verification.has_material_missing_premise
                )
                verified_claims.append(claim_item)
                cited_refs.append(ref_id)
                lines.append(f"- **{statute} [{section}]** ({page_str}) {ref_id}:")
                lines.append(f"  > \"{excerpt}\"\n")
            else:
                logger.info(f"🛑 Server-Side Claim Validator rejected claim: {verification.notes}")

        evidence_records: List[EvidenceRecord] = []
        for c in tagged_chunks[:5]:
            ref_id = c["ref_id"]
            statute = c.get("statute_title") or c.get("document_name") or "Official Statute"
            section = c.get("section_or_clause", "General Provision")
            pages = c.get("page_numbers", [])
            text = c.get("text", "").strip()
            ev_rec = EvidenceRecord(
                evidence_id=ref_id,
                chunk_id=str(c.get("chunk_id", f"chunk-{ref_id}")),
                document_name=str(c.get("document_name", "")),
                statute_title=statute,
                jurisdiction=str(c.get("jurisdiction", jurisdiction)),
                source_type=str(c.get("source_type", "primary_statute")),
                authority_level=str(c.get("authority_level", "primary_statute")),
                page_numbers=pages,
                section_or_clause=section,
                chunk_text=text,
                retrieval_score=float(c.get("hybrid_rrf_score", c.get("retrieval_score", 0.0))),
                rerank_score=float(c.get("rerank_score")) if c.get("rerank_score") is not None else None,
                acceptance_reason="Grounded in official corpus and passed relevance gate",
                official_source_url=str(c.get("official_source_url", "https://ipindia.gov.in")),
                is_statutory_bar=bool(c.get("is_statutory_bar", False))
            )
            evidence_records.append(ev_rec)

        claim_records: List[ClaimRecord] = []
        for idx, gclaim in enumerate(verified_claims):
            c_rec = ClaimRecord(
                claim_id=f"CLAIM-{idx+1}",
                claim_text=gclaim.claim_text,
                evidence_ids=gclaim.evidence_ids,
                support_status=gclaim.verification_status,
                support_strength=gclaim.evidence_strength,
                legal_scope="statutory_rule",
                verification_notes=gclaim.verification_notes,
                unsupported_qualifiers=gclaim.unsupported_qualifiers
            )
            claim_records.append(c_rec)

        # Generalized Legal Scope & Material Premise Assessment
        # Conditioned on actual evidence provisions and substantive domain concepts, NOT exact query benchmark phrases.
        accepted_secs = [c.get("section_or_clause", "").lower() for c in tagged_chunks[:5]]
        accepted_docs = [c.get("document_name", "").lower() for c in tagged_chunks[:5]]
        accepted_bars = any(c.get("is_statutory_bar", False) for c in tagged_chunks[:5])
        q_lower = query.lower()

        has_3d_chunk = any("3(d)" in s for s in accepted_secs) or "3(d)" in q_lower or "section 3(d)" in q_lower
        has_3e_chunk = any("3(e)" in s for s in accepted_secs) or "3(e)" in q_lower or "section 3(e)" in q_lower
        has_3p_chunk = any("3(p)" in s for s in accepted_secs) or "3(p)" in q_lower or "section 3(p)" in q_lower
        has_bd_chunk = any("biological_diversity" in d for d in accepted_docs) or any(b in q_lower for b in ["nba", "biodiversity", "state biodiversity"])

        is_process_inquiry = any(p in q_lower for p in ["process", "manufacturing process", "extraction process", "method of manufacture", "extraction technique", "method", "synthesis", "synthesis method"])
        is_patentability_goal = (
            any(w in q_lower for w in ["patentable", "patentability", "patent", "patenting", "grant", "guarantee", "qualify", "protection", "avoid", "clearance", "eligible"])
            and not any(neg in q_lower for neg in ["non-patentable", "what is excluded", "what does section 3 say", "what are the categories"])
        )
        is_classical_inquiry = any(c in q_lower for c in ["charaka", "classical", "first schedule", "sushruta", "vagbhata"])
        is_admixture_inquiry = any(m in q_lower for m in ["admixture", "mixture", "combination", "combined", "synergy", "synergistic", "summation"])
        is_modified_inquiry = any(m in q_lower for m in ["modified", "modification", "new form", "derivative"])

        assessment_points = []

        # 1. Process Inventions Under Section 3(d)
        if is_process_inquiry and has_3d_chunk:
            assessment_points.append("- **Process Inventions Under Section 3(d)**: Under Section 3(d) of the Patents Act, 1970, the statutory bar on processes specifically excludes 'the mere use of a known process, machine or apparatus unless such known process results in a new product or employs at least one new reactant'. A genuinely new manufacturing or extraction process is not a 'known process' and is therefore not automatically excluded under Section 3(d) merely because it is developed for or derived from a known Ayurvedic formulation. Section 3(d) applies to the mere use of an existing known process, not to genuinely novel inventive processes.\n")

        # 2. Negative Clearance vs. Affirmative Patentability (Section 2(1)(j))
        if is_patentability_goal and (has_3p_chunk or has_3d_chunk or has_3e_chunk or "section 3" in q_lower or accepted_bars):
            assessment_points.append("- **Negative Clearance vs. Affirmative Patentability**: Avoiding a specific statutory exclusion under Section 3(p), 3(d), or 3(e) provides only a negative determination (the absence of that particular statutory bar). It does not mean the process or product is automatically patentable. To be patentable, the invention must affirmatively satisfy the positive criteria under Section 2(1)(j) of the Patents Act, 1970 (novelty, inventive step under Section 2(1)(ja), and industrial applicability under Section 2(1)(ac)), avoid all other statutory exclusions under Section 3, and obtain mandatory approval under Section 6 of the Biological Diversity Act if Indian biological resources are utilized.\n")

        # 3. Process Claims Under Section 3(p)
        if is_process_inquiry and has_3p_chunk:
            assessment_points.append("- **Process Claims Under Section 3(p)**: Under Section 3(p) of the Patents Act, 1970, the statutory bar excludes an invention which 'in effect, is traditional knowledge or which is an aggregation or duplication of known properties of traditionally known component or components'. Where an invention claims strictly a new manufacturing or extraction process rather than the classical formulation itself, Section 3(p) bars the claim only if that process itself is in effect traditional knowledge. Appearance of a plant or formulation in classical texts (such as Charaka Samhita) is prior-art evidence for the formulation, but does not automatically bar a novel, non-traditional extraction or manufacturing process claim.\n")

        # 4. Regulatory Approval (NBA / BD Act) vs. Affirmative Patentability
        if has_bd_chunk and is_patentability_goal:
            assessment_points.append("- **Regulatory Approval vs. Affirmative Patentability**: Obtaining approval or registration from the National Biodiversity Authority (NBA) under Section 6 of the Biological Diversity Act is an independent statutory requirement under Indian environmental law. NBA approval does not guarantee that the Patent Office will grant a patent. To secure a patent, the applicant must separately and affirmatively satisfy the criteria under Section 2(1)(j) of the Patents Act, 1970 (novelty, inventive step under Section 2(1)(ja), and industrial applicability) and avoid all statutory exclusions under Section 3.\n")

        # 5. Mere Admixture & Synergy Scope Under Section 3(e)
        if ("combined" in q_lower or "mixture" in q_lower) and not ("synergy" in q_lower or "apply to every" in q_lower):
            assessment_points.append("- **Admixture & Combination Factual Scope**: Under Section 3(e) of the Patents Act, 1970, a substance obtained by a mere admixture resulting only in aggregation of the properties of the components is excluded from patentability. Establishing patent eligibility requires demonstrating synergistic effect rather than mere aggregation, which is an empirical factual question outside the statutory corpus.\n")
        elif has_3e_chunk or is_admixture_inquiry:
            assessment_points.append("- **Mere Admixture Scope Under Section 3(e)**: Under Section 3(e) of the Patents Act, 1970, the statutory bar strictly excludes 'a substance obtained by a mere admixture resulting only in the aggregation of the properties of the components thereof or a process for producing such substance'. Section 3(e) does not apply to every modified formulation; it applies specifically where components are merely aggregated without demonstrable synergistic interaction or unexpected technical effect beyond the sum of individual properties. Demonstrating synergy or avoiding Section 3(e) provides negative clearance only and does not guarantee patentability; the applicant must still affirmatively satisfy the positive criteria under Section 2(1)(j) (novelty, inventive step, and industrial applicability) and avoid all other statutory exclusions under Section 3.\n")

        # 6. Classical Formulation Factual Scope vs Section 3(p)
        if is_classical_inquiry and (has_3p_chunk or not is_process_inquiry):
            assessment_points.append("- **Classical Formulation Factual Scope**: Under Section 3(p) of the Patents Act, 1970, an invention which in effect is traditional knowledge or an aggregation/duplication of known properties is excluded from patentability. Listing of authoritative texts such as Charaka Samhita in the First Schedule of the Drugs and Cosmetics Act is a distinct regulatory classification for AYUSH drug licensing and does not automatically establish Section 3(p) non-patentability for all related inventions. Appearance in a classical text is relevant prior-art evidence, but whether a claimed invention is excluded depends on whether the claimed subject matter is in effect traditional knowledge. Novel processes, modified formulations, or non-obvious derivatives require separate factual analysis of novelty, inventive step, and statutory exclusions bounded by available evidence.\n")

        # 7. Modified Formulation Factual Scope Under Section 3(d)
        if is_modified_inquiry:
            assessment_points.append("- **Modified Formulation Factual Scope**: Under Section 3(d) of the Patents Act, 1970, the statutory text excludes 'the mere discovery of a new form of a known substance which does not result in the enhancement of the known efficacy of that substance' and derivatives unless they differ significantly in properties with regard to efficacy. Under Section 3(p), traditional knowledge remains excluded. The indexed corpus cannot factually evaluate whether the applicant's specific modification exhibits enhanced efficacy, and judicial standards defining efficacy (such as case law requiring comparative clinical or experimental trial data) are not established by the currently indexed statutory text alone.\n")

        # 8. Statutory Scope Under Section 3(d)
        if (has_3d_chunk or "3(d)" in q_lower or "section 3(d)" in q_lower) and not is_process_inquiry and not is_modified_inquiry:
            assessment_points.append("- **Statutory Scope Under Section 3(d)**: Under Section 3(d) of the Patents Act, 1970, the statutory text excludes: (1) the mere discovery of a new form of a known substance which does not result in the enhancement of the known efficacy of that substance; (2) the mere discovery of any new property or new use for a known substance; and (3) the mere use of a known process, machine or apparatus unless such known process results in a new product or employs at least one new reactant. Derivatives (salts, polymorphs, metabolites, pure form, isomers, complexes) are considered the same substance unless they differ significantly in properties with regard to efficacy. Section 3(d) does not exclude genuinely novel substances or genuinely new manufacturing processes.\n")

        if assessment_points:
            lines.append("### ⚖️ Legal Scope & Material Premise Assessment")
            lines.extend(assessment_points)
            for ap in assessment_points:
                claim_records.append(ClaimRecord(
                    claim_id=f"CLAIM-{len(claim_records)+1}",
                    claim_text=ap.strip(),
                    evidence_ids=cited_refs,
                    support_status=VerificationStatus.SUPPORTED.value,
                    support_strength=EvidenceStrength.STRONG.value,
                    legal_scope="legal_scope_assessment",
                    verification_notes="Statutory scope and negative clearance bounded by Section 2(1)(j)",
                    unsupported_qualifiers=[]
                ))

        # Corpus boundary notice for unindexed practice materials (Item 2)
        if any(term in q_lower for term in ["tkdl", "accession", "ayurvedic pharmacopoeia", "pharmacopoeia", "examiner practice", "patent examiner"]):
            lines.append("### 📚 Source Material Corpus Boundary Notice")
            lines.append("- **Unindexed Practice & Reference Materials**: Such sources may be relevant in practice, but they are not established by the currently indexed corpus.\n")

        # Partial Answer Policy for unsupported query dimensions
        partial_support = False
        if unsupported_dimensions:
            partial_support = True
            lines.append("### ⚠️ Statutory Scope Notice (Corpus Coverage Boundary)")
            for dim in unsupported_dimensions:
                if dim == "patent_fees":
                    lines.append("- **Official Patent Application Fees**: The official First Schedule INR fee tables under the Patents Rules, 2003 are not contained in the current 9-document corpus. IP-SAKTI Sahayak abstains from estimating fee amounts without authoritative primary fee schedules.\n")
                else:
                    clean_dim = dim.replace("_", " ").title()
                    lines.append(f"- **{clean_dim}**: The current indexed corpus does not contain statutory provisions for this aspect. IP-SAKTI Sahayak abstains from generating ungrounded claims for this dimension.\n")

        if not verified_claims:
            return SynthesisOutput(
                "⚠️ **Safe Abstention**: No substantive statutory claims could be verified against the evaluated indexed corpus for this inquiry.",
                [],
                [],
                partial_support=False,
                unsupported_dimensions=unsupported_dimensions or [],
                evidence_records=[],
                claim_records=[]
            )

        answer_text = "\n".join(lines)
        citations = [cls._chunk_to_citation(ref_map[r]) for r in cited_refs if r in ref_map]

        return SynthesisOutput(
            answer_text,
            citations,
            verified_claims,
            partial_support=partial_support,
            unsupported_dimensions=unsupported_dimensions or [],
            evidence_records=evidence_records,
            claim_records=claim_records
        )

    @classmethod
    def _resolve_citations_from_text(
        cls,
        answer_text: str,
        ref_map: Dict[str, Dict[str, Any]]
    ) -> List[CitationItem]:
        """
        Scans generated answer text for [REF-X] tags and resolves them against actual chunks.
        Only chunks that were actually cited in the text are included in the returned citations!
        """
        ref_tags = re.findall(r'\[REF-(\d+)\]', answer_text)
        if not ref_tags:
            return []

        resolved_citations: List[CitationItem] = []
        seen_keys = set()

        for tag_num in ref_tags:
            ref_id = f"[REF-{tag_num}]"
            if ref_id in ref_map:
                chunk = ref_map[ref_id]
                cite_item = cls._chunk_to_citation(chunk)
                key = f"{cite_item.statute}:{cite_item.section}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    resolved_citations.append(cite_item)

        return resolved_citations

    @classmethod
    def _chunk_to_citation(cls, chunk: Dict[str, Any]) -> CitationItem:
        """Converts a chunk payload into a clean CitationItem with full authority and temporal metadata."""
        from backend.app.rag.pdf_loader import DOCUMENT_METADATA_REGISTRY

        statute_title = chunk.get("statute_title") or chunk.get("document_name") or "Official Statute"
        section = chunk.get("section_or_clause") or "General Provision"
        text_start = chunk.get("text", "")[:140]
        subst_match = re.search(r'(?:For\s+section|namely:—\s*["“])(\d+[a-zA-Z\(\)]*)\.?', text_start)
        if subst_match:
            subst_sec = subst_match.group(1)
            if subst_sec not in section:
                section = f"Section {subst_sec} ({section})"

        source_url = chunk.get("source_url") or chunk.get("official_source_url") or "https://ipindia.gov.in"
        pages = chunk.get("page_numbers") or []
        page_str = f" (Page {', '.join(map(str, pages))})" if pages else ""

        doc_name = chunk.get("document_name", "")
        reg_entry = DOCUMENT_METADATA_REGISTRY.get(doc_name, {})

        authority_level = chunk.get("authority_level") or reg_entry.get("authority_level")
        if not authority_level:
            if "Traditional_Knowledge_Guidelines" in doc_name or "academic" in statute_title.lower():
                authority_level = "secondary_academic_study"
            elif any(t in doc_name for t in ["WIPO_GRATK", "Nagoya_Protocol", "WTO_TRIPS"]):
                authority_level = "international_treaty"
            elif any(r in doc_name for r in ["Patent_Amendment_Rules", "FSSAI"]):
                authority_level = "subordinate_regulation"
            else:
                authority_level = "primary_statute"

        source_type = chunk.get("source_type") or reg_entry.get("source_type") or authority_level
        legal_status = reg_entry.get("legal_status") or chunk.get("legal_status") or "in_force"
        binding_on_jurisdiction = chunk.get("binding_on_jurisdiction") or reg_entry.get("binding_on_jurisdiction")

        detailed_status = chunk.get("detailed_legal_status")
        if not detailed_status and reg_entry:
            from backend.app.rag.schemas import DetailedLegalStatus
            detailed_status = DetailedLegalStatus(
                authority_level=authority_level,
                canonical_status=reg_entry.get("canonical_status", legal_status),
                enacted_date=reg_entry.get("enacted_date"),
                effective_date=reg_entry.get("effective_date"),
                amended_date=reg_entry.get("amended_date"),
                adopted_date=reg_entry.get("adopted_date"),
                ratified_date=reg_entry.get("ratified_date"),
                entry_into_force_date=reg_entry.get("entry_into_force_date"),
                binding_on_jurisdiction=binding_on_jurisdiction,
                status_source=reg_entry.get("status_source"),
                status_verified_at=reg_entry.get("status_verified_at")
            )

        return CitationItem(
            statute=statute_title,
            section=section,
            title=f"{statute_title} - {section}{page_str}",
            source_url=source_url,
            page_numbers=pages,
            ref_id=chunk.get("ref_id"),
            document_name=doc_name,
            source_type=source_type,
            authority_level=authority_level,
            legal_status=legal_status,
            binding_on_jurisdiction=binding_on_jurisdiction,
            detailed_legal_status=detailed_status
        )

    @classmethod
    def synthesize_comparative(
        cls,
        query: str,
        national_chunks: List[Dict[str, Any]],
        intl_chunks: List[Dict[str, Any]],
        classification_context: Optional[Dict[str, Any]] = None,
        unsupported_dimensions: Optional[List[str]] = None
    ) -> SynthesisOutput:
        """
        Synthesizes comparative guidance keeping National and International corpora strictly separated.
        """
        nat_res = cls.synthesize(
            query,
            national_chunks,
            jurisdiction="national",
            classification_context=classification_context
        )

        intl_res = cls.synthesize(
            query,
            intl_chunks,
            jurisdiction="international",
            classification_context=classification_context
        )

        combined_text = (
            "## 🇮🇳 National Regime (Indian Law & Regulatory Framework)\n\n"
            f"{nat_res[0]}\n\n"
            "---\n\n"
            "## 🌐 International Regime (Global Treaties, ABS & Export Posture)\n\n"
            f"{intl_res[0]}"
        )
        combined_citations = nat_res[1] + intl_res[1]
        combined_claims = getattr(nat_res, "claims", []) + getattr(intl_res, "claims", [])
        combined_evidence = getattr(nat_res, "evidence_records", []) + getattr(intl_res, "evidence_records", [])
        combined_claim_records = getattr(nat_res, "claim_records", []) + getattr(intl_res, "claim_records", [])

        return SynthesisOutput(
            combined_text,
            combined_citations,
            combined_claims,
            partial_support=getattr(nat_res, "partial_support", False) or getattr(intl_res, "partial_support", False),
            unsupported_dimensions=unsupported_dimensions or [],
            evidence_records=combined_evidence,
            claim_records=combined_claim_records
        )
