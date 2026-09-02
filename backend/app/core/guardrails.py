import re
from typing import Dict, Any, List, Optional, Set
from backend.app.core.config import settings
from backend.app.rag.schemas import (
    ClaimVerificationResult,
    VerificationStatus,
    EvidenceStrength,
    GroundedClaim,
    AtomicProposition
)

class GuardrailsEngine:
    """
    Safety, grounding, and compliance layer for IP-SAKTI Sahayak.
    Ensures safe abstention, computes evidence-based confidence scores, and enforces disclaimers.
    """

    OUT_OF_SCOPE_PATTERNS = [
        r'\b(crypto\w*|bitcoin|ethereum|nft|token\w*|blockchain|web3|defi)\b',
        r'\b(tax\w*|taxation|gst|tariff\w*|customs duty|money laundering)\b',
        r'\b(weapon\w*|explosive\w*|firearm\w*|ammunition|missile\w*|bomb\w*)\b',
        r'\b(casino|gambling|betting|sports betting|lottery)\b',
        r'\b(hack\w*|crack\w*|ddos|sql injection|exploit|phishing)\b',
        r'\b(stock market prediction|crypto trading signal|forex trading)\b',
        r'\b(quantum computing|quantum cryptography)\b',
        r'\b(criminal homicide|murder|theft|robbery)\b',
        r'\b(software\w*|algorithms?|source code|computer programs? per se)\b',
        r'\b(ayurveda intellectual property protection act|national ayurvedic herbal export promotion act|ayurvedic clinical trials regulation act|ayurveda special patent fast-track act)\b'
    ]

    AYURVEDA_LEGAL_KEYWORDS = [
        "patent", "trademark", "geographical indication", "gi", "design", "copyright",
        "ayush", "ayurveda", "siddha", "unani", "herbal", "classical", "formulation",
        "section 3", "3(p)", "3(e)", "3(d)", "rule 161", "schedule t", "first schedule",
        "biodiversity", "abs", "nba", "sbb", "form iii", "form i", "traditional knowledge",
        "tkdl", "wipo", "gratk", "nagoya", "trips", "fssai", "ayurveda aahar", "drug",
        "ingredient", "ingredients", "label", "labelling", "package", "packaging",
        "clinical", "manufacturing", "practitioner", "healer", "vaidya", "composition",
        "curcumin", "extract", "extracts", "phytopharmaceutical", "genetic resources",
        "genetic", "access", "prior informed consent", "benefit sharing", "benefit-sharing"
    ]

    STOPWORDS: Set[str] = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "about", "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "from", "up", "down", "of", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
        "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "can", "cannot", "will", "just",
        "should", "now", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "having", "do", "does", "did", "doing", "this", "that", "these", "those"
    }

    KNOWN_DOC_PATTERNS = [
        (r'\bpatents?\s+act\b', {'patent', 'patents', 'act'}),
        (r'\bdrugs?\s+and\s+cosmetics?\b', {'drug', 'drugs', 'cosmetics', 'act'}),
        (r'\bbiological\s+diversity\b', {'biological', 'diversity', 'act'}),
        (r'\bnagoya\s+protocol\b', {'nagoya', 'protocol'}),
        (r'\bwipo\b|\bgratk\b', {'wipo', 'gratk', 'treaty'}),
        (r'\btrips\b', {'trips', 'agreement'}),
        (r'\bfssai\b|\bayurveda\s+aahar\b', {'fssai', 'ayurveda', 'aahar', 'regulations'}),
    ]

    QUERY_META_TERMS: Set[str] = {
        'what', 'does', 'say', 'about', 'tell', 'explain', 'give', 'under', 'which',
        'how', 'can', 'could', 'would', 'is', 'are', 'was', 'were', 'be', 'been',
        'meaning', 'mean', 'describe', 'describes', 'described', 'discuss', 'provide',
        'provides', 'provided', 'provisions', 'provision', 'law', 'statute', 'rule',
        'rules', 'act', 'section', 'article', 'treaty', 'the', 'a', 'an', 'and', 'or',
        'in', 'on', 'at', 'to', 'for', 'with', 'of', 'regime', 'framework', 'requirement',
        'requirements', 'details', 'mention', 'issue', 'implications', 'specific', 'impose',
        'imposes', 'imposed', 'general', 'particular', 'exact', 'specify', 'specifies',
        'specified', 'regarding', 'granted', 'grant', 'role', 'play', 'characterize', 'compare',
        'comparison', 'comparing', 'domestic', 'national', 'international', 'versus', 'vs',
        'differ', 'difference', 'differs', 'override', 'overrides', 'overridden', 'prevail', 'prevails',
        'always', 'never', 'automatically', 'automatic', 'guarantee', 'guarantees', 'guaranteed',
        'mandatory', 'necessarily', 'strict', 'strictly', 'solely', 'only', 'all', 'none', 'every',
        'apply', 'applies', 'applicable', 'they', 'their', 'them', 'where', 'when', 'foreign',
        'domestic', 'local', 'without', 'exception', 'exceptions', 'mention', 'mentioning', 'mentioned',
        'category', 'categories', 'class', 'classes', 'subject', 'matter', 'govern', 'governs',
        'governed', 'governing', 'currently', 'presently', 'today', 'bound', 'signing', 'signed',
        'interact', 'interaction', 'relationship', 'relate', 'related', 'worldwide', 'mandates', 'mandate',
        'need', 'needs', 'preparing', 'prepare', 'patients', 'patient', 'want', 'wants', 'produce', 'producing', 'seek', 'seeking'
    }

    SYNONYM_GROUPS: List[Set[str]] = [
        {'fee', 'fees', 'cost', 'costs', 'payment', 'payments', 'tariff'},
        {'filing', 'file', 'filed'},
        {'patent', 'patents', 'patentable', 'patentability', 'monopoly', 'monopolies', 'exclusive', 'exclusivity'},
        {'bar', 'bars', 'barred', 'exclusion', 'exclusions', 'excluded', 'prohibition', 'prohibitions', 'prohibited', 'not patentable', 'not inventions', 'reject', 'rejects', 'rejection', 'rejected', 'refuse', 'refused', 'refusal', 'deny', 'denied', 'denial', 'non-patentable', 'non'},
        {'tradition', 'traditional', 'tk', 'codified', 'classical', 'ancient', 'historic', 'heritage', 'charaka', 'charak', 'samhita', 'sushruta', 'ashtanga', 'authoritative', 'ayurveda', 'ayurvedic'},
        {'knowledge', 'wisdom', 'lore'},
        {'formulation', 'formulations', 'formula', 'formulas', 'composition', 'compositions', 'preparation', 'product', 'substance', 'invention', 'component', 'components'},
        {'synergy', 'synergistic', 'unexpected', 'enhancement'},
        {'admixture', 'mixture', 'aggregation', 'combination'},
        {'origin', 'source', 'provenance', 'geographical'},
        {'disclosure', 'disclose', 'disclosed', 'declaration', 'declaring', 'declare'},
        {'benefit', 'benefits', 'benefit-sharing', 'abs', 'approval', 'nba', 'sbb', 'intimation', 'prior'},
        {'practitioner', 'practitioners', 'vaid', 'vaids', 'vaidya', 'vaidyas', 'hakim', 'hakims', 'healer', 'ayush', 'practising', 'practicing'},
        {'label', 'labels', 'labelling', 'labeling', 'packing', 'package'},
        {'aahar', 'aahara', 'food', 'diet', 'dietary', 'supplement', 'supplements', 'nutraceutical'},
        {'plant', 'plants', 'herb', 'herbs', 'herbal', 'medicinal', 'botanical', 'biological', 'genetic', 'resource', 'resources'},
        {'rule', 'rules', 'standard', 'standards', 'regulation', 'regulations', 'norm', 'norms', 'guideline', 'guidelines', 'provision', 'provisions'},
        {'obtain', 'obtained', 'acquire', 'acquired', 'access', 'accessed', 'source', 'sourced'}
    ]

    @classmethod
    def verify_query_evidence_alignment(
        cls,
        query: str,
        candidates: List[Dict[str, Any]],
        jurisdiction: str = "national",
        is_comparative: bool = False
    ) -> Dict[str, Any]:
        """
        Query-to-Evidence Concept Alignment Gate.
        Verifies that candidate evidence genuinely contains answers to the user's specific inquiry concepts,
        preventing spurious keyword matches (e.g. Nagoya licence fees passed for Indian patent filing fees).
        """
        if not candidates:
            return {"aligned": False, "ratio": 0.0, "reason": "No candidate evidence provided."}

        q_lower = query.lower()
        comp_flag = is_comparative or any(comp in q_lower for comp in ["comparison", "comparative", "compare", "both", "versus", "vs", "override", "under indian law and"])

        # 1. Strict Cross-Jurisdiction Boundary Enforcement
        if jurisdiction == "international":
            # Questions about domestic Indian statutory schemes or Indian filing under international corpus
            if any(term in q_lower for term in ["indian patent", "indian filing", "under indian law", "patents act", "rule 161", "rule 158b", "charaka"]) and not comp_flag:
                return {
                    "aligned": False,
                    "ratio": 0.0,
                    "reason": "Query asks about Indian domestic statutory regime under international jurisdiction."
                }

        # 2. Named Statute / Treaty Presence Check: If query explicitly asks about specific treaties/statutes,
        # ensure those exact authorities are represented in retrieved evidence according to jurisdiction
        # First: detect unindexed or fictitious enactments cited in the query
        for act_match in re.finditer(r'\b([a-z\s]+?)\s+act\b', q_lower):
            cand_act = act_match.group(1).strip()
            cand_act_words = [w for w in cand_act.split() if w not in cls.STOPWORDS and w not in cls.QUERY_META_TERMS]
            cand_act_core = " ".join(cand_act_words)
            if cand_act_core and not any(known in cand_act_core for known in ["patent", "patents", "drugs", "cosmetics", "biological", "diversity", "biodiversity", "food", "fssai"]):
                return {
                    "aligned": False,
                    "ratio": 0.0,
                    "reason": f"Query cites unindexed enactment '{cand_act_core.title()} Act', which is not part of the official indexed legal corpus."
                }

        if jurisdiction == "national":
            nat_docs = {
                'patents act': 'Patents_Act_1970.PDF',
                'drugs and cosmetics': 'Drugs_and_Cosmetics_Act_Ayurveda.pdf',
                'biological diversity': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'ayurveda aahar': 'FSSAI_Ayurveda_Aahar_Regulations_2022.pdf'
            }
            for doc_key, doc_file in nat_docs.items():
                if doc_key in q_lower:
                    if not any(c.get("document_name") == doc_file for c in candidates):
                        return {
                            "aligned": False,
                            "ratio": 0.0,
                            "reason": f"Query specifically asks about {doc_key}, but retrieved evidence contains no authoritative provisions from {doc_file}."
                        }
        elif jurisdiction == "international":
            intl_docs = {
                'nagoya': 'Nagoya_Protocol_ABS.pdf',
                'wipo': 'WIPO_GRATK_Treaty_2024.pdf',
                'gratk': 'WIPO_GRATK_Treaty_2024.pdf',
                'trips': 'WTO_TRIPS_Agreement.pdf'
            }
            for doc_key, doc_file in intl_docs.items():
                if doc_key in q_lower:
                    if not any(c.get("document_name") == doc_file for c in candidates):
                        return {
                            "aligned": False,
                            "ratio": 0.0,
                            "reason": f"Query specifically asks about {doc_key}, but retrieved evidence contains no authoritative provisions from {doc_file}."
                        }

        # 3. Extract stripped document names mentioned in query
        stripped_tokens = set()
        for pat, tokens in cls.KNOWN_DOC_PATTERNS:
            if re.search(pat, q_lower):
                stripped_tokens.update(tokens)

        # In international queries, strip Indian territorial terms since treaties use Contracting Party / Country of Origin
        if jurisdiction == "international":
            stripped_tokens.update({'india', 'indian'})

        # In comparative queries, strip terms belonging to the alternate jurisdiction
        if comp_flag:
            if jurisdiction in ["international", "comparative"]:
                stripped_tokens.update({'india', 'indian', 'section', '3(p)', '3(p', '3(e)', '3(e', '3(d)', '3(d', '3p', '3e', '3d', 'act', 'rule', 'patents', 'cosmetics', 'diversity', 'fssai'})
            if jurisdiction in ["national", "comparative"]:
                stripped_tokens.update({'trips', 'wipo', 'gratk', 'nagoya', 'treaty', 'protocol', 'international', 'article', '27'})
            elif jurisdiction == "national":
                stripped_tokens.update({'trips', 'nagoya', 'wipo', 'gratk', 'article', '27', 'treaty', 'protocol', 'agreement'})

        # 4. Extract inquiry concept tokens
        tokens = [w for w in re.findall(r'\b[a-z0-9\(\)]+\b', q_lower) if w not in cls.STOPWORDS and w not in cls.QUERY_META_TERMS and len(w) >= 3]
        inquiry_terms = [t for t in tokens if t not in stripped_tokens]
        if not inquiry_terms:
            if is_comparative or any(re.search(pat, q_lower) for pat, _ in cls.KNOWN_DOC_PATTERNS):
                return {"aligned": True, "ratio": 1.0, "matched": list(stripped_tokens)}
            inquiry_terms = tokens

        # 5. Check presence across all candidate chunks
        evidence_text = " ".join([
            (c.get("text", "") + " " + c.get("statute_title", "") + " " + c.get("section_or_clause", "")).lower()
            for c in candidates
        ])

        matched_terms = []
        unmatched_terms = []

        for term in inquiry_terms:
            # Direct match
            if term in evidence_text:
                matched_terms.append(term)
                continue
            # Synonym match
            syn_found = False
            for sgroup in cls.SYNONYM_GROUPS:
                if term in sgroup:
                    if any(syn in evidence_text for syn in sgroup):
                        syn_found = True
                        break
            if syn_found:
                matched_terms.append(term)
            else:
                unmatched_terms.append(term)

        ratio = len(matched_terms) / len(inquiry_terms) if inquiry_terms else 1.0

        if ratio < 0.50:
            return {
                "aligned": False,
                "ratio": round(ratio, 2),
                "unmatched": unmatched_terms,
                "reason": f"Retrieved evidence covers only {round(ratio * 100)}% of inquiry concepts (missing: {unmatched_terms})"
            }

        # 6. Compound Concept Co-occurrence Verification
        # Prevents cross-chunk keyword stitching (e.g. 'filing' on page 4 and 'fee' on page 27)
        compound_pairs = [
            ({'filing', 'file', 'filed'}, {'fee', 'fees'}),
            ({'commercialization', 'commercial'}, {'fee', 'fees'}),
            ({'traditional'}, {'knowledge'}),
            ({'country'}, {'origin'}),
            ({'first'}, {'schedule'}),
            ({'rule'}, {'161'}),
        ]
        for group_a, group_b in compound_pairs:
            if any(w in q_lower for w in group_a) and any(w in q_lower for w in group_b):
                found_cooccurrence = False
                for c in candidates:
                    txt = (c.get('text', '') + ' ' + c.get('statute_title', '') + ' ' + c.get('section_or_clause', '')).lower()
                    chunk_has_a = any(w in txt for w in group_a)
                    chunk_has_b = any(w in txt for w in group_b)
                    if chunk_has_a and chunk_has_b:
                        found_cooccurrence = True
                        break
                if not found_cooccurrence:
                    return {
                        "aligned": False,
                        "ratio": 0.0,
                        "reason": f"Query requires compound co-occurrence of {group_a} and {group_b}, but no single retrieved chunk contains both."
                    }

        # 7. Explicit Section/Article check
        if jurisdiction == "national":
            sec_matches = re.findall(r'\b(?:section|rule)\s+([0-9]+[a-z0-9\(\)]*)|\b([0-9]+[\(][a-z0-9]+[\)])', q_lower)
        elif jurisdiction == "international":
            sec_matches = re.findall(r'\barticle\s+([0-9]+[a-z0-9\(\)]*)', q_lower)
        else:
            sec_matches = re.findall(r'\b(?:section|rule|article)\s+([0-9]+[a-z0-9\(\)]*)|\b([0-9]+[\(][a-z0-9]+[\)])', q_lower)

        target_sections = set()
        for sm in sec_matches:
            target_sections.update([s for s in sm if s])

        if target_sections:
            sec_found = False
            for sec in target_sections:
                m_sub = re.match(r'^([0-9]+)[\(]?([a-z0-9]+)[\)]?$', sec)
                for c in candidates:
                    sec_clause = c.get("section_or_clause", "").lower()
                    text_lower = c.get("text", "").lower()
                    if sec in sec_clause or sec in text_lower:
                        sec_found = True
                        break
                    if m_sub:
                        s_num, s_clause = m_sub.groups()
                        clause_pat = f"({s_clause})"
                        if clause_pat in text_lower or f"clause ({s_clause})" in text_lower or s_clause in sec_clause:
                            sec_found = True
                            break
                if not sec_found and m_sub:
                    s_num, s_clause = m_sub.groups()
                    clause_pat = f"({s_clause})"
                    has_clause = any(clause_pat in c.get("text", "").lower() or s_clause in c.get("section_or_clause", "").lower() for c in candidates)
                    has_sec = any(s_num in c.get("section_or_clause", "").lower() or f"section {s_num}" in c.get("text", "").lower() or f"{s_num}." in c.get("text", "").lower() or c.get("document_name") == "Patents_Act_1970.PDF" for c in candidates)
                    if has_clause and has_sec:
                        sec_found = True
                if sec_found:
                    break
            if not sec_found:
                return {
                    "aligned": False,
                    "ratio": round(ratio, 2),
                    "reason": f"Query specifically requested sections {target_sections}, but candidate evidence did not contain matching provisions."
                }

        return {
            "aligned": True,
            "ratio": round(ratio, 2),
            "matched": matched_terms
        }

    @classmethod
    def check_query_safety(cls, query: str) -> Dict[str, Any]:
        """Validates if the user query is within the statutory/regulatory scope of the assistant."""
        query_lower = query.lower()

        for pattern in cls.OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, query_lower):
                return {
                    "is_safe": False,
                    "reason": "The request contains topics outside the scope of Ayurvedic IPR and regulatory law.",
                    "suggestion": "Please ask questions regarding Ayurvedic patents, GI, Trademarks, Biodiversity ABS, or drug licensing."
                }

        # Check domain relevance
        has_domain_term = any(k in query_lower for k in cls.AYURVEDA_LEGAL_KEYWORDS)
        if len(query.split()) > 4 and not has_domain_term:
            # Check if general legal / regulatory question
            general_legal = [
                "law", "section", "act", "rule", "license", "authority", "court", "origin", "treaty",
                "require", "requirement", "provision", "compliance", "exemption", "bar"
            ]
            if not any(g in query_lower for g in general_legal):
                return {
                    "is_safe": False,
                    "reason": "The query does not appear to relate to Ayurvedic Intellectual Property or Drug Regulations.",
                    "suggestion": "Please rephrase specifying Ayurvedic formulations, patent sections (e.g. §3(p)), or regulatory acts."
                }

        return {"is_safe": True, "reason": "Query within domain scope."}

    @classmethod
    def evaluate_grounding(
        cls,
        answer_text: str,
        retrieved_chunks: List[Dict[str, Any]],
        cited_ref_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculates an evidence-based grounding score by measuring:
        1. Whether safe abstention was triggered or retrieved evidence was empty.
        2. Presence of valid deterministic evidence references ([REF-1], [REF-2]).
        3. Lexical overlap between substantive claims in the answer and the cited evidence passages.
        """
        is_comparative = "## 🇮🇳" in answer_text and "## 🌐" in answer_text
        has_abstain_text = "insufficient evidence" in answer_text.lower() or "safe abstention" in answer_text.lower()
        if not retrieved_chunks or not answer_text or (has_abstain_text and (not is_comparative or not cited_ref_ids)):
            return {
                "confidence_score": 0.0,
                "is_grounded": False,
                "abstain": True,
                "overlap_ratio": 0.0,
                "reason": "Insufficient statutory evidence in indexed corpus."
            }

        # Tokenize substantive answer text excluding meta-disclosures and stopwords
        eval_text = re.split(r'###\s+⚖️\s+Legal Scope|###\s+📚\s+Source Material|###\s+⚠️\s+Statutory Scope', answer_text)[0]
        answer_tokens = [w for w in re.findall(r'\b[a-z]{3,}\b', eval_text.lower()) if w not in cls.STOPWORDS]
        if not answer_tokens:
            answer_tokens = [w for w in re.findall(r'\b[a-z]{3,}\b', answer_text.lower()) if w not in cls.STOPWORDS]
        if not answer_tokens:
            return {
                "confidence_score": 0.0,
                "is_grounded": False,
                "abstain": True,
                "overlap_ratio": 0.0,
                "reason": "Answer text contained no substantive tokens."
            }

        # Select evidence text from cited chunks, or all top retrieved chunks if unmapped
        evidence_text_parts = []
        for c in retrieved_chunks:
            evidence_text_parts.append(c.get("text", "").lower())

        combined_evidence = " ".join(evidence_text_parts)
        evidence_tokens = set(re.findall(r'\b[a-z]{3,}\b', combined_evidence))

        if not evidence_tokens:
            return {
                "confidence_score": 0.0,
                "is_grounded": False,
                "abstain": True,
                "overlap_ratio": 0.0,
                "reason": "Retrieved evidence contained no text."
            }

        # Compute content token overlap
        matched_tokens = [t for t in answer_tokens if t in evidence_tokens]
        overlap_ratio = len(matched_tokens) / len(answer_tokens)

        # Base confidence derived from lexical overlap
        # Check if explicit evidence IDs ([REF-X]) are present
        has_ref_tags = bool(re.search(r'\[REF-\d+\]', answer_text))
        ref_bonus = 0.20 if has_ref_tags else 0.0

        # Top retrieval score influence
        top_score = max([c.get("rerank_score", c.get("dense_score", c.get("hybrid_rrf_score", 0.3))) for c in retrieved_chunks])
        score_component = min(0.20, max(0.0, float(top_score) * 0.20)) if top_score > 0 else 0.05

        confidence_score = round(min(0.95, (overlap_ratio * 0.65) + ref_bonus + score_component), 2)
        is_grounded = overlap_ratio >= settings.GROUNDING_MIN_OVERLAP and confidence_score >= settings.CONFIDENCE_THRESHOLD

        evidence_strength = cls.audit_evidence_strength([], is_grounded, confidence_score)

        return {
            "confidence_score": confidence_score if is_grounded else min(confidence_score, 0.45),
            "evidence_strength": evidence_strength,
            "is_grounded": is_grounded,
            "abstain": not is_grounded,
            "overlap_ratio": round(overlap_ratio, 2)
        }

    @classmethod
    def audit_evidence_strength(
        cls,
        verified_claims: List[GroundedClaim],
        is_grounded: bool,
        confidence_score: float
    ) -> str:
        """
        Calibrates evidence strength into human-interpretable categories:
        - Strong Evidence: Rigorously supported by primary statutes/regulations with high concept fidelity.
        - Moderate Evidence: Partially supported or grounded in secondary/guideline materials.
        - Insufficient Evidence: Lacks direct statutory grounding, missing provisions, or triggered safe abstention.
        """
        if not is_grounded or confidence_score < settings.CONFIDENCE_THRESHOLD:
            return EvidenceStrength.INSUFFICIENT.value

        if verified_claims:
            all_supported = all(c.verification_status == VerificationStatus.SUPPORTED.value for c in verified_claims)
            has_primary = any(any(getattr(cite, 'source_type', '') == "primary_statute" or getattr(cite, 'authority_level', '') == "primary_statute" for cite in c.citations) for c in verified_claims)
            if all_supported and has_primary and confidence_score >= 0.80:
                return EvidenceStrength.STRONG.value
            return EvidenceStrength.MODERATE.value

        if confidence_score >= 0.85:
            return EvidenceStrength.STRONG.value
        elif confidence_score >= 0.60:
            return EvidenceStrength.MODERATE.value
        else:
            return EvidenceStrength.INSUFFICIENT.value

    @classmethod
    def detect_unsupported_qualifiers(
        cls,
        claim_text: str,
        evidence_text: str,
        assigned_chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Detects unsupported material qualifiers (Section 4):
        - absolute universality / necessity: 'always', 'never', 'automatically', 'guaranteed', 'all', 'without exception'
        - mandatory vs discretionary: 'must' vs 'may' where evidence has exceptions
        - temporal inflation: 'currently binding', 'currently in force', 'now binding in India'
        - source authority inflation: academic study cited as binding statutory authority
        """
        unsupported = []
        c_lower = claim_text.lower()
        ev_lower = evidence_text.lower()

        # 1. Universality / Automaticity where evidence has exemptions/provisos
        has_proviso_or_exception = any(cond in ev_lower for cond in [
            "provided that", "proviso", "except", "exempt", "unless", "in specified", "subject to", "prescribed manner"
        ])
        for q in ["always", "automatically", "guaranteed", "without exception", "freely"]:
            if re.search(rf'\b{q}\b', c_lower):
                if not re.search(rf'\b{q}\b', ev_lower):
                    if has_proviso_or_exception or "freely" in q:
                        unsupported.append(q)

        # 2. Temporal inflation (e.g. adopted treaty claimed to be currently binding/in force in India)
        if any(term in c_lower for term in [
            "currently binding", "currently in force", "now binding in india", "binding domestic law", "binding on india currently", "legally binding in india"
        ]):
            for chunk in assigned_chunks:
                legal_status = str(chunk.get("legal_status", "")).lower()
                binding_on = str(chunk.get("binding_on_jurisdiction", "")).lower()
                doc_name = str(chunk.get("document_name", "")).lower()
                if "gratk" in doc_name or "adopted" in legal_status or "pending" in legal_status or "not currently binding" in binding_on:
                    unsupported.append("currently binding")
                    break

        # 3. Source-authority inflation: academic study cited as binding statutory law
        if any(b_term in c_lower for b_term in [
            "binding statute", "statutory mandate", "statutory requirement", "binding law", "primary statute"
        ]):
            for chunk in assigned_chunks:
                auth = str(chunk.get("authority_level", "")).lower()
                doc_name = str(chunk.get("document_name", "")).lower()
                if auth == "secondary_academic_study" or "academic" in doc_name:
                    unsupported.append("binding statutory authority")
                    break

        return unsupported

    @classmethod
    def strip_unsupported_qualifiers(
        cls,
        claim_text: str,
        unsupported_qualifiers: Optional[List[str]] = None
    ) -> str:
        """
        Safely strips unsupported absolute/mandatory qualifiers from a claim.

        SAFETY INVARIANT (Item 4):
        - NEVER transforms an unsupported statement into a new substantive legal conclusion.
        - NEVER maps 'always non-patentable' -> 'presumptively excluded ... unless novel non-obvious features...'
          or introduces any other unverified legal rules/exceptions during post-processing.
        - Pure qualifier stripping only: removes unsupported qualifier words (e.g. 'always', 'automatically',
          'guaranteed', 'without exception', 'freely') without injecting new legal qualifications.
        """
        if not claim_text:
            return ""

        qualifiers_to_strip = unsupported_qualifiers or [
            "always and without exception",
            "without exception",
            "always",
            "automatically",
            "guaranteed",
            "freely"
        ]

        cleaned = claim_text
        for q in qualifiers_to_strip:
            if q in ["currently binding", "binding statutory authority"]:
                continue
            pattern = rf'\b{re.escape(q)}\b'
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Normalize any resulting double spaces or dangling punctuation
        cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned).strip()
        cleaned = re.sub(r'\s+([,.\?!])', r'\1', cleaned)
        return cleaned

    @classmethod
    def decompose_claim_propositions(
        cls,
        claim_text: str,
        assigned_chunks: List[Dict[str, Any]]
    ) -> List[AtomicProposition]:
        """
        Decomposes compound legal claims into atomic propositions (Section 1).
        Distinguishes:
        - statutory_rule: statutory prescription (e.g. Section 3(p) exclusion, Rule 161 labelling)
        - factual_premise: applicant formulation classification (e.g. Charaka recipe is TK)
        - application_conclusion: ultimate grant/denial conclusion
        - temporal_status: in force / adopted status
        """
        propositions = []
        c_lower = claim_text.lower()
        assigned_ref_ids = [c.get("ref_id", "") for c in assigned_chunks if c.get("ref_id")]
        ev_text = " ".join([c.get("text", "").lower() for c in assigned_chunks])

        # 1. Statutory rule proposition
        sec_matches = re.findall(r'\b(?:section|rule|article|regulation)\s+([0-9]+[a-z0-9\(\)]*)', c_lower)
        if sec_matches or any(k in c_lower for k in ["prohibits", "excludes", "requires", "mandates", "prescribes", "not an invention"]):
            prop_status = VerificationStatus.UNSUPPORTED.value
            notes = "Statutory rule not found in assigned evidence."
            if any(sec in ev_text for sec in sec_matches) or any(k in ev_text for k in ["not inventions", "shall not", "required", "proviso", "tradition", "admixture"]):
                prop_status = VerificationStatus.SUPPORTED.value
                notes = "Statutory rule directly supported by assigned evidence."
            propositions.append(AtomicProposition(
                proposition_text=f"Statutory rule governing {sec_matches if sec_matches else 'subject matter'}",
                proposition_type="statutory_rule",
                assigned_evidence_ids=assigned_ref_ids,
                status=prop_status,
                notes=notes
            ))

        # 2. Factual Premise / Specific Formulation Classification
        if any(term in c_lower for term in [
            "charaka formulation", "classical formulation", "user's formulation", "applicant's formulation",
            "modified formulation", "new mixture", "combined known", "specific formulation"
        ]):
            has_factual_evidence = any("charaka" in c.get("text", "").lower() for c in assigned_chunks)
            if has_factual_evidence:
                f_status = VerificationStatus.PARTIALLY_SUPPORTED.value
                f_notes = "Authoritative classical text is recognized in statute schedule, but applicant's specific preparation requires independent factual verification."
                missing_premise = "material_missing_premise"
            else:
                f_status = VerificationStatus.UNSUPPORTED.value
                f_notes = "Corpus establishes general statutory rule, but does not establish factual classification of applicant's specific product."
                missing_premise = "material_missing_premise"

            propositions.append(AtomicProposition(
                proposition_text="Factual classification of applicant's formulation under statutory category",
                proposition_type="factual_premise",
                assigned_evidence_ids=assigned_ref_ids,
                status=f_status,
                missing_premise_type=missing_premise,
                notes=f_notes
            ))

        # 3. Application conclusion (patent grant/bar)
        if any(term in c_lower for term in ["cannot be patented", "can be patented", "excluded from patentability", "patent protection"]):
            propositions.append(AtomicProposition(
                proposition_text="Legal conclusion on patent eligibility",
                proposition_type="application_conclusion",
                assigned_evidence_ids=assigned_ref_ids,
                status=VerificationStatus.SUPPORTED.value if any(p.status == VerificationStatus.SUPPORTED.value for p in propositions) else VerificationStatus.PARTIALLY_SUPPORTED.value,
                notes="Conclusion conditioned on factual satisfaction of statutory premises."
            ))

        return propositions

    @classmethod
    def verify_claim_against_evidence(
        cls,
        claim_text: str,
        assigned_chunks: List[Dict[str, Any]]
    ) -> ClaimVerificationResult:
        """
        Server-side Claim-to-Evidence Validator.
        Evaluates a specific generated claim against ONLY the exact evidence chunks assigned to it.
        Enforces:
        1. No orphan claims (assigned_chunks must not be empty).
        2. Exact statutory section/rule/article alignment where asserted.
        3. Unsupported qualifier detection (Section 4).
        4. Atomic proposition decomposition & material missing premise tracking (Sections 1 & 8).
        5. Direct contradiction / statutory bar check.
        6. Substantive concept and synonym coverage.
        7. Strict separation of retrieval similarity from legal entailment (Section 3).
        """
        if not assigned_chunks:
            return ClaimVerificationResult(
                claim_text=claim_text,
                status=VerificationStatus.UNSUPPORTED.value,
                assigned_evidence_ids=[],
                valid_evidence_ids=[],
                overlap_score=0.0,
                concept_alignment_score=0.0,
                retrieval_relevance=0.0,
                evidence_relevance=0.0,
                claim_entailment=VerificationStatus.UNSUPPORTED.value,
                legal_conclusion_confidence=0.0,
                propositions=[],
                unsupported_qualifiers=[],
                has_material_missing_premise=False,
                notes="No evidence chunks assigned to this substantive claim."
            )

        c_text_lower = claim_text.lower()
        assigned_ref_ids = [c.get("ref_id", "") for c in assigned_chunks if c.get("ref_id")]

        # Combine text of ONLY assigned chunks
        evidence_text = " ".join([
            (c.get("text", "") + " " + c.get("statute_title", "") + " " + c.get("section_or_clause", "")).lower()
            for c in assigned_chunks
        ])

        # 1. Section / Article alignment check
        claim_sec_matches = re.findall(r'\b(?:section|rule|article)\s+([0-9]+[a-z0-9\(\)]*)|\b([0-9]+[\(][a-z0-9]+[\)])', c_text_lower)
        target_sections = set()
        for sm in claim_sec_matches:
            target_sections.update([s for s in sm if s])

        if target_sections:
            sec_match_found = False
            for sec in target_sections:
                m_sub = re.match(r'^([0-9]+)[\(]?([a-z0-9]+)[\)]?$', sec)
                for c in assigned_chunks:
                    sec_clause = c.get("section_or_clause", "").lower()
                    chunk_text = c.get("text", "").lower()
                    if sec in sec_clause or sec in chunk_text:
                        sec_match_found = True
                        break
                    if m_sub:
                        s_num, s_clause = m_sub.groups()
                        clause_pat = f"({s_clause})"
                        if clause_pat in chunk_text or f"clause ({s_clause})" in chunk_text or s_clause in sec_clause:
                            sec_match_found = True
                            break
                if sec_match_found:
                    break
            if not sec_match_found:
                return ClaimVerificationResult(
                    claim_text=claim_text,
                    status=VerificationStatus.UNSUPPORTED.value,
                    assigned_evidence_ids=assigned_ref_ids,
                    valid_evidence_ids=[],
                    overlap_score=0.0,
                    concept_alignment_score=0.0,
                    retrieval_relevance=0.0,
                    evidence_relevance=0.0,
                    claim_entailment=VerificationStatus.UNSUPPORTED.value,
                    legal_conclusion_confidence=0.0,
                    propositions=[],
                    unsupported_qualifiers=[],
                    has_material_missing_premise=False,
                    notes=f"Citation mismatch: Claim asserts section {target_sections} not present in assigned evidence chunks."
                )

        # 2. Unsupported Qualifiers Check (Section 4)
        unsupported_qualifiers = cls.detect_unsupported_qualifiers(claim_text, evidence_text, assigned_chunks)

        # 3. Proposition Decomposition & Material Missing Premise (Sections 1 & 8)
        propositions = cls.decompose_claim_propositions(claim_text, assigned_chunks)
        has_material_missing = any(p.missing_premise_type == "material_missing_premise" for p in propositions)

        # 4. Extract substantive inquiry concepts from claim
        tokens = [w for w in re.findall(r'\b[a-z0-9\(\)]+\b', c_text_lower) if w not in cls.STOPWORDS and w not in cls.QUERY_META_TERMS and len(w) >= 3]
        if not tokens:
            tokens = [w for w in re.findall(r'\b[a-z]{3,}\b', c_text_lower) if w not in cls.STOPWORDS]

        matched_tokens = []
        for t in tokens:
            if t in evidence_text:
                matched_tokens.append(t)
                continue
            for sgroup in cls.SYNONYM_GROUPS:
                if t in sgroup and any(syn in evidence_text for syn in sgroup):
                    matched_tokens.append(t)
                    break

        overlap_ratio = len(matched_tokens) / len(tokens) if tokens else 0.0

        # 5. Direct contradiction / polarity check
        if any(perm in c_text_lower for perm in ["freely patentable", "can be patented without novelty", "eligible for patent without"]):
            if any(bar in evidence_text for bar in ["not inventions", "shall not be patented", "prohibited"]):
                return ClaimVerificationResult(
                    claim_text=claim_text,
                    status=VerificationStatus.CONTRADICTED.value,
                    assigned_evidence_ids=assigned_ref_ids,
                    valid_evidence_ids=[],
                    overlap_score=round(overlap_ratio, 2),
                    concept_alignment_score=0.0,
                    retrieval_relevance=0.8,
                    evidence_relevance=round(overlap_ratio, 2),
                    claim_entailment=VerificationStatus.CONTRADICTED.value,
                    legal_conclusion_confidence=0.0,
                    propositions=propositions,
                    unsupported_qualifiers=unsupported_qualifiers,
                    has_material_missing_premise=has_material_missing,
                    notes="Claim contradicts statutory bar stated in assigned evidence."
                )

        # 6. Status determination with qualifier & premise sensitivity
        if any(q in ["currently binding", "binding statutory authority"] for q in unsupported_qualifiers):
            status = VerificationStatus.UNSUPPORTED.value
            notes = f"Rejected: Claim asserts unsupported legal status or authority: {unsupported_qualifiers}."
            legal_conf = 0.0
        elif unsupported_qualifiers:
            status = VerificationStatus.PARTIALLY_SUPPORTED.value
            notes = f"Partially supported: Claim asserts absolute qualifiers {unsupported_qualifiers} not established in evidence."
            legal_conf = 0.65
        elif has_material_missing:
            status = VerificationStatus.PARTIALLY_SUPPORTED.value
            notes = "Partially supported: Statutory rule is verified, but applicant formulation details constitute a material missing premise."
            legal_conf = 0.70
        elif overlap_ratio >= 0.55:
            status = VerificationStatus.SUPPORTED.value
            notes = "Claim is verified and supported by assigned evidence."
            legal_conf = 0.95
        elif overlap_ratio >= 0.35:
            status = VerificationStatus.PARTIALLY_SUPPORTED.value
            notes = "Claim is partially supported by assigned evidence."
            legal_conf = 0.70
        else:
            status = VerificationStatus.UNSUPPORTED.value
            notes = f"Insufficient concept overlap with assigned evidence ({round(overlap_ratio*100)}%)."
            legal_conf = 0.0

        return ClaimVerificationResult(
            claim_text=claim_text,
            status=status,
            assigned_evidence_ids=assigned_ref_ids,
            valid_evidence_ids=assigned_ref_ids if status in [VerificationStatus.SUPPORTED.value, VerificationStatus.PARTIALLY_SUPPORTED.value] else [],
            overlap_score=round(overlap_ratio, 2),
            concept_alignment_score=round(overlap_ratio, 2),
            retrieval_relevance=0.85 if assigned_chunks else 0.0,
            evidence_relevance=round(overlap_ratio, 2),
            claim_entailment=status,
            legal_conclusion_confidence=legal_conf,
            propositions=propositions,
            unsupported_qualifiers=unsupported_qualifiers,
            has_material_missing_premise=has_material_missing,
            notes=notes
        )

    @classmethod
    def inject_disclaimer(cls, text: str) -> str:
        """Enforces statutory standing disclaimer on generated guidance."""
        disclaimer = f"\n\n---\n⚖️ **Legal Notice**: {settings.LEGAL_DISCLAIMER}"
        if "Legal Notice" not in text and "Disclaimer" not in text:
            return text + disclaimer
        return text
