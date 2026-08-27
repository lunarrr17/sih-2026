import re
from typing import Dict, Any, List
from backend.app.core.config import settings

class GuardrailsEngine:
    """
    Safety, grounding, and compliance layer for IP-SAKTI Sahayak.
    Ensures safe abstention, computes confidence scores, and enforces disclaimers.
    """

    OUT_OF_SCOPE_PATTERNS = [
        r'\b(crypto|bitcoin|ethereum|nft|token)\b',
        r'\b(tax evasion|evade tax|money laundering|tax fraud)\b',
        r'\b(weapon|explosive|firearm|ammunition)\b',
        r'\b(casino|gambling|betting|sports betting)\b',
        r'\b(hack|crack|ddos|sql injection|exploit)\b',
        r'\b(stock market prediction|crypto trading signal)\b'
    ]

    AYURVEDA_LEGAL_KEYWORDS = [
        "patent", "trademark", "geographical indication", "gi", "design", "copyright",
        "ayush", "ayurveda", "siddha", "unani", "herbal", "classical", "formulation",
        "section 3", "3(p)", "3(e)", "3(d)", "rule 161", "schedule t", "first schedule",
        "biodiversity", "abs", "nba", "sbb", "form iii", "form i", "traditional knowledge",
        "tkdl", "wipo", "gratk", "nagoya", "trips", "fssai", "ayurveda aahar", "drug",
        "ingredient", "ingredients", "label", "labelling", "package", "packaging",
        "clinical", "manufacturing", "practitioner", "healer", "vaidya", "composition"
    ]

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
    def evaluate_grounding(cls, answer_text: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates a confidence grounding score based on citation density and chunk relevance."""
        if not retrieved_chunks:
            return {
                "confidence_score": 0.20,
                "is_grounded": False,
                "abstain": True
            }

        # Check presence of legal citations in generated answer
        citation_matches = re.findall(
            r'(\[.*?\]|Section\s+\d+|Rule\s+\d+|Article\s+\d+|First Schedule|Schedule\s+[A-Z])',
            answer_text,
            re.IGNORECASE
        )
        citation_count = len(citation_matches)
        
        base_score = 0.65
        if citation_count >= 2:
            base_score += 0.20
        elif citation_count == 1:
            base_score += 0.10

        # Check rerank / hybrid scores of retrieved chunks
        top_score = max([c.get("rerank_score", c.get("hybrid_rrf_score", 0.5)) for c in retrieved_chunks])
        if top_score > 0:
            base_score = min(0.95, base_score + 0.05)

        confidence_score = round(base_score, 2)
        is_grounded = confidence_score >= settings.CONFIDENCE_THRESHOLD

        return {
            "confidence_score": confidence_score,
            "is_grounded": is_grounded,
            "abstain": not is_grounded
        }

    @classmethod
    def inject_disclaimer(cls, text: str) -> str:
        """Enforces statutory standing disclaimer on generated guidance."""
        disclaimer = f"\n\n---\n⚖️ **Legal Notice**: {settings.LEGAL_DISCLAIMER}"
        if "Legal Notice" not in text and "Disclaimer" not in text:
            return text + disclaimer
        return text
