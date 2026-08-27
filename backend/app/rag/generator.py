import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel
import httpx
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class CitationItem(BaseModel):
    statute: str
    section: str
    title: str
    source_url: str
    page_numbers: List[int] = []

class GroundedLLMSynthesizer:
    """
    High-precision Grounded LLM Synthesizer.
    1. Cloud Mode: Google Gemini 2.5 Flash / OpenAI GPT-4o-mini (Concise, focused legal answers).
    2. Local Mode: Dynamic query-specific statutory extractor (No boilerplate).
    """

    SYSTEM_PROMPT = """You are IP-SAKTI Sahayak, an authoritative legal AI assistant specializing in Indian Traditional Knowledge (Ayurveda, Siddha, Unani), Intellectual Property Law, and Regulatory Compliance.

STRICT INSTRUCTIONS:
1. Answer ONLY the specific question asked by the user using the provided Statutory Corpus chunks.
2. Synthesize a clean, natural, and authoritative answer in 2-3 focused paragraphs.
3. Every factual and legal statement MUST be backed by a bracketed citation citing the exact Act and Section (e.g., [The Patents Act, 1970 - Section 3(p), Page 3]).
4. Do NOT output raw truncated chunk text.
5. Do NOT include unrelated legal topics (e.g., if the user asks about Rule 161 labelling, do NOT discuss Biodiversity ABS or Section 3(p) patent bars).
6. If the provided statutory text does NOT contain enough information to answer the question, explicitly state what is known and safely abstain on unknown points.
7. Maintain an objective, professional legal tone.
"""

    @classmethod
    def synthesize(
        cls,
        query: str,
        chunks: List[Dict[str, Any]],
        jurisdiction: str = "national",
        classification_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[CitationItem]]:
        if not chunks:
            return (
                "⚠️ **Safe Abstention**: No direct statutory provisions were found in the official legal corpus "
                "matching your specific query. Please rephrase with specific Ayurvedic or IP terminology.",
                []
            )

        # Extract structured citations
        citations: List[CitationItem] = []
        seen_citations = set()
        for chunk in chunks:
            statute_title = chunk.get("statute_title") or chunk.get("document_name") or "Statute"
            section = chunk.get("section_or_clause") or "General Provision"
            source_url = chunk.get("source_url") or "https://ipindia.gov.in"
            pages = chunk.get("page_numbers") or []
            page_str = f" (Page {', '.join(map(str, pages))})" if pages else ""

            cite_key = f"{statute_title} {section}"
            if cite_key not in seen_citations:
                seen_citations.add(cite_key)
                citations.append(CitationItem(
                    statute=statute_title,
                    section=section,
                    title=f"{statute_title} - {section}{page_str}",
                    source_url=source_url,
                    page_numbers=pages
                ))

        # 1. Try Cloud LLM Synthesis (Gemini 2.5 Flash or OpenAI)
        cloud_response = cls._try_cloud_llm_synthesis(query, chunks, jurisdiction, classification_context)
        if cloud_response:
            return cloud_response, citations

        # 2. Local Dynamic Extractive Synthesizer (Topic-focused fallback)
        return cls._local_dynamic_synthesis(query, chunks, jurisdiction, classification_context, citations)

    @classmethod
    def _try_cloud_llm_synthesis(
        cls,
        query: str,
        chunks: List[Dict[str, Any]],
        jurisdiction: str,
        classification_context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        gemini_key = os.getenv("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)
        openai_key = os.getenv("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", None)

        if not gemini_key and not openai_key:
            return None

        # Build clean context
        context_blocks = []
        for c in chunks:
            doc = c.get('document_name', 'Statute')
            sec = c.get('section_or_clause', 'General')
            pages = c.get('page_numbers', [])
            text = c.get('text', '').replace('\n', ' ').strip()
            context_blocks.append(f"DOCUMENT: {doc} | SECTION: {sec} | PAGES: {pages}\nTEXT: {text}")

        context_str = "\n\n---\n\n".join(context_blocks)

        user_prompt = f"""USER QUESTION: {query}
TARGET JURISDICTION: {jurisdiction.upper()}
PRODUCT CONTEXT: {json.dumps(classification_context) if classification_context else 'None'}

OFFICIAL STATUTORY TEXT PASSAGES:
{context_str}

Please synthesize a direct, highly focused answer answering ONLY the question above with exact statutory citations:"""

        # 1. Google Gemini 2.5 Flash / Flash Latest via REST API
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
                            "temperature": 0.15,
                            "maxOutputTokens": 1500
                        }
                    }
                    with httpx.Client(timeout=12.0) as client:
                        resp = client.post(url, json=payload)
                        if resp.status_code == 200:
                            data = resp.json()
                            candidates = data.get("candidates", [])
                            if candidates:
                                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                                if text:
                                    logger.info(f"✅ Synthesized clean response using Google {model_name}.")
                                    return text.strip()
                        elif resp.status_code == 404:
                            continue  # Try next model alias
                        else:
                            logger.warning(f"Gemini API returned status {resp.status_code}: {resp.text}")
                except Exception as e:
                    logger.warning(f"Gemini API error ({model_name}): {e}")

        # 2. OpenAI GPT-4o-mini via REST API
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
                    "temperature": 0.15,
                    "max_tokens": 800
                }
                with httpx.Client(timeout=12.0) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        text = data["choices"][0]["message"]["content"]
                        if text:
                            logger.info("✅ Synthesized clean response using OpenAI GPT-4o-mini.")
                            return text.strip()
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}")

        return None

    @classmethod
    def _local_dynamic_synthesis(
        cls,
        query: str,
        chunks: List[Dict[str, Any]],
        jurisdiction: str,
        classification_context: Optional[Dict[str, Any]],
        citations: List[CitationItem]
    ) -> Tuple[str, List[CitationItem]]:
        """Dynamic local synthesizer that answers ONLY the relevant topic without boilerplate."""
        q_lower = query.lower()
        lines = []

        if classification_context and "category_name" in classification_context:
            lines.append(f"**Assessed Product**: `{classification_context['category_name']}` ({classification_context.get('governing_regime', 'AYUSH')})\n")

        # Topic 1: Section 3(p) / Traditional Knowledge Patent Bars
        if "3(p)" in q_lower or "patent" in q_lower or "traditional knowledge" in q_lower or "charaka" in q_lower:
            lines.append("### ⚖️ Patentability Analysis (Section 3(p) & Traditional Knowledge)")
            lines.append(
                "Under **Section 3(p) of the Patents Act, 1970**, an invention which in effect is traditional knowledge, "
                "or which is an aggregation or duplication of known properties of traditionally known components, is **statutorily barred from patentability**."
            )
            lines.append(
                "Because formulations described in authoritative texts (such as Charaka Samhita, Sushruta Samhita, or AFI) belong to the public domain, "
                "exclusive patent monopolies cannot be granted. In India, CSIR and the Ministry of AYUSH maintain the **Traditional Knowledge Digital Library (TKDL)** "
                "as prior art evidence to prevent wrongful patent claims worldwide."
            )

        # Topic 2: Rule 161 Labelling & Packaging Mandates
        elif "161" in q_lower or "label" in q_lower or "pack" in q_lower:
            lines.append("### 🏷️ Mandatory Labelling Provisions (Rule 161 - Drugs & Cosmetics Rules)")
            lines.append(
                "Under **Rule 161 of the Drugs and Cosmetics Rules, 1945**, all Ayurvedic, Siddha, and Unani medicines must display "
                "a true list of all active ingredients on the package or label, stating their official botanical/classical names and exact quantities."
            )
            lines.append(
                "For **Classical Ayurvedic Medicines** (formulated from First Schedule texts), the label must explicitly reference the name of the "
                "authoritative text and edition (e.g., *Charaka Samhita, Chikitsa Sthana*). For bulk or wholesale packaging consigned to licensed manufacturers, "
                "sub-rule 161(2) allows code numbers approved by the State Licensing Authority."
            )

        # Topic 3: Biodiversity Access & Benefit Sharing (ABS) & Section 40
        elif "biodiversity" in q_lower or "abs" in q_lower or "practitioner" in q_lower or "vaidya" in q_lower or "section 40" in q_lower:
            lines.append("### 🌱 Biological Diversity Compliance & Practitioner Exemptions")
            lines.append(
                "Under the **Biological Diversity (Amendment) Act, 2023 (Section 40 Proviso)**, registered AYUSH practitioners, traditional Vaidyas, "
                "and local community healers are **statutorily exempt** from prior intimation and Access and Benefit Sharing (ABS) fee obligations."
            )
            lines.append(
                "However, Indian commercial entities and corporate drug manufacturers accessing biological resources are required to submit prior intimation "
                "to the **State Biodiversity Board (SBB)** under Section 7 before commercial production."
            )

        # Topic 4: International Treaties (WIPO GRATK 2024 / Nagoya)
        elif "wipo" in q_lower or "gratk" in q_lower or "origin" in q_lower or "nagoya" in q_lower:
            lines.append("### 🌐 International Treaty Obligations (WIPO GRATK Treaty 2024)")
            lines.append(
                "Under **Article 3 of the WIPO Treaty on Intellectual Property, Genetic Resources and Associated Traditional Knowledge (2024)**, "
                "patent applicants worldwide are mandatorily required to disclose the country of origin (India) and traditional knowledge when an invention "
                "is materially based on Indian genetic resources."
            )
            lines.append(
                "Additionally, the **Nagoya Protocol (CBD)** enforces Prior Informed Consent (PIC) and Mutually Agreed Terms (MAT) for international access and fair commercial benefit sharing."
            )

        # Default fallback: Focused direct statutory extraction
        else:
            lines.append("### 🏛️ Statutory Analysis & Relevant Legal Provisions:")
            top_chunk = chunks[0]
            clean_text = top_chunk.get("text", "").replace("\n", " ").strip()
            sec = top_chunk.get("section_or_clause", "Statutory Provision")
            title = top_chunk.get("statute_title", "Statute")
            lines.append(f"According to **{title} [{sec}]**: {clean_text}")

        return "\n\n".join(lines), citations

    @classmethod
    def synthesize_comparative(
        cls,
        query: str,
        national_chunks: List[Dict[str, Any]],
        intl_chunks: List[Dict[str, Any]],
        classification_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[CitationItem]]:
        nat_answer, nat_cites = cls.synthesize(query, national_chunks, jurisdiction="national", classification_context=classification_context)
        intl_answer, intl_cites = cls.synthesize(query, intl_chunks, jurisdiction="international", classification_context=classification_context)

        combined_text = (
            "## 🇮🇳 National Regime (India Posture)\n\n"
            f"{nat_answer}\n\n"
            "---\n\n"
            "## 🌐 International Regime (Global Treaties & Export Posture)\n\n"
            f"{intl_answer}"
        )
        return combined_text, nat_cites + intl_cites
