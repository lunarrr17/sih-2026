import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from backend.app.core.config import settings
from backend.app.core.qdrant_client_provider import QdrantClientProvider
from backend.app.rag.vector_store import QdrantHybridVectorStore
from backend.app.rag.reranker import LegalCrossEncoderReranker
from backend.app.rag.pdf_loader import PDFStatutoryLoader
from backend.app.rag.indexer import QdrantCorpusIndexer

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Unified legal retrieval interface coordinating:
    1. Dense semantic search in Qdrant collections.
    2. BM25 sparse keyword search for statutory sections.
    3. Cross-Encoder reranker prioritizing legal statutory bars.
    4. Strict jurisdiction isolation ('national' vs 'international').
    """

    def __init__(self):
        self.vector_store = QdrantHybridVectorStore()
        self.reranker = LegalCrossEncoderReranker()
        self._is_initialized = False

    def initialize(self, force_reindex: bool = False):
        """Loads statutory chunks into in-memory BM25 indices and ensures Qdrant collections are populated."""
        if self._is_initialized and not force_reindex:
            return

        raw_dir = settings.DATA_DIR / "raw_documents"
        if raw_dir.exists():
            loader = PDFStatutoryLoader(chunk_size=700, chunk_overlap=100)
            chunks_dict = loader.load_all_raw_documents(raw_dir)
            nat_chunks = chunks_dict["national"]
            intl_chunks = chunks_dict["international"]

            self.vector_store.register_chunks_for_sparse_search(
                national_chunks=nat_chunks,
                international_chunks=intl_chunks
            )

            # Check if Qdrant collections need initial vector population
            client = self.vector_store.client
            try:
                existing_cols = [c.name for c in client.get_collections().collections]
                needs_index = (
                    settings.QDRANT_COLLECTION_NATIONAL not in existing_cols or
                    client.get_collection(settings.QDRANT_COLLECTION_NATIONAL).points_count == 0
                )
                if needs_index and nat_chunks:
                    logger.info("🌿 Initializing Qdrant collections with dense statutory vectors...")
                    indexer = QdrantCorpusIndexer(client=client, embedder=self.vector_store.embedder)
                    nat_emb = self.vector_store.in_memory_embeddings.get("national")
                    intl_emb = self.vector_store.in_memory_embeddings.get("international")
                    indexer.index_chunks(nat_chunks, settings.QDRANT_COLLECTION_NATIONAL, batch_size=256, precomputed_embeddings=nat_emb)
                    indexer.index_chunks(intl_chunks, settings.QDRANT_COLLECTION_INTERNATIONAL, batch_size=256, precomputed_embeddings=intl_emb)
                    logger.info("✅ Qdrant collections populated with dense vectors.")
            except Exception as e:
                logger.warning(f"Could not connect to Qdrant for initial indexing ({e}). Using in-memory fallback.")

        self._is_initialized = True

    def _apply_relevance_gate(
        self,
        candidates: List[Dict[str, Any]],
        query: str = "",
        jurisdiction: str = "national",
        is_comparative: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Filters candidate evidence against statutory relevance thresholds, strict jurisdiction
        boundaries, section/rule precision, and query-to-evidence concept alignment.
        If the evidence fails any of these criteria, returns empty list to trigger safe abstention.
        """
        if not candidates:
            return []

        # 1. Enforce strict jurisdiction isolation
        effective_jur = jurisdiction.lower()
        if effective_jur in ["national", "international"]:
            candidates = [c for c in candidates if c.get("jurisdiction", "").lower() == effective_jur]
            if not candidates:
                logger.info(f"🛑 Relevance Gate: All candidates discarded due to jurisdiction mismatch ({effective_jur}). Abstaining.")
                return []

        # 2. Check if cross-encoder scores are present
        has_section_match = any(c.get("is_section_match") for c in candidates)
        has_rerank_scores = any("rerank_score" in c for c in candidates)
        if has_rerank_scores and not has_section_match:
            top_score = max(c.get("rerank_score", -99.0) for c in candidates)
            if top_score < settings.RETRIEVAL_MIN_RERANK_SCORE:
                logger.info(f"🛑 Relevance Gate: Top rerank score {top_score} < threshold {settings.RETRIEVAL_MIN_RERANK_SCORE}. Abstaining.")
                return []
            # Filter individual candidates whose scores are far below the cutoff (never filter explicit section matches)
            candidates = [c for c in candidates if c.get("is_section_match") or c.get("rerank_score") is None or c.get("rerank_score", 0.0) >= settings.RETRIEVAL_MIN_RERANK_SCORE - 2.5]
        elif has_rerank_scores and has_section_match:
            candidates = [c for c in candidates if c.get("is_section_match") or c.get("rerank_score") is None or c.get("rerank_score", 0.0) >= settings.RETRIEVAL_MIN_RERANK_SCORE - 2.5]
        else:
            # Dense cosine similarity or BM25 fallback gate
            top_dense = max(c.get("dense_score", 0.0) for c in candidates)
            top_bm25 = max(c.get("raw_bm25_score", 0.0) for c in candidates)

            if top_dense < settings.RETRIEVAL_MIN_SIMILARITY and top_bm25 < settings.RETRIEVAL_MIN_BM25_SCORE:
                logger.info(f"🛑 Relevance Gate: Top dense {top_dense} and BM25 {top_bm25} below thresholds. Abstaining.")
                return []

        if not candidates:
            return []

        # 3. Document-Level Scoping: If user specifically asks about a single named document,
        # restrict candidates to that document (prevents cross-document term stitching hallucinations)
        if query:
            q_lower = query.lower()
            doc_name_map = {
                'nagoya': 'Nagoya_Protocol_ABS.pdf',
                'wipo': 'WIPO_GRATK_Treaty_2024.pdf',
                'gratk': 'WIPO_GRATK_Treaty_2024.pdf',
                'trips': 'WTO_TRIPS_Agreement.pdf',
                'patents act': 'Patents_Act_1970.PDF',
                'patent act': 'Patents_Act_1970.PDF',
                'patent statute': 'Patents_Act_1970.PDF',
                'indian patent statute': 'Patents_Act_1970.PDF',
                'section 3(d)': 'Patents_Act_1970.PDF',
                'section 3(e)': 'Patents_Act_1970.PDF',
                'section 3(p)': 'Patents_Act_1970.PDF',
                'section 3': 'Patents_Act_1970.PDF',
                'section 2(1)(j)': 'Patents_Act_1970.PDF',
                'section 2': 'Patents_Act_1970.PDF',
                'section 64': 'Patents_Act_1970.PDF',
                'section 25': 'Patents_Act_1970.PDF',
                'patent rules 2024': 'Patent_Amendment_Rules_2024.pdf',
                'patents rules': 'Patent_Amendment_Rules_2024.pdf',
                'patent amendment rules': 'Patent_Amendment_Rules_2024.pdf',
                'drugs and cosmetics': 'Drugs_and_Cosmetics_Act_Ayurveda.pdf',
                'drug rules': 'Drugs_and_Cosmetics_Act_Ayurveda.pdf',
                'drugs rules': 'Drugs_and_Cosmetics_Act_Ayurveda.pdf',
                'd&c act': 'Drugs_and_Cosmetics_Act_Ayurveda.pdf',
                'd&c rules': 'Drugs_and_Cosmetics_Act_Ayurveda.pdf',
                'biological diversity': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'biodiversity act': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'nba': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'sbb': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'national biodiversity authority': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'state biodiversity board': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'ayurveda aahar': 'FSSAI_Ayurveda_Aahar_Regulations_2022.pdf',
                'fssai': 'FSSAI_Ayurveda_Aahar_Regulations_2022.pdf'
            }
            if not any(comp in q_lower for comp in ['comparison', 'comparative', 'compare', 'both', 'versus', 'vs', 'under indian law and']):
                for doc_key, doc_file in doc_name_map.items():
                    if doc_key in q_lower:
                        doc_scoped = [c for c in candidates if c.get('document_name') == doc_file]
                        if doc_scoped:
                            candidates = doc_scoped
                        break

        # 4. Section / Rule / Article Precision: Prioritize exact section matches if specified in query
        if query:
            q_lower = query.lower()
            if effective_jur == "national":
                sec_matches = re.findall(r'\b(?:section|rule)\s+([0-9]+[a-z0-9\(\)]*)|\b([0-9]+[\(][a-z0-9]+[\)])', q_lower)
            elif effective_jur == "international":
                sec_matches = re.findall(r'\barticle\s+([0-9]+[a-z0-9\(\)]*)', q_lower)
            else:
                sec_matches = re.findall(r'\b(?:section|rule|article)\s+([0-9]+[a-z0-9\(\)]*)|\b([0-9]+[\(][a-z0-9]+[\)])', q_lower)

            target_sections = set()
            for sm in sec_matches:
                target_sections.update([s for s in sm if s])

            if target_sections:
                matching_section_chunks = []
                other_chunks = []
                for c in candidates:
                    sec_clause = c.get("section_or_clause", "").lower()
                    text_lower = c.get("text", "").lower()
                    matched = False
                    for ts in target_sections:
                        if ts in sec_clause or ts in text_lower:
                            matched = True
                            break
                        m_sub = re.match(r'^([0-9]+)[\(]?([a-z0-9]+)[\)]?$', ts)
                        if m_sub:
                            s_num, s_clause = m_sub.groups()
                            if s_num in sec_clause or f"section {s_num}" in text_lower or f"{s_num}." in text_lower:
                                if f"({s_clause})" in text_lower or f"clause ({s_clause})" in text_lower or s_clause in sec_clause:
                                    matched = True
                                    break
                    if matched:
                        matching_section_chunks.append(c)
                    else:
                        other_chunks.append(c)
                if matching_section_chunks:
                    candidates = matching_section_chunks + other_chunks

        # 4. Prioritize primary statutes over secondary academic studies for core statutory questions
        is_statutory_inquiry = any(w in query.lower() for w in [
            "patent", "patented", "patentable", "section", "provision", "provisions",
            "revocation", "opposition", "rule", "act", "prohibited", "bar", "traditional knowledge"
        ])
        if is_statutory_inquiry and not any(acad in query.lower() for acad in ["academic", "study", "iosr", "scholarly", "guidelines discussion"]):
            primary_chunks = [c for c in candidates if "Traditional_Knowledge_Guidelines" not in c.get("document_name", "") and c.get("source_type") != "secondary_academic_study" and c.get("authority_level") != "secondary_academic_study"]
            secondary_chunks = [c for c in candidates if c not in primary_chunks]
            if primary_chunks:
                # If primary chunks exist, do not dilute with secondary academic studies
                candidates = primary_chunks if len(primary_chunks) >= 2 else (primary_chunks + secondary_chunks)

        # 5. Query-to-Evidence Concept Alignment Gate
        if query:
            from backend.app.core.guardrails import GuardrailsEngine
            check_jur = "comparative" if is_comparative else effective_jur
            alignment = GuardrailsEngine.verify_query_evidence_alignment(query, candidates, jurisdiction=check_jur)
            if not alignment.get("aligned", True):
                logger.info(f"🛑 Relevance Gate: Concept Alignment Gate failed ({alignment.get('reason')}). Discarding candidates to force safe abstention.")
                return []

        return candidates

    def decompose_query(self, query: str, jurisdiction: str = "national") -> List[Dict[str, str]]:
        """
        Conservative multi-concept query decomposer.
        Detects queries requiring multi-hop evidence across distinct statutory domains.
        Returns sub-queries for independent retrieval.
        """
        q_lower = query.lower()
        dimensions = []

        # 1. Patent application official fees
        fee_cues = ["fee", "fees", "cost", "filing fee", "patent fee", "fees for patent", "how much does a patent cost", "inr fee", "fee schedule"]
        is_fee_query = any(f in q_lower for f in fee_cues)
        if is_fee_query:
            dimensions.append({
                "dimension": "patent_fees",
                "subquery": "official patent application filing fee schedule Indian Patent Office First Schedule"
            })

        # 2. Classical text / formulation status
        classical_cues = ["charaka", "sushruta", "ashtanga", "first schedule", "samhita", "classical text", "classical texts", "classical books", "classical rules", "classical vs", "classical versus", "classical medicine under the first schedule"]
        if any(c in q_lower for c in classical_cues) or ("classical" in q_lower and any(d in q_lower for d in ["definition", "distinction", "status", "licensing", "proprietary"])):
            dimensions.append({
                "dimension": "classical_status",
                "subquery": "Drugs and Cosmetics Act First Schedule recognized classical texts Charaka Samhita Rule 158B classical Ayurvedic formulation definition"
            })

        # 3. Patentability / traditional knowledge / IP
        # Note: 'patent or proprietary' (or 'patent and proprietary') is a statutory AYUSH drug category under D&C Rule 158B,
        # NOT a Patent Act inquiry unless patent eligibility/grant/bar is also specifically raised.
        q_patent_check = q_lower.replace("patent or proprietary", "").replace("patent and proprietary", "")
        is_academic_query = any(acad in q_lower for acad in ["academic", "study", "iosr", "scholarly"])
        is_tk_patent_inquiry = any(tk in q_lower for tk in ["traditional knowledge", "traditional", "ancient", "herbal", "ayurved", "classical"]) and any(p in q_patent_check for p in ["patent", "patentable", "patented", "patentability", "exclusivity"]) and not any(other in q_lower for other in ["fee", "rules, 2024", "rules 2024", "form 27", "working"])
        patent_substantive_cues = ["patentability", "patentable", "can i patent", "is it patentable", "section 3", "3(p)", "3(e)", "3(d)", "novelty", "inventive step", "mere admixture", "traditional knowledge", "ip issues", "ip protection", "intellectual property", "ipr"]
        has_patent_word = any(p in q_patent_check for p in ["patent", "patenting", "ip issues", "ip protection", "intellectual property", "ipr"]) or bool(re.search(r'\bip\b', q_patent_check))
        if (any(p in q_lower for p in patent_substantive_cues) or has_patent_word or is_tk_patent_inquiry) and not is_fee_query and not is_academic_query and not any(r in q_lower for r in ["form 27", "working of patent", "working of patents", "commercial working"]):
            has_3d = any(d in q_lower for d in ["section 3(d)", "3(d)", "new form of known substance", "new form of a known substance"])
            has_3e = any(e in q_lower for e in ["section 3(e)", "3(e)", "mere admixture", "aggregation of properties", "aggregation of the properties", "synergy", "synergistic", "summation"])
            has_3p = any(p in q_lower for p in ["section 3(p)", "3(p)", "traditional knowledge"])
            has_patentability_inquiry = (
                any(w in q_lower for w in ["patentable", "patentability", "patent", "patenting", "grant", "guarantee", "qualify", "protection"])
                and not any(neg in q_lower for neg in ["non-patentable", "not patentable", "what is excluded", "what does section 3", "what are the categories"])
                and any(w in q_lower for w in ["can", "is", "does", "would", "could", "avoid", "mean", "guarantee", "qualify", "receive", "obtain", "available", "consider", "immediately"])
            )

            if any(rev in q_lower for rev in ["revocation", "revoked", "revoke", "section 64"]):
                subq = "Patents Act 1970 Section 64 revocation of patent grounds geographical origin biological material traditional knowledge"
            elif any(opp in q_lower for opp in ["opposition", "oppose", "section 25"]):
                subq = "Patents Act 1970 Section 25 opposition to the grant of patent grounds geographical origin biological material"
            elif any(re.search(pat, q_lower) for pat in [r'\bsection\s+2\b', r'\bsection\s+2\(1\)', r'\b2\(1\)\(j\)', r'\binventive step\b', r'\bdefinition of invention\b']):
                subq = "Patents Act 1970 Section 2 definitions invention inventive step capable of industrial application"
            elif has_3d and not has_3p and not has_3e:
                subq = "Patents Act 1970 Section 3(d) new form of known substance enhancement of known efficacy mere use of a known process"
            elif has_3e and not has_3p and not has_3d:
                subq = "Patents Act 1970 Section 3(e) substance obtained by mere admixture aggregation of properties process for producing such substance"
            elif has_3p and not has_3d and not has_3e and not has_patentability_inquiry:
                subq = "Patents Act 1970 Section 3(p) traditional knowledge exclusion aggregation or duplication of known properties"
            elif "section 3" in q_lower and not has_3d and not has_3e and not has_3p and not has_patentability_inquiry:
                subq = "Patents Act 1970 Section 3 Chapter II Inventions not patentable The following are not inventions within the meaning of this Act"
            elif has_patentability_inquiry and not has_3d and not has_3e:
                subq = "Patents Act 1970 Section 2(1)(j) definition of invention novelty inventive step Section 3(p) traditional knowledge"
            elif any(m in q_lower for m in ["modified", "modification", "new form", "efficacy"]) and not has_3e:
                subq = "Patents Act 1970 Section 3(d) new form of known substance enhancement of known efficacy Section 3(p) traditional knowledge"
            elif any(mix in q_lower for mix in ["combined", "combination", "mixture", "admixture", "ingredients into a new"]):
                subq = "Patents Act 1970 Section 3(e) mere admixture resulting only in aggregation of properties components thereof"
            else:
                subq = "Patents Act 1970 Section 3(p) traditional knowledge exclusion Section 3(e) mere admixture not patentable inventions"

            dimensions.append({
                "dimension": "patents_patentability",
                "subquery": subq
            })

        # 4. Biodiversity / ABS prior approval
        bio_cues = ["biodiversity", "abs", "benefit-sharing", "benefit sharing", "nba", "national biodiversity authority", "sbb", "section 6", "section 7", "ayush practitioner", "medicinal plants", "biological resource", "biological diversity"]
        if any(b in q_lower for b in bio_cues):
            subq = "Biological Diversity Act prior approval National Biodiversity Authority commercial utilization access benefit sharing"
            if "section 6" in q_lower or "patent" in q_lower or "intellectual property" in q_lower or "ipr" in q_lower:
                subq = "Biological Diversity Act section 6 applying for an intellectual property right patent approval National Biodiversity Authority"
            dimensions.append({
                "dimension": "biodiversity_abs",
                "subquery": subq
            })

        # 5. Food safety / Ayurveda Aahar
        aahar_cues = ["ayurveda aahar", "fssai", "dietary supplement", "dietary supplements", "food supplement", "food supplements", "supplement", "supplements", "food safety", "disease risk reduction"]
        if any(a in q_lower for a in aahar_cues):
            dimensions.append({
                "dimension": "ayurveda_aahar",
                "subquery": "Food Safety and Standards Ayurveda Aahar Regulations 2022 Regulation 4 Schedule A disease claims"
            })

        # 6. Drug manufacturing / Rule 161 labelling
        label_cues = ["rule 161", "rule 158b", "labelling", "label requirements", "packaging", "manufacturing license", "drugs and cosmetics", "d&c act", "d&c rules"]
        if any(l in q_lower for l in label_cues):
            dimensions.append({
                "dimension": "regulatory_licensing",
                "subquery": "Drugs and Cosmetics Rules Rule 161 labelling requirements Rule 158B licensing Ayurvedic drugs"
            })

        # Single named statute scoping check:
        # If the query explicitly designates a single statute (e.g. "Under the Biological Diversity Act..."),
        # restrict retrieval to that governing statute unless multi-statute inquiry is explicit
        has_dc = any(d in q_lower for d in ["drugs and cosmetics", "d&c", "first schedule", "rule 158b", "rule 161", "drug rules"])
        has_patent = any(p in q_lower for p in ["patents act", "patent act", "section 3", "3(p)", "3(e)", "3(d)", "patentability", "non-patentability", "patent exclusivity"])
        has_bd = any(b in q_lower for b in ["biological diversity", "bd act", "nba", "sbb"])
        has_fssai = any(f in q_lower for f in ["fssai", "ayurveda aahar"])
        statutes_count = sum([1 for flag in [has_dc, has_patent, has_bd, has_fssai] if flag])
        is_multi_statute = (statutes_count >= 2) or any(m in q_lower for m in ["which statutes", "across", "and both", "and seeks both", "cumulative", "governing statutes", "versus", "compare", "under indian law and"])

        if not is_multi_statute:
            if any(bd in q_lower for bd in ["biological diversity act", "bd act", "biodiversity act"]):
                if not any(pat in q_lower for pat in ["patents act", "patent act", "section 3(p)", "section 3(d)", "section 3(e)"]):
                    dimensions = [d for d in dimensions if d["dimension"] == "biodiversity_abs"]
            elif "under the drugs and cosmetics" in q_lower or "under rule 161" in q_lower or "under rule 158b" in q_lower or "according to drug rules" in q_lower:
                dimensions = [d for d in dimensions if d["dimension"] in ["regulatory_licensing", "classical_status"]]
            elif "under section 3" in q_lower or "under the patents act" in q_lower or any(rev in q_lower for rev in ["revocation", "opposition", "section 64", "section 25"]):
                dimensions = [d for d in dimensions if d["dimension"] in ["patents_patentability", "patent_fees"]]

        return dimensions if (
            len(dimensions) >= 2
            or is_tk_patent_inquiry
            or (len(dimensions) == 1 and any(b in q_lower for b in ["nba", "sbb", "biodiversity"]))
            or (len(dimensions) == 1 and dimensions[0]["dimension"] in ["patents_patentability", "patent_fees"])
        ) else []

    def _partition_explicit_sections(self, query: str, candidates: List[Dict[str, Any]], jurisdiction: str = "national") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Partitions candidates into those matching an explicit statutory section and others, strictly respecting jurisdiction."""
        q_lower = query.lower()
        if jurisdiction == "international":
            explicit_sec_match = re.search(r'\b(article\s+\d+[a-zA-Z\(\)]*)\b', q_lower)
        else:
            explicit_sec_match = re.search(r'\b(section\s+\d+[a-zA-Z\(\)]*|rule\s+\d+[a-zA-Z\(\)]*|regulation\s+\d+[a-zA-Z\(\)]*)\b', q_lower)

        if not explicit_sec_match:
            return [], candidates

        target = explicit_sec_match.group(1).replace(" ", "")
        matching_title = []
        matching_body = []
        others = []
        clean_num = re.sub(r'^(section|rule|article|regulation)', '', target)

        for c in candidates:
            # Prevent Patent_Amendment_Rules_2024.pdf gazette / form text from matching Section 3
            if "patent_amendment_rules" in str(c.get("document_name", "")).lower() and clean_num.startswith("3"):
                others.append(c)
                continue

            sec_clause = str(c.get("section_or_clause", "")).lower().replace(" ", "")
            text_prefix = c.get("text", "")[:250].lower().replace(" ", "")

            # Require exact clause match or word-bounded number match within proper provision
            matched_in_title = False
            matched_in_body = False

            if target in sec_clause:
                matched_in_title = True
            elif target in text_prefix:
                matched_in_body = True
            elif clean_num:
                # Check for exact provision match with word boundaries
                num_pat = rf'\b{re.escape(clean_num)}\b'
                raw_sec = str(c.get("section_or_clause", "")).lower()
                raw_text = c.get("text", "")[:250].lower()

                if jurisdiction == "international" and "article" in raw_sec:
                    if re.search(num_pat, raw_sec):
                        matched_in_title = True
                elif jurisdiction != "international":
                    if ("section" in raw_sec or "rule" in raw_sec or "regulation" in raw_sec) and re.search(num_pat, raw_sec):
                        matched_in_title = True
                    elif re.search(rf'section\s+{re.escape(clean_num)}\b', raw_text) or re.search(rf'rule\s+{re.escape(clean_num)}\b', raw_text) or re.search(rf'"{re.escape(clean_num)}\.', raw_text):
                        matched_in_body = True

            if matched_in_title:
                c["is_section_match"] = True
                matching_title.append(c)
            elif matched_in_body:
                c["is_section_match"] = True
                matching_body.append(c)
            else:
                others.append(c)

        matching = matching_title + matching_body
        if matching:
            doc_target = matching[0].get("document_name")
            if doc_target:
                same_doc_others = [c for c in others if c.get("document_name") == doc_target]
                diff_doc_others = [c for c in others if c.get("document_name") != doc_target]
                others = same_doc_others + diff_doc_others

        return matching, others

    def _search_single_query(
        self,
        query: str,
        jurisdiction: str = "national",
        top_k: int = 4,
        enable_reranking: bool = True
    ) -> List[Dict[str, Any]]:
        """Executes search for a single focused query."""
        raw_candidates = self.vector_store.hybrid_search(query, jurisdiction=jurisdiction, top_k=max(top_k * 6, 30))
        if not raw_candidates:
            return []

        # Early Document Scoping for targeted single-statute inquiries
        if query:
            q_lower = query.lower()
            doc_name_map = {
                'nagoya': 'Nagoya_Protocol_ABS.pdf',
                'wipo': 'WIPO_GRATK_Treaty_2024.pdf',
                'gratk': 'WIPO_GRATK_Treaty_2024.pdf',
                'trips': 'WTO_TRIPS_Agreement.pdf',
                'patents act': 'Patents_Act_1970.PDF',
                'patent act': 'Patents_Act_1970.PDF',
                'patent statute': 'Patents_Act_1970.PDF',
                'indian patent statute': 'Patents_Act_1970.PDF',
                'section 3(d)': 'Patents_Act_1970.PDF',
                'section 3(e)': 'Patents_Act_1970.PDF',
                'section 3(p)': 'Patents_Act_1970.PDF',
                'section 3': 'Patents_Act_1970.PDF',
                'section 2(1)(j)': 'Patents_Act_1970.PDF',
                'section 2': 'Patents_Act_1970.PDF',
                'section 64': 'Patents_Act_1970.PDF',
                'section 25': 'Patents_Act_1970.PDF',
                'patent rules 2024': 'Patent_Amendment_Rules_2024.pdf',
                'patents rules': 'Patent_Amendment_Rules_2024.pdf',
                'patent amendment rules': 'Patent_Amendment_Rules_2024.pdf',
                'drugs and cosmetics': 'Drugs_and_Cosmetics_Act_Ayurveda.pdf',
                'drug rules': 'Drugs_and_Cosmetics_Act_Ayurveda.pdf',
                'drugs rules': 'Drugs_and_Cosmetics_Act_Ayurveda.pdf',
                'd&c act': 'Drugs_and_Cosmetics_Act_Ayurveda.pdf',
                'd&c rules': 'Drugs_and_Cosmetics_Act_Ayurveda.pdf',
                'biological diversity': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'biodiversity act': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'nba': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'sbb': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'national biodiversity authority': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'state biodiversity board': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'ayurveda aahar': 'FSSAI_Ayurveda_Aahar_Regulations_2022.pdf',
                'fssai': 'FSSAI_Ayurveda_Aahar_Regulations_2022.pdf'
            }
            if not any(comp in q_lower for comp in ['comparison', 'comparative', 'compare', 'contrast', 'both', 'versus', 'vs', 'under indian law and', 'differ', 'difference']):
                for doc_key, doc_file in doc_name_map.items():
                    if re.search(rf'\b{re.escape(doc_key)}\b', q_lower):
                        doc_scoped = [c for c in raw_candidates if c.get('document_name') == doc_file]
                        if len(doc_scoped) >= 2:
                            raw_candidates = doc_scoped
                        break

        matching, others = self._partition_explicit_sections(query, raw_candidates, jurisdiction=jurisdiction)

        # Prioritize primary statutes over secondary academic studies before reranking
        is_statutory_inquiry = any(w in query.lower() for w in [
            "patent", "patented", "patentable", "section", "provision", "provisions",
            "revocation", "opposition", "rule", "act", "prohibited", "bar", "traditional knowledge"
        ])
        if is_statutory_inquiry and not any(acad in query.lower() for acad in ["academic", "study", "iosr", "scholarly", "guidelines discussion"]):
            primary_others = [
                c for c in others
                if "Traditional_Knowledge_Guidelines" not in c.get("document_name", "")
                and c.get("source_type") != "secondary_academic_study"
                and c.get("authority_level") != "secondary_academic_study"
            ]
            if len(primary_others) >= 4:
                others = primary_others

        if enable_reranking:
            reranked_results = self.reranker.rerank(query, others, top_n=top_k)
            combined = matching[:top_k] + [c for c in reranked_results if c not in matching]
            return self._apply_relevance_gate(combined[:top_k], query=query, jurisdiction=jurisdiction)

        combined = matching + others
        return self._apply_relevance_gate(combined[:top_k], query=query, jurisdiction=jurisdiction)

    def search(
        self,
        query: str,
        jurisdiction: str = "national",
        top_k: int = 5,
        enable_reranking: bool = True
    ) -> Any:
        """
        Executes hybrid search + cross-encoder reranking + evidence relevance gating.
        Supports conservative multi-concept query decomposition.
        jurisdiction options: 'national', 'international', 'comparative'
        """
        if not self._is_initialized:
            self.initialize()

        target_jur = jurisdiction.lower()

        # Handle comparative dual retrieval
        if target_jur in ["comparative", "both"]:
            nat_candidates = self.vector_store.hybrid_search(query, jurisdiction="national", top_k=max(top_k * 6, 30))
            intl_candidates = self.vector_store.hybrid_search(query, jurisdiction="international", top_k=max(top_k * 6, 30))

            q_lower = query.lower()
            nat_doc_map = {
                'biological diversity': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'biodiversity': 'Biological_Diversity_Amendment_Act_2023.pdf',
                'patents act': 'Patents_Act_1970.PDF',
                'patent act': 'Patents_Act_1970.PDF',
                'drugs and cosmetics': 'Drugs_and_Cosmetics_Act_Ayurveda.pdf',
                'ayurveda aahar': 'FSSAI_Ayurveda_Aahar_Regulations_2022.pdf'
            }
            for doc_key, doc_file in nat_doc_map.items():
                if doc_key in q_lower:
                    scoped = [c for c in nat_candidates if c.get('document_name') == doc_file]
                    if len(scoped) >= 2:
                        nat_candidates = scoped
                    break

            intl_doc_map = {
                'wipo': 'WIPO_GRATK_Treaty_2024.pdf',
                'gratk': 'WIPO_GRATK_Treaty_2024.pdf',
                'nagoya': 'Nagoya_Protocol_ABS.pdf',
                'trips': 'WTO_TRIPS_Agreement.pdf'
            }
            for doc_key, doc_file in intl_doc_map.items():
                if doc_key in q_lower:
                    scoped = [c for c in intl_candidates if c.get('document_name') == doc_file]
                    if len(scoped) >= 2:
                        intl_candidates = scoped
                    break

            nat_matching, nat_others = self._partition_explicit_sections(query, nat_candidates, jurisdiction="national")
            intl_matching, intl_others = self._partition_explicit_sections(query, intl_candidates, jurisdiction="international")

            if enable_reranking:
                nat_reranked = self.reranker.rerank(query, nat_others, top_n=top_k)
                intl_reranked = self.reranker.rerank(query, intl_others, top_n=top_k)
                nat_top = nat_matching[:top_k] + [c for c in nat_reranked if c not in nat_matching]
                intl_top = intl_matching[:top_k] + [c for c in intl_reranked if c not in intl_matching]
            else:
                nat_top = (nat_matching + nat_others)[:top_k]
                intl_top = (intl_matching + intl_others)[:top_k]

            return {
                "national": self._apply_relevance_gate(nat_top[:top_k], query=query, jurisdiction="national", is_comparative=True),
                "international": self._apply_relevance_gate(intl_top[:top_k], query=query, jurisdiction="international", is_comparative=True)
            }

        # Single jurisdiction retrieval
        effective_jur = "national" if target_jur in ["national", "india"] else "international"

        # Check if concept-tailored query decomposition applies (National jurisdiction only)
        if effective_jur == "national":
            decomposed = self.decompose_query(query, jurisdiction=effective_jur)
            if decomposed:
                per_dim_k = max(2, top_k // len(decomposed)) if len(decomposed) > 1 else top_k
                logger.info(f"🔀 Concept-tailored query detected ({len(decomposed)} dimensions). Executing sub-retrievals...")
                combined_chunks: List[Dict[str, Any]] = []
                seen_chunk_ids = set()
                supported_dims: List[str] = []
                unsupported_dims: List[str] = []

                for sub in decomposed:
                    sub_candidates = self._search_single_query(
                        query=sub["subquery"],
                        jurisdiction=effective_jur,
                        top_k=per_dim_k,
                        enable_reranking=enable_reranking
                    )
                    if sub_candidates:
                        supported_dims.append(sub["dimension"])
                        for c in sub_candidates:
                            cid = c.get("chunk_id", c.get("text", "")[:40])
                            if cid not in seen_chunk_ids:
                                seen_chunk_ids.add(cid)
                                c_copy = dict(c)
                                c_copy["decomposed_dimension"] = sub["dimension"]
                                combined_chunks.append(c_copy)
                    else:
                        unsupported_dims.append(sub["dimension"])

                self.last_decomposition = {
                    "is_decomposed": True,
                    "supported_dimensions": supported_dims,
                    "unsupported_dimensions": unsupported_dims,
                    "subqueries": [s["subquery"] for s in decomposed]
                }

                if not combined_chunks:
                    return []

                return combined_chunks

            elif len(decomposed) == 1:
                # Single-dimension query where decomposition formulated a targeted subquery
                # (e.g. patents_patentability or biodiversity_abs)
                sub_candidates = self._search_single_query(
                    query=decomposed[0]["subquery"],
                    jurisdiction=effective_jur,
                    top_k=top_k,
                    enable_reranking=enable_reranking
                )
                orig_candidates = self._search_single_query(
                    query=query,
                    jurisdiction=effective_jur,
                    top_k=top_k,
                    enable_reranking=enable_reranking
                )
                combined = sub_candidates + [c for c in orig_candidates if c not in sub_candidates]
                return self._apply_relevance_gate(combined[:top_k], query=query, jurisdiction=effective_jur)

        self.last_decomposition = {
            "is_decomposed": False,
            "supported_dimensions": [],
            "unsupported_dimensions": [],
            "subqueries": []
        }
        return self._search_single_query(query, jurisdiction=effective_jur, top_k=top_k, enable_reranking=enable_reranking)

# Global singleton instance
retriever = HybridRetriever()
