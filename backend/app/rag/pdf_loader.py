import re
import os
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.app.core.config import settings
from backend.app.rag.schemas import LegalChunk, StatutoryMetadata, DetailedLegalStatus

# Registry mapping PDF filenames to official metadata, URLs, and granular legal status
DOCUMENT_METADATA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Patents_Act_1970.PDF": {
        "statute_title": "The Patents Act, 1970 (as amended)",
        "jurisdiction": "national",
        "official_source_url": "https://ipindia.gov.in/writereaddata/Portal/IPOAct/1_31_1_patent-act-1970-11march2015.pdf",
        "category": "classical",
        "source_type": "primary_statute",
        "authority_level": "primary_statute",
        "legal_status": "in_force (as amended)",
        "enacted_date": "1970-09-19",
        "effective_date": "1972-04-20",
        "amended_date": "2005-04-04",
        "binding_on_jurisdiction": "India (primary statute)",
        "status_source": "Gazette of India, Act No. 39 of 1970",
        "status_verified_at": "2024-03"
    },
    "Patent_Amendment_Rules_2024.pdf": {
        "statute_title": "Patents (Amendment) Rules, 2024",
        "jurisdiction": "national",
        "official_source_url": "https://ipindia.gov.in/writereaddata/Portal/News/860_1_Patent_Amendment_Rules_2024.pdf",
        "category": "classical",
        "source_type": "subordinate_regulation",
        "authority_level": "subordinate_regulation",
        "legal_status": "in_force (notified March 2024)",
        "enacted_date": "2024-03-15",
        "effective_date": "2024-03-15",
        "binding_on_jurisdiction": "India (subordinate regulation)",
        "status_source": "Gazette of India, G.S.R. 211(E)"
    },
    "Drugs_and_Cosmetics_Act_Ayurveda.pdf": {
        "statute_title": "The Drugs and Cosmetics Act, 1940 & Rules (Ayurveda Provisions)",
        "jurisdiction": "national",
        "official_source_url": "https://ayush.gov.in/docs/drugs-and-cosmetics-act.pdf",
        "category": "classical",
        "source_type": "primary_statute",
        "authority_level": "primary_statute",
        "legal_status": "in_force (as amended)",
        "enacted_date": "1940-04-10",
        "effective_date": "1940-04-10",
        "binding_on_jurisdiction": "India (primary statute)"
    },
    "Biological_Diversity_Amendment_Act_2023.pdf": {
        "statute_title": "The Biological Diversity (Amendment) Act, 2023",
        "jurisdiction": "national",
        "official_source_url": "http://nbaindia.org/uploaded/pdf/BDAct_2023_Amendment.pdf",
        "category": "classical",
        "source_type": "primary_statute",
        "authority_level": "primary_statute",
        "legal_status": "in_force (enacted Aug 2023)",
        "enacted_date": "2023-08-03",
        "effective_date": "2023-08-03",
        "binding_on_jurisdiction": "India (primary statute)"
    },
    "IPO_Traditional_Knowledge_Guidelines.pdf": {
        "statute_title": "Academic Legal Study: Traditional Knowledge Protection & Patent Guidelines (IOSR-JHSS)",
        "jurisdiction": "national",
        "official_source_url": "https://www.iosrjournals.org/iosr-jhss/papers/Vol3-issue1/G0313542.pdf",
        "category": "classical",
        "source_type": "secondary_academic_study",
        "authority_level": "secondary_academic_study",
        "legal_status": "academic_publication (non-binding scholarship 2012)",
        "enacted_date": "2012-09-01",
        "binding_on_jurisdiction": "Non-binding academic scholarship (IOSR-JHSS 2012)"
    },
    "FSSAI_Ayurveda_Aahar_Regulations_2022.pdf": {
        "statute_title": "Food Safety and Standards (Ayurveda Aahar) Regulations, 2022",
        "jurisdiction": "national",
        "official_source_url": "https://www.fssai.gov.in/upload/uploadfiles/files/FSSAI_Ayurveda_Aahar_Regulations_2022.pdf",
        "category": "classical",
        "source_type": "subordinate_regulation",
        "authority_level": "subordinate_regulation",
        "legal_status": "in_force (notified May 2022)",
        "enacted_date": "2022-05-05",
        "effective_date": "2022-05-05",
        "binding_on_jurisdiction": "India (subordinate regulation)"
    },
    "WIPO_GRATK_Treaty_2024.pdf": {
        "statute_title": "WIPO Treaty on Intellectual Property, Genetic Resources and Associated Traditional Knowledge (2024)",
        "jurisdiction": "international",
        "official_source_url": "https://www.wipo.int/edocs/mdocs/tk/en/gratk_dc/gratk_dc_7.pdf",
        "category": "classical",
        "source_type": "international_treaty",
        "authority_level": "international_treaty",
        "legal_status": "adopted (May 2024, pending entry into force under Article 17)",
        "adopted_date": "2024-05-24",
        "entry_into_force_date": "pending (requires 15 ratifications under Art. 17)",
        "binding_on_jurisdiction": "Not currently binding on India (adopted, pending ratification & entry into force)",
        "status_source": "WIPO Diplomatic Conference (GRATK/DC/7)"
    },
    "Nagoya_Protocol_ABS.pdf": {
        "statute_title": "Nagoya Protocol on Access and Benefit-Sharing (CBD)",
        "jurisdiction": "international",
        "official_source_url": "https://www.cbd.int/abs/doc/protocol/nagoya-protocol-en.pdf",
        "category": "classical",
        "source_type": "international_treaty",
        "authority_level": "international_treaty",
        "legal_status": "in_force (ratified by India on 9 October 2012; entered into force globally and for India on 12 October 2014)",
        "adopted_date": "2010-10-29",
        "signature_date": "2011-05-11",
        "ratified_date": "2012-10-09",
        "global_entry_into_force_date": "2014-10-12",
        "entry_into_force_date": "2014-10-12",
        "entry_into_force_for_india_date": "2014-10-12",
        "binding_on_jurisdiction": "India (ratified 9 October 2012; entry into force 12 October 2014, legally binding international treaty)",
        "status_source": "CBD Secretariat / UN Treaty Collection",
        "status_verified_at": "2024-05"
    },
    "WTO_TRIPS_Agreement.pdf": {
        "statute_title": "WTO TRIPS Agreement (Article 27 & Traditional Knowledge)",
        "jurisdiction": "international",
        "official_source_url": "https://www.wto.org/english/docs_e/legal_e/27-trips.pdf",
        "category": "classical",
        "source_type": "international_treaty",
        "authority_level": "international_treaty",
        "legal_status": "in_force (entered into force Jan 1995, binding on WTO members)",
        "adopted_date": "1994-04-15",
        "entry_into_force_date": "1995-01-01",
        "binding_on_jurisdiction": "India (WTO member, legally binding international treaty)"
    }
}

class PDFStatutoryLoader:
    """
    High-precision statutory PDF parser using PyMuPDF (fitz).
    Splits legal documents along natural statutory boundaries (Sections, Rules, Articles),
    identifies legal section and rule headers, and creates structured LegalChunks.
    """

    def __init__(self, chunk_size: int = 750, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n## ",
                "\nSection ",
                "\nRule ",
                "\nArticle ",
                "\nRegulation ",
                "\nSchedule ",
                "\nCHAPTER ",
                "\n\n",
                "\n",
                ". ",
                " "
            ]
        )

    def _extract_pages_text(self, pdf_path: Path) -> List[tuple]:
        """Extracts (page_num, text) tuples using fitz or pypdf."""
        pages = []
        if HAS_PYMUPDF:
            doc = fitz.open(str(pdf_path))
            for idx, page in enumerate(doc, start=1):
                txt = page.get_text()
                if txt and len(txt.strip()) >= 20:
                    pages.append((idx, txt))
        else:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            for idx, page in enumerate(reader.pages, start=1):
                txt = page.extract_text()
                if txt and len(txt.strip()) >= 20:
                    pages.append((idx, txt))
        return pages

    def load_pdf(self, pdf_path: Path, jurisdiction: str) -> List[LegalChunk]:
        """Loads and chunks a single PDF document."""
        if not pdf_path.exists():
            print(f"[WARN] PDF file {pdf_path} not found.", flush=True)
            return []

        filename = pdf_path.name
        meta_info = DOCUMENT_METADATA_REGISTRY.get(filename, {
            "statute_title": filename.replace(".pdf", "").replace("_", " "),
            "jurisdiction": jurisdiction,
            "official_source_url": "https://ayush.gov.in",
            "category": "classical"
        })

        pages = self._extract_pages_text(pdf_path)
        print(f"  -> Reading [{jurisdiction.upper()}] {filename} ({len(pages)} pages)...", flush=True)

        chunks: List[LegalChunk] = []

        for page_idx, page_text in pages:
            # Clean repetitive whitespaces/newlines
            cleaned_text = re.sub(r'[ \t]+', ' ', page_text)
            cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()

            split_texts = self.text_splitter.split_text(cleaned_text)

            for split_idx, text_segment in enumerate(split_texts):
                section_identifier = self._detect_section_or_clause(text_segment, filename=filename)
                is_bar = self._detect_statutory_bar(text_segment, section_identifier)

                # Generate deterministic chunk ID
                chunk_hash = hashlib.md5(f"{filename}_{page_idx}_{split_idx}".encode()).hexdigest()[:8]
                chunk_id = f"{pdf_path.stem.lower()}_p{page_idx}_c{split_idx}_{chunk_hash}"

                detailed_status = DetailedLegalStatus(
                    authority_level=meta_info.get("authority_level", "primary_statute"),
                    canonical_status=meta_info.get("canonical_status", meta_info.get("legal_status", "in_force")),
                    enacted_date=meta_info.get("enacted_date"),
                    effective_date=meta_info.get("effective_date"),
                    amended_date=meta_info.get("amended_date"),
                    adopted_date=meta_info.get("adopted_date"),
                    ratified_date=meta_info.get("ratified_date"),
                    entry_into_force_date=meta_info.get("entry_into_force_date"),
                    binding_on_jurisdiction=meta_info.get("binding_on_jurisdiction"),
                    status_source=meta_info.get("status_source"),
                    status_verified_at=meta_info.get("status_verified_at")
                )

                metadata = StatutoryMetadata(
                    document_name=filename,
                    statute_title=meta_info["statute_title"],
                    section_or_clause=section_identifier,
                    jurisdiction=meta_info["jurisdiction"],
                    category=meta_info.get("category", "classical"),
                    source_type=meta_info.get("source_type", "primary_statute"),
                    authority_level=meta_info.get("authority_level", meta_info.get("source_type", "primary_statute")),
                    legal_status=meta_info.get("legal_status", "in_force"),
                    binding_on_jurisdiction=meta_info.get("binding_on_jurisdiction"),
                    detailed_legal_status=detailed_status,
                    page_numbers=[page_idx],
                    official_source_url=meta_info["official_source_url"],
                    is_statutory_bar=is_bar
                )

                chunks.append(LegalChunk(
                    chunk_id=chunk_id,
                    text=text_segment,
                    metadata=metadata
                ))

        print(f"    [OK] Extracted {len(chunks)} chunks from {filename}", flush=True)
        return chunks

    def load_all_raw_documents(self, raw_docs_dir: Path) -> Dict[str, List[LegalChunk]]:
        """Loads all national and international PDFs from the raw_documents folder with disk caching."""
        cache_path = getattr(settings, "CHUNKS_CACHE_FILE", None)
        if cache_path and cache_path.exists():
            try:
                cache_mtime = cache_path.stat().st_mtime
                # Check if any PDF is newer than cache
                is_stale = False
                for pdf_file in raw_docs_dir.rglob("*.pdf"):
                    if pdf_file.stat().st_mtime > cache_mtime:
                        is_stale = True
                        break
                for pdf_file in raw_docs_dir.rglob("*.PDF"):
                    if pdf_file.stat().st_mtime > cache_mtime:
                        is_stale = True
                        break

                if not is_stale:
                    with open(cache_path, "rb") as f:
                        cached_results = pickle.load(f)
                    if cached_results.get("national") and cached_results.get("international"):
                        # Synchronize updated registry metadata (e.g. titles, source_type, legal status) to cached chunks
                        for jur in ["national", "international"]:
                            for chunk in cached_results[jur]:
                                reg = DOCUMENT_METADATA_REGISTRY.get(chunk.metadata.document_name)
                                if reg:
                                    chunk.metadata.statute_title = reg["statute_title"]
                                    chunk.metadata.official_source_url = reg["official_source_url"]
                                    chunk.metadata.source_type = reg.get("source_type", "primary_statute")
                                    chunk.metadata.authority_level = reg.get("authority_level", "primary_statute")
                                    chunk.metadata.legal_status = reg.get("legal_status", "in_force")
                                    chunk.metadata.binding_on_jurisdiction = reg.get("binding_on_jurisdiction")
                                    chunk.metadata.detailed_legal_status = DetailedLegalStatus(
                                        authority_level=reg.get("authority_level", "primary_statute"),
                                        canonical_status=reg.get("canonical_status", reg.get("legal_status", "in_force")),
                                        enacted_date=reg.get("enacted_date"),
                                        effective_date=reg.get("effective_date"),
                                        amended_date=reg.get("amended_date"),
                                        adopted_date=reg.get("adopted_date"),
                                        ratified_date=reg.get("ratified_date"),
                                        entry_into_force_date=reg.get("entry_into_force_date"),
                                        binding_on_jurisdiction=reg.get("binding_on_jurisdiction"),
                                        status_source=reg.get("status_source"),
                                        status_verified_at=reg.get("status_verified_at")
                                    )
                                if jur == "national" and ("mere discovery of a new form" in chunk.text.lower() or ("known substance" in chunk.text.lower() and "enhancement of the known efficacy" in chunk.text.lower())):
                                    chunk.metadata.section_or_clause = "Section 3(d)"
                                    chunk.metadata.is_statutory_bar = True
                                if jur == "national" and "patent_amendment_rules" in chunk.metadata.document_name.lower():
                                    if chunk.metadata.section_or_clause.startswith("Section"):
                                        chunk.metadata.section_or_clause = "General Provision"
                                if jur == "international" and "trips" in chunk.metadata.document_name.lower():
                                    if 13 in chunk.metadata.page_numbers:
                                        if "article 27" in chunk.text.lower() or "patentable subject matter" in chunk.text.lower() or "exclude from patentability" in chunk.text.lower():
                                            chunk.metadata.section_or_clause = "Article 27"
                        print(f"📦 Loaded {len(cached_results['national'])} national and {len(cached_results['international'])} international chunks from cache.", flush=True)
                        return cached_results
            except Exception as e:
                print(f"⚠️ Failed to load chunks cache: {e}. Re-indexing...", flush=True)

        national_dir = raw_docs_dir / "national"
        intl_dir = raw_docs_dir / "international"

        results = {
            "national": [],
            "international": []
        }

        if national_dir.exists():
            seen_files = set()
            for pdf_file in sorted(national_dir.iterdir()):
                if pdf_file.is_file() and pdf_file.suffix.lower() == ".pdf":
                    if pdf_file.name.lower() not in seen_files:
                        seen_files.add(pdf_file.name.lower())
                        results["national"].extend(self.load_pdf(pdf_file, jurisdiction="national"))

        if intl_dir.exists():
            seen_files = set()
            for pdf_file in sorted(intl_dir.iterdir()):
                if pdf_file.is_file() and pdf_file.suffix.lower() == ".pdf":
                    if pdf_file.name.lower() not in seen_files:
                        seen_files.add(pdf_file.name.lower())
                        results["international"].extend(self.load_pdf(pdf_file, jurisdiction="international"))

        # Save to disk cache if path is configured
        if cache_path and (results["national"] or results["international"]):
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(results, f)
                print(f"💾 Cached {len(results['national']) + len(results['international'])} chunks to {cache_path}", flush=True)
            except Exception as e:
                print(f"⚠️ Failed to write chunks cache: {e}", flush=True)

        return results

    def _detect_section_or_clause(self, text: str, filename: str = "") -> str:
        """Detects specific section/rule/article mentions in the text."""
        lower = text.lower()
        is_patent_rules = "patent_amendment_rules" in filename.lower()

        if not is_patent_rules:
            if "mere discovery of a new form" in lower or ("known substance" in lower and "enhancement of the known efficacy" in lower) or ("(d)" in lower and "new form of a known substance" in lower):
                return "Section 3(d)"
            if "traditional knowledge" in lower and ("(p)" in lower or "not patentable" in lower or "aggregation" in lower):
                return "Section 3(p)"
            if "mere admixture" in lower or ("(e)" in lower and "aggregation of properties" in lower):
                return "Section 3(e)"
        if "rule 161" in lower or "provisions of rule 161" in lower or "true list of" in lower:
            return "Rule 161"
        if "rule 158b" in lower or "patent or proprietary" in lower:
            return "Rule 158B"
        if "first schedule" in lower or "ayurvedic formulary" in lower:
            return "First Schedule"

        # 1. Clean out Gazette notification headers and preambles
        clean_text = re.sub(r'\[?\s*part\s*[-–—I|]+\s*(?:sec\.|section)\s*[0-9\(\)a-z]+\]?', '', text, flags=re.IGNORECASE)
        clean_text = re.sub(r'sub-section\s*\([0-9]+\)', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'section\s+3,\s+sub-section\s*\([0-9ivxa-z]+\)\s+inviting', '', clean_text, flags=re.IGNORECASE)

        match = re.search(
            r'\b((?:Section|Sec\.)\s+[0-9]{1,3}[A-Za-z]?(?:\([a-zA-Z0-9]+\))*|'
            r'Rule\s+[0-9]{1,3}[A-Za-z]?(?:\([a-zA-Z0-9]+\))*|'
            r'Article\s+[0-9]{1,3}[A-Za-z]?(?:\([a-zA-Z0-9]+\))*|'
            r'Regulation\s+[0-9]{1,3}[A-Za-z]?(?:\([a-zA-Z0-9]+\))*|'
            r'First\s+Schedule|Schedule\s+[A-Z0-9]+)\b',
            clean_text,
            re.IGNORECASE
        )
        if match:
            val = match.group(1).strip()
            val = re.sub(r'^Sec\.\s*', 'Section ', val, flags=re.IGNORECASE)
            # In Patent Amendment Rules, there are no Sections of an Act — only amendment Rules
            if is_patent_rules and val.lower().startswith("section"):
                return "General Provision"
            return val

        # 2. Check for numbered section headers like '3. What are not inventions'
        if not is_patent_rules:
            m_num = re.search(r'(?:^|\n)([0-9]{1,3}[A-Za-z]?)\.\s+([A-Za-z\s]{3,35})', clean_text)
            if m_num:
                return f"Section {m_num.group(1)}"

        return "General Provision"

    def _detect_statutory_bar(self, text: str, section: str) -> bool:
        """Flags key legal bars (like Section 3(p), Section 3(e), Section 3(d), traditional knowledge bars)."""
        lower = text.lower()
        if "3(p)" in section or "3(p)" in lower or ("traditional knowledge" in lower and "not patentable" in lower):
            return True
        if "3(e)" in section or "mere admixture" in lower:
            return True
        if "3(d)" in section or "new form of a known substance" in lower:
            return True
        return False
