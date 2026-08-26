import re
import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    from pypdf import PdfReader
    HAS_PYMUPDF = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.app.core.config import settings
from backend.app.rag.schemas import LegalChunk, StatutoryMetadata

# Registry mapping PDF filenames to official metadata and URLs
DOCUMENT_METADATA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Patents_Act_1970.PDF": {
        "statute_title": "The Patents Act, 1970 (as amended)",
        "jurisdiction": "national",
        "official_source_url": "https://ipindia.gov.in/writereaddata/Portal/IPOAct/1_31_1_patent-act-1970-11march2015.pdf",
        "category": "classical"
    },
    "Patent_Amendment_Rules_2024.pdf": {
        "statute_title": "Patents (Amendment) Rules, 2024",
        "jurisdiction": "national",
        "official_source_url": "https://ipindia.gov.in/writereaddata/Portal/News/860_1_Patent_Amendment_Rules_2024.pdf",
        "category": "classical"
    },
    "Drugs_and_Cosmetics_Act_Ayurveda.pdf": {
        "statute_title": "The Drugs and Cosmetics Act, 1940 & Rules (Ayurveda Provisions)",
        "jurisdiction": "national",
        "official_source_url": "https://ayush.gov.in/docs/drugs-and-cosmetics-act.pdf",
        "category": "classical"
    },
    "Biological_Diversity_Amendment_Act_2023.pdf": {
        "statute_title": "The Biological Diversity (Amendment) Act, 2023",
        "jurisdiction": "national",
        "official_source_url": "http://nbaindia.org/uploaded/pdf/BDAct_2023_Amendment.pdf",
        "category": "classical"
    },
    "IPO_Traditional_Knowledge_Guidelines.pdf": {
        "statute_title": "IPO Guidelines for Examination of Traditional Knowledge Patents",
        "jurisdiction": "national",
        "official_source_url": "https://ipindia.gov.in/writereaddata/Portal/IPOGuidelines/1_86_1_Guidelines_for_Examination_of_Traditional_Knowledge.pdf",
        "category": "classical"
    },
    "FSSAI_Ayurveda_Aahar_Regulations_2022.pdf": {
        "statute_title": "Food Safety and Standards (Ayurveda Aahar) Regulations, 2022",
        "jurisdiction": "national",
        "official_source_url": "https://www.fssai.gov.in",
        "category": "classical"
    },
    "WIPO_GRATK_Treaty_2024.pdf": {
        "statute_title": "WIPO Treaty on Intellectual Property, Genetic Resources and Associated Traditional Knowledge (2024)",
        "jurisdiction": "international",
        "official_source_url": "https://www.wipo.int/edocs/mdocs/tk/en/gratk_dc/gratk_dc_7.pdf",
        "category": "classical"
    },
    "Nagoya_Protocol_ABS.pdf": {
        "statute_title": "Nagoya Protocol on Access and Benefit Sharing (CBD)",
        "jurisdiction": "international",
        "official_source_url": "https://www.cbd.int/abs/doc/protocol/nagoya-protocol-en.pdf",
        "category": "classical"
    },
    "WTO_TRIPS_Agreement.pdf": {
        "statute_title": "WTO Agreement on Trade-Related Aspects of Intellectual Property Rights (TRIPS)",
        "jurisdiction": "international",
        "official_source_url": "https://www.wto.org/english/docs_e/legal_e/27-trips.pdf",
        "category": "classical"
    }
}

class PDFStatutoryLoader:
    """
    Extracts text page-by-page from raw official statutory PDFs,
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
                section_identifier = self._detect_section_or_clause(text_segment)
                is_bar = self._detect_statutory_bar(text_segment, section_identifier)

                # Generate deterministic chunk ID
                chunk_hash = hashlib.md5(f"{filename}_{page_idx}_{split_idx}".encode()).hexdigest()[:8]
                chunk_id = f"{pdf_path.stem.lower()}_p{page_idx}_c{split_idx}_{chunk_hash}"

                metadata = StatutoryMetadata(
                    document_name=filename,
                    statute_title=meta_info["statute_title"],
                    section_or_clause=section_identifier,
                    jurisdiction=meta_info["jurisdiction"],
                    category=meta_info.get("category", "classical"),
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
        """Loads all national and international PDFs from the raw_documents folder without case duplicates."""
        national_dir = raw_docs_dir / "national"
        intl_dir = raw_docs_dir / "international"

        results = {
            "national": [],
            "international": []
        }

        if national_dir.exists():
            # Deduplicate filenames in a case-insensitive manner
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

        return results

    def _detect_section_or_clause(self, text: str) -> str:
        """Detects specific section/rule/article mentions in the text."""
        match = re.search(
            r'(Section\s+[0-9A-Za-z\(\)]+|Rule\s+[0-9A-Za-z\(\)]+|Article\s+[0-9A-Za-z\(\)]+|First\s+Schedule|Schedule\s+[A-Z0-9]+|Regulation\s+[0-9A-Za-z\(\)]+)',
            text,
            re.IGNORECASE
        )
        if match:
            return match.group(1).title()
        return "General Provision"

    def _detect_statutory_bar(self, text: str, section: str) -> bool:
        """Flags key legal bars (like Section 3(p), Section 3(e), traditional knowledge bars)."""
        lower = text.lower()
        if "3(p)" in section or "3(p)" in lower or "traditional knowledge" in lower and "not patentable" in lower:
            return True
        if "3(e)" in section or "mere admixture" in lower:
            return True
        return False
