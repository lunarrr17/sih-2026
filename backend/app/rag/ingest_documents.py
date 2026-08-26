import sys
import os
import time
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.core.config import settings
from backend.app.core.qdrant_client_provider import QdrantClientProvider
from backend.app.rag.pdf_loader import PDFStatutoryLoader
from backend.app.rag.indexer import QdrantCorpusIndexer
from backend.app.rag.embedder import DenseEmbedder

def run_ingestion():
    print("[1/4] Initializing Qdrant and Dense Embedder...")
    start_time = time.time()
    
    client = QdrantClientProvider.get_client()
    embedder = DenseEmbedder.get_instance()
    indexer = QdrantCorpusIndexer(client=client, embedder=embedder)
    
    print("\n[2/4] Parsing National and International Statutory PDFs...")
    loader = PDFStatutoryLoader(chunk_size=700, chunk_overlap=100)
    raw_dir = settings.DATA_DIR / "raw_documents"
    
    chunks_dict = loader.load_all_raw_documents(raw_dir)
    nat_chunks = chunks_dict["national"]
    intl_chunks = chunks_dict["international"]
    
    print(f"[OK] Extracted {len(nat_chunks)} National chunks and {len(intl_chunks)} International chunks.")
    
    print("\n[3/4] Indexing Chunks into Qdrant Collections...")
    nat_count = indexer.index_chunks(nat_chunks, settings.QDRANT_COLLECTION_NATIONAL, batch_size=32)
    intl_count = indexer.index_chunks(intl_chunks, settings.QDRANT_COLLECTION_INTERNATIONAL, batch_size=32)
    
    print("\n[4/4] Verifying Index Stats...")
    nat_info = client.get_collection(settings.QDRANT_COLLECTION_NATIONAL)
    intl_info = client.get_collection(settings.QDRANT_COLLECTION_INTERNATIONAL)
    
    elapsed = round(time.time() - start_time, 2)
    print("=" * 60)
    print(f"INGESTION COMPLETE in {elapsed}s")
    print(f"National Collection ({settings.QDRANT_COLLECTION_NATIONAL}): {nat_info.points_count} points")
    print(f"International Collection ({settings.QDRANT_COLLECTION_INTERNATIONAL}): {intl_info.points_count} points")
    print("=" * 60)

if __name__ == "__main__":
    run_ingestion()
