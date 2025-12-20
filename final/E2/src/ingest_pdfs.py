#!/usr/bin/env python3
"""Ingesta PDFs reales al RAG."""
import fitz  # PyMuPDF
from pathlib import Path
import shutil
from rag import SimpleRAG, chunk_text


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrae texto de un PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def ingest_pdfs(raw_dir: str, rag: SimpleRAG):
    """Ingesta todos los PDFs de un directorio."""
    raw_path = Path(raw_dir)
    total_chunks = 0

    for pdf_file in raw_path.glob("*.pdf"):
        print(f"\n📄 Procesando: {pdf_file.name}")

        # Extraer texto
        text = extract_text_from_pdf(str(pdf_file))
        print(f"   Texto extraído: {len(text)} caracteres")

        # Dividir en chunks
        chunks = chunk_text(text, chunk_size=800, overlap=150)
        print(f"   Chunks generados: {len(chunks)}")

        # Crear IDs y metadata
        base_name = pdf_file.stem
        ids = [f"{base_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": pdf_file.name, "chunk": i} for i in range(len(chunks))]

        # Ingestar
        rag.add_documents(chunks, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)

    return total_chunks


def main():
    print("="*60)
    print("  INGESTA DE PDFs REALES")
    print("="*60)

    # Paths
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    db_path = Path(__file__).parent.parent / "out" / "chroma_db"

    # Limpiar DB anterior
    if db_path.exists():
        print(f"\n🗑️  Limpiando DB anterior: {db_path}")
        shutil.rmtree(db_path)

    # Crear RAG
    rag = SimpleRAG(persist_dir=str(db_path))

    # Ingestar PDFs
    total = ingest_pdfs(str(raw_dir), rag)

    print(f"\n{'='*60}")
    print(f"✅ Total chunks ingestados: {total}")
    print(f"✅ Documentos en DB: {rag.count()}")
    print("="*60)

    # Test query
    print("\n🔍 Test query:")
    q = "¿Qué temas cubre el curso de NLP?"
    results = rag.query(q, k=3)
    print(f"Q: {q}")
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r[:150]}...")


if __name__ == "__main__":
    main()
