# RAG básico con ChromaDB
import os
from typing import List
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


class SimpleRAG:
    """RAG minimalista con ChromaDB."""

    def __init__(self, collection_name: str = "docs", persist_dir: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def add_documents(self, docs: List[str], ids: List[str] = None, metadatas: List[dict] = None):
        """Ingesta documentos a la base vectorial."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(docs))]
        embeddings = self.embedder.encode(docs).tolist()
        self.collection.add(
            documents=docs,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

    def query(self, question: str, k: int = 3) -> List[str]:
        """Recupera k documentos relevantes."""
        q_emb = self.embedder.encode([question]).tolist()
        results = self.collection.query(query_embeddings=q_emb, n_results=k)
        return results["documents"][0] if results["documents"] else []

    def count(self) -> int:
        """Cantidad de documentos en la colección."""
        return self.collection.count()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Divide texto en chunks por párrafos."""
    # Limpiar texto (remover líneas vacías múltiples)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    text_clean = "\n".join(lines)

    # Dividir por párrafos (doble salto de línea o líneas largas)
    paragraphs = []
    current = []
    for line in text_clean.split("\n"):
        current.append(line)
        # Si el párrafo actual es suficientemente largo, cortar
        current_text = "\n".join(current)
        if len(current_text) >= chunk_size * 0.8:
            paragraphs.append(current_text)
            current = []
    if current:
        paragraphs.append("\n".join(current))

    # Crear chunks con overlap
    chunks = []
    for para in paragraphs:
        if len(para) > chunk_size:
            # Dividir párrafos muy largos
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunk = para[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start = end - overlap
        else:
            if para.strip():
                chunks.append(para.strip())

    return [c for c in chunks if len(c) > 50]  # Filtrar chunks muy pequeños


def ingest_file(rag: SimpleRAG, filepath: str):
    """Ingesta un archivo de texto al RAG."""
    path = Path(filepath)
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    ids = [f"{path.stem}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": path.name, "chunk": i} for i in range(len(chunks))]
    rag.add_documents(chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


if __name__ == "__main__":
    # Test: ingestar corpus y hacer query
    import shutil

    db_path = "./chroma_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    rag = SimpleRAG(persist_dir=db_path)

    # Ingestar sílabos
    data_path = Path(__file__).parent.parent / "data" / "silabos.txt"
    n = ingest_file(rag, str(data_path))
    print(f"Ingestados {n} chunks")
    print(f"Total docs en DB: {rag.count()}")

    # Test query
    q = "¿Cuáles son los prerequisitos de CC0C2?"
    results = rag.query(q, k=3)
    print(f"\nQuery: {q}")
    print("Resultados:")
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r[:200]}...")
