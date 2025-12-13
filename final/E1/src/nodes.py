# Nodos del grafo multi-agente
from typing import List
from .state import AgentState
from .rag import SimpleRAG, ingest_file
from .llm import decompose_question, generate_answer, synthesize_answers
from pathlib import Path

# RAG global (se inicializa una vez)
_rag = None


def get_rag() -> SimpleRAG:
    """Obtiene o inicializa el RAG (debe ejecutarse ingest_pdfs.py primero)."""
    global _rag
    if _rag is None:
        db_path = Path(__file__).parent.parent / "out" / "chroma_db"
        _rag = SimpleRAG(persist_dir=str(db_path))
        if _rag.count() == 0:
            print("⚠️  RAG vacío. Ejecuta: python ingest_pdfs.py")
    return _rag


def manager_node(state: AgentState) -> dict:
    """Descompone la pregunta en subtareas usando LLM."""
    question = state["question"]
    subtasks = decompose_question(question)

    return {
        "subtasks": subtasks,
        "logs": [{
            "node": "manager",
            "action": "decompose",
            "input": question,
            "output": subtasks,
            "thinking": f"Analizando pregunta... {'Es simple, una sola tarea' if len(subtasks) == 1 else f'Descompuesta en {len(subtasks)} subtareas'}"
        }]
    }


def retriever_node(state: AgentState) -> dict:
    """Recupera contexto relevante del RAG."""
    question = state["question"]
    subtasks = state.get("subtasks", [question])

    rag = get_rag()
    all_context = []

    for task in subtasks:
        docs = rag.query(task, k=3)
        all_context.extend(docs)

    # Eliminar duplicados manteniendo orden
    seen = set()
    unique_context = []
    for doc in all_context:
        if doc not in seen:
            seen.add(doc)
            unique_context.append(doc)

    # Preparar chunks para logging
    chunks_preview = [
        {"idx": i+1, "preview": doc[:150] + "..." if len(doc) > 150 else doc, "full": doc}
        for i, doc in enumerate(unique_context)
    ]

    return {
        "context": unique_context,
        "logs": [{
            "node": "retriever",
            "action": "retrieve",
            "n_docs": len(unique_context),
            "chunks": chunks_preview,
            "thinking": f"Buscando en ChromaDB para {len(subtasks)} subtarea(s)... Encontré {len(unique_context)} chunks relevantes"
        }]
    }


def worker_node(state: AgentState) -> dict:
    """Genera respuesta basada en contexto usando LLM."""
    question = state["question"]
    subtasks = state.get("subtasks", [question])
    context = state.get("context", [])

    # Generar respuesta para cada subtarea
    partial_answers = []
    for task in subtasks:
        answer = generate_answer(task, context)
        partial_answers.append(answer)

    return {
        "partial_answers": partial_answers,
        "logs": [{
            "node": "worker",
            "action": "answer",
            "n_subtasks": len(subtasks),
            "n_context_docs": len(context),
            "partial_answers": partial_answers,
            "thinking": f"Generando respuesta usando {len(context)} docs de contexto..."
        }]
    }


def synthesizer_node(state: AgentState) -> dict:
    """Consolida respuestas parciales en respuesta final usando LLM."""
    partial = state.get("partial_answers", [])
    question = state["question"]

    if partial:
        final = synthesize_answers(question, partial)
    else:
        final = "No se pudo generar una respuesta."

    return {
        "final_answer": final,
        "logs": [{
            "node": "synthesizer",
            "action": "consolidate",
            "n_partials": len(partial),
            "final_answer": final,
            "thinking": f"{'Consolidando ' + str(len(partial)) + ' respuestas parciales...' if len(partial) > 1 else 'Una sola respuesta, pasando directo...'}"
        }]
    }
