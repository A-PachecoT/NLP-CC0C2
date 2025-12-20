# Nodos del grafo multi-agente - E2 con Critic
from typing import List
from .state import AgentState
from .rag import SimpleRAG
from .llm import (
    decompose_question, generate_answer, synthesize_answers,
    critic_review, revise_answer, generate_multiple_answers, select_best_answer
)
from pathlib import Path

# RAG global
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

    # Eliminar duplicados
    seen = set()
    unique_context = []
    for doc in all_context:
        if doc not in seen:
            seen.add(doc)
            unique_context.append(doc)

    # Preparar chunks para logging (preview de cada uno)
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
    """Genera respuesta con chain-of-thought."""
    question = state["question"]
    subtasks = state.get("subtasks", [question])
    context = state.get("context", [])

    partial_answers = []
    for task in subtasks:
        answer = generate_answer(task, context, chain_of_thought=True)
        partial_answers.append(answer)

    return {
        "partial_answers": partial_answers,
        "logs": [{
            "node": "worker",
            "action": "answer_cot",
            "n_subtasks": len(subtasks),
            "n_context_docs": len(context),
            "partial_answers": partial_answers,
            "thinking": f"Generando respuesta con Chain-of-Thought usando {len(context)} docs de contexto..."
        }]
    }


def worker_self_consistency_node(state: AgentState) -> dict:
    """Worker con self-consistency: genera múltiples respuestas y elige la mejor."""
    question = state["question"]
    context = state.get("context", [])

    # Generar 3 respuestas
    answers = generate_multiple_answers(question, context, n=3)

    # Seleccionar la mejor
    best_answer, best_idx = select_best_answer(question, answers, context)

    return {
        "partial_answers": [best_answer],
        "logs": [{
            "node": "worker_sc",
            "action": "self_consistency",
            "n_candidates": len(answers),
            "selected": best_idx + 1,
            "all_candidates": answers,
            "best_answer": best_answer,
            "thinking": f"Self-consistency: Generé {len(answers)} respuestas candidatas, seleccioné la #{best_idx + 1}"
        }]
    }


def synthesizer_node(state: AgentState) -> dict:
    """Consolida respuestas parciales."""
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


def critic_node(state: AgentState) -> dict:
    """Critic: revisa la respuesta y solicita revisión si hay problemas."""
    question = state["question"]
    answer = state.get("final_answer", "")
    context = state.get("context", [])

    review = critic_review(question, answer, context)

    # Si necesita revisión, generar nueva respuesta
    if review.get("verdict") == "NEEDS_REVISION":
        revised = revise_answer(question, answer, context, review)
        return {
            "final_answer": revised,
            "critic_review": review,
            "logs": [{
                "node": "critic",
                "action": "revise",
                "verdict": review.get("verdict"),
                "issues": review.get("issues", []),
                "original_answer": answer,
                "revised_answer": revised,
                "thinking": f"❌ Detecté problemas: {review.get('issues', [])}. Generando respuesta corregida..."
            }]
        }

    return {
        "critic_review": review,
        "logs": [{
            "node": "critic",
            "action": "approve",
            "verdict": review.get("verdict"),
            "grounded": review.get("grounded"),
            "complete": review.get("complete"),
            "accurate": review.get("accurate"),
            "thinking": f"✅ Respuesta validada - Grounded: {review.get('grounded')}, Complete: {review.get('complete')}, Accurate: {review.get('accurate')}"
        }]
    }
