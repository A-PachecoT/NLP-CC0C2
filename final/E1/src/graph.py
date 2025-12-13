# Grafo LangGraph para sistema multi-agente
import json
from pathlib import Path
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import manager_node, retriever_node, worker_node, synthesizer_node


def build_graph():
    """Construye el grafo Manager -> Retriever -> Worker -> Synthesizer."""
    graph = StateGraph(AgentState)

    # Agregar nodos
    graph.add_node("manager", manager_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("worker", worker_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Definir flujo
    graph.set_entry_point("manager")
    graph.add_edge("manager", "retriever")
    graph.add_edge("retriever", "worker")
    graph.add_edge("worker", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_query(question: str, save_logs: bool = True) -> dict:
    """Ejecuta una consulta en el grafo."""
    app = build_graph()
    initial_state = {
        "question": question,
        "subtasks": [],
        "context": [],
        "partial_answers": [],
        "final_answer": None,
        "logs": []
    }
    result = app.invoke(initial_state)

    # Guardar logs enriquecidos
    if save_logs:
        save_enriched_logs(question, result)

    return result


def run_query_stream(question: str):
    """Ejecuta consulta con streaming - yield de cada paso."""
    app = build_graph()
    initial_state = {
        "question": question,
        "subtasks": [],
        "context": [],
        "partial_answers": [],
        "final_answer": None,
        "logs": []
    }

    # Stream cada paso del grafo
    for event in app.stream(initial_state):
        yield event


def save_enriched_logs(question: str, result: dict):
    """Guarda logs enriquecidos con chunks y razonamiento."""
    from datetime import datetime

    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    log_file = out_dir / "query_logs.jsonl"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "subtasks": result.get("subtasks", []),
        "context_full": result.get("context", []),
        "n_context": len(result.get("context", [])),
        "logs": result.get("logs", []),
        "final_answer": result.get("final_answer", "")
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# Grafo compilado para LangGraph CLI/Studio
graph = build_graph()


if __name__ == "__main__":
    # Test
    questions = [
        "¿Cuáles son los prerequisitos de CC0C2?",
        "¿Cuántos créditos necesito para graduarme?",
        "¿Qué cursos cubren transformers y redes neuronales?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("="*60)
        result = run_query(q)
        print(f"\nRespuesta:\n{result['final_answer'][:800]}")
        print(f"\nLogs: {len(result['logs'])} nodos ejecutados")
