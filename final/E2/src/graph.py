# Grafo LangGraph para sistema multi-agente - E2 con Critic
import json
from pathlib import Path
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    manager_node, retriever_node, worker_node, synthesizer_node,
    critic_node, worker_self_consistency_node
)


def build_graph(use_critic: bool = True, use_self_consistency: bool = False) -> StateGraph:
    """Construye el grafo con opciones de Critic y Self-Consistency."""
    graph = StateGraph(AgentState)

    # Agregar nodos
    graph.add_node("manager", manager_node)
    graph.add_node("retriever", retriever_node)

    if use_self_consistency:
        graph.add_node("worker", worker_self_consistency_node)
    else:
        graph.add_node("worker", worker_node)

    graph.add_node("synthesizer", synthesizer_node)

    if use_critic:
        graph.add_node("critic", critic_node)

    # Definir flujo
    graph.set_entry_point("manager")
    graph.add_edge("manager", "retriever")
    graph.add_edge("retriever", "worker")
    graph.add_edge("worker", "synthesizer")

    if use_critic:
        graph.add_edge("synthesizer", "critic")
        graph.add_edge("critic", END)
    else:
        graph.add_edge("synthesizer", END)

    return graph.compile()


def run_query(question: str, save_logs: bool = True, use_critic: bool = True,
              use_self_consistency: bool = False) -> dict:
    """Ejecuta una consulta en el grafo."""
    app = build_graph(use_critic=use_critic, use_self_consistency=use_self_consistency)

    initial_state = {
        "question": question,
        "subtasks": [],
        "context": [],
        "partial_answers": [],
        "final_answer": None,
        "critic_review": None,
        "logs": []
    }
    result = app.invoke(initial_state)

    # Guardar logs enriquecidos
    if save_logs:
        save_enriched_logs(question, result, use_critic, use_self_consistency)

    return result


def run_query_stream(question: str, use_critic: bool = True, use_self_consistency: bool = False):
    """Ejecuta consulta con streaming - yield de cada paso."""
    app = build_graph(use_critic=use_critic, use_self_consistency=use_self_consistency)

    initial_state = {
        "question": question,
        "subtasks": [],
        "context": [],
        "partial_answers": [],
        "final_answer": None,
        "critic_review": None,
        "logs": []
    }

    # Stream cada paso del grafo
    for event in app.stream(initial_state):
        yield event


def save_enriched_logs(question: str, result: dict, use_critic: bool, use_self_consistency: bool):
    """Guarda logs enriquecidos con chunks y razonamiento."""
    from datetime import datetime

    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    log_file = out_dir / "query_logs.jsonl"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "subtasks": result.get("subtasks", []),
        "context_full": result.get("context", []),  # chunks completos
        "n_context": len(result.get("context", [])),
        "logs": result.get("logs", []),
        "critic_review": result.get("critic_review"),
        "final_answer": result.get("final_answer", ""),
        "config": {
            "use_critic": use_critic,
            "use_self_consistency": use_self_consistency
        }
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# Grafo compilado para LangGraph CLI/Studio (con Critic por defecto)
graph = build_graph(use_critic=True, use_self_consistency=False)


if __name__ == "__main__":
    # Test E2 con Critic
    questions = [
        "¿Cuáles son los prerequisitos de CC0C2?",
        "¿Cuántos créditos necesito para graduarme?",
        "¿Qué nivel de inglés necesito?",
    ]

    print("\n" + "="*60)
    print("  E2: Multi-Agent QA con CRITIC")
    print("="*60)

    for q in questions:
        print(f"\n{'─'*60}")
        print(f"Q: {q}")
        print("─"*60)

        result = run_query(q, use_critic=True)

        print(f"\n📝 Respuesta:\n{result['final_answer'][:600]}")

        # Mostrar review del critic
        review = result.get("critic_review", {})
        if review:
            verdict = review.get("verdict", "N/A")
            emoji = "✅" if verdict == "APPROVED" else "🔄"
            print(f"\n{emoji} Critic: {verdict}")
            if review.get("issues"):
                print(f"   Issues: {review['issues']}")

        print(f"\n📊 {len(result['logs'])} nodos ejecutados")
