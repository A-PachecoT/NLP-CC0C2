#!/usr/bin/env python3
"""Demo E2: Multi-Agent QA con Critic - Modos verbose, stream y show-logs."""
import json
import sys
from pathlib import Path
from .graph import run_query, run_query_stream


# Colores ANSI para terminal
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def print_header():
    print(f"\n{C.BOLD}{'='*60}")
    print("  MULTI-AGENT QA SYSTEM - E2")
    print("  LangGraph + RAG + Critic + GPT-4.1-mini")
    print(f"{'='*60}{C.END}")


def print_verbose_step(node_name: str, log: dict):
    """Imprime paso a paso en modo verbose."""
    icons = {
        "manager": "🎯",
        "retriever": "🔍",
        "worker": "⚙️",
        "worker_sc": "🔄",
        "synthesizer": "📝",
        "critic": "🔎"
    }
    icon = icons.get(node_name, "▸")

    print(f"\n{C.CYAN}{icon} {node_name.upper()}{C.END}")
    print(f"   {C.DIM}{log.get('thinking', '')}{C.END}")

    # Detalles específicos por nodo
    if node_name == "manager":
        subtasks = log.get("output", [])
        for i, task in enumerate(subtasks, 1):
            print(f"   {C.YELLOW}→ Subtarea {i}: {task}{C.END}")

    elif node_name == "retriever":
        chunks = log.get("chunks", [])
        print(f"   {C.GREEN}Chunks recuperados:{C.END}")
        for chunk in chunks[:5]:  # max 5
            preview = chunk.get("preview", "")[:100]
            print(f"   {C.DIM}[{chunk.get('idx')}] {preview}...{C.END}")

    elif node_name == "worker":
        answers = log.get("partial_answers", [])
        if answers:
            print(f"   {C.GREEN}Respuesta parcial:{C.END}")
            print(f"   {C.DIM}{answers[0][:200]}...{C.END}" if len(answers[0]) > 200 else f"   {C.DIM}{answers[0]}{C.END}")

    elif node_name == "worker_sc":
        print(f"   {C.YELLOW}Candidatos generados: {log.get('n_candidates', 0)}{C.END}")
        print(f"   {C.GREEN}Seleccionado: #{log.get('selected', 1)}{C.END}")

    elif node_name == "critic":
        if log.get("action") == "approve":
            print(f"   {C.GREEN}✅ APPROVED{C.END}")
            print(f"   Grounded: {log.get('grounded')} | Complete: {log.get('complete')} | Accurate: {log.get('accurate')}")
        else:
            print(f"   {C.RED}🔄 NEEDS_REVISION{C.END}")
            print(f"   Issues: {log.get('issues', [])}")


def run_verbose(question: str, use_critic: bool = True):
    """Ejecuta con output verbose paso a paso."""
    print(f"\n{C.BOLD}❓ {question}{C.END}")
    print(f"{C.DIM}{'─'*60}{C.END}")

    result = run_query(question, use_critic=use_critic, save_logs=True)

    # Imprimir cada paso
    for log in result.get("logs", []):
        print_verbose_step(log.get("node", "unknown"), log)

    # Respuesta final
    print(f"\n{C.BOLD}{'─'*60}")
    print(f"📝 RESPUESTA FINAL:{C.END}")
    print(f"{result.get('final_answer', 'Sin respuesta')}")

    return result


def run_stream(question: str, use_critic: bool = True):
    """Ejecuta con streaming en tiempo real."""
    print(f"\n{C.BOLD}❓ {question}{C.END}")
    print(f"{C.DIM}{'─'*60}{C.END}")

    final_result = {}
    for event in run_query_stream(question, use_critic=use_critic):
        # event es dict con nombre del nodo como key
        for node_name, state in event.items():
            logs = state.get("logs", [])
            if logs:
                print_verbose_step(node_name, logs[-1])
            final_result = state

    # Respuesta final
    print(f"\n{C.BOLD}{'─'*60}")
    print(f"📝 RESPUESTA FINAL:{C.END}")
    print(f"{final_result.get('final_answer', 'Sin respuesta')}")

    return final_result


def show_last_logs(n: int = 1):
    """Muestra los últimos n logs guardados."""
    log_file = Path(__file__).parent.parent / "out" / "query_logs.jsonl"

    if not log_file.exists():
        print(f"{C.RED}No hay logs guardados.{C.END}")
        return

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        print(f"{C.RED}No hay logs guardados.{C.END}")
        return

    # Últimos n logs
    for line in lines[-n:]:
        entry = json.loads(line)
        print(f"\n{C.BOLD}{'='*60}")
        print(f"📋 LOG: {entry.get('timestamp', 'N/A')}{C.END}")
        print(f"{C.CYAN}Q: {entry.get('question')}{C.END}")
        print(f"\n{C.YELLOW}Subtareas:{C.END} {entry.get('subtasks', [])}")

        # Chunks
        print(f"\n{C.GREEN}Chunks recuperados ({entry.get('n_context', 0)}):{C.END}")
        for i, chunk in enumerate(entry.get("context_full", [])[:3], 1):
            print(f"  [{i}] {chunk[:150]}...")

        # Logs de cada nodo
        print(f"\n{C.BLUE}Ejecución de nodos:{C.END}")
        for log in entry.get("logs", []):
            node = log.get("node", "?")
            thinking = log.get("thinking", "")
            print(f"  {node}: {thinking}")

        # Critic review
        review = entry.get("critic_review")
        if review:
            print(f"\n{C.CYAN}Critic Review:{C.END}")
            print(f"  Verdict: {review.get('verdict')}")
            print(f"  Grounded: {review.get('grounded')} | Complete: {review.get('complete')} | Accurate: {review.get('accurate')}")
            if review.get("issues"):
                print(f"  Issues: {review['issues']}")

        # Respuesta
        print(f"\n{C.BOLD}Respuesta:{C.END}")
        print(f"{entry.get('final_answer', 'N/A')}")


def demo_questions(mode: str = "normal"):
    """Demo con preguntas predefinidas."""
    questions = [
        "¿Cuáles son los prerequisitos de CC0C2?",
        "¿Qué temas cubre el curso de NLP?",
        "¿Cuántos créditos tiene el curso de Minería de Datos?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n{C.BOLD}[{i}/{len(questions)}]{C.END}")

        if mode == "verbose":
            run_verbose(q)
        elif mode == "stream":
            run_stream(q)
        else:
            result = run_query(q, use_critic=True)
            print(f"Q: {q}")
            print(f"A: {result.get('final_answer', '')[:300]}")
            review = result.get("critic_review", {})
            if review:
                print(f"Critic: {review.get('verdict')} | Grounded: {review.get('grounded')}")

        print()


def interactive_mode(mode: str = "verbose"):
    """Modo interactivo con verbose o stream."""
    print(f"\n{C.CYAN}Modo interactivo ({mode}). Escribe 'salir' para terminar.{C.END}\n")

    while True:
        try:
            q = input(f"{C.BOLD}❓ Pregunta: {C.END}").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if q.lower() in ["salir", "exit", "q", ""]:
            break

        if mode == "verbose":
            run_verbose(q)
        elif mode == "stream":
            run_stream(q)
        else:
            result = run_query(q, use_critic=True)
            print(f"\n📝 {result.get('final_answer', '')}")

        print()


def print_usage():
    """Muestra ayuda de uso."""
    print(f"""
{C.BOLD}USO:{C.END}
  python demo.py                    Demo básico con preguntas predefinidas
  python demo.py --verbose          Demo con output paso a paso
  python demo.py --stream           Demo con streaming en tiempo real
  python demo.py --interactive      Modo interactivo (verbose)
  python demo.py --show-logs [n]    Muestra últimos n logs (default: 1)
  python demo.py --help             Muestra esta ayuda

{C.BOLD}EJEMPLOS:{C.END}
  python demo.py --verbose
  python demo.py --stream
  python demo.py --show-logs 3
  python demo.py --interactive
""")


if __name__ == "__main__":
    print_header()

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "--help" or arg == "-h":
            print_usage()

        elif arg == "--verbose":
            demo_questions("verbose")

        elif arg == "--stream":
            demo_questions("stream")

        elif arg == "--interactive":
            mode = sys.argv[2] if len(sys.argv) > 2 else "verbose"
            interactive_mode(mode)

        elif arg == "--show-logs":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            show_last_logs(n)

        else:
            print(f"{C.RED}Opción desconocida: {arg}{C.END}")
            print_usage()
    else:
        demo_questions("verbose")
