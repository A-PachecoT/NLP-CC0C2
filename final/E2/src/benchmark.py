#!/usr/bin/env python3
"""Benchmark: comparar sistema con y sin Critic."""
import json
import time
from pathlib import Path
from .graph import run_query


# Preguntas basadas en el contenido real de los PDFs
TEST_QUESTIONS = [
    {
        "question": "¿Cuáles son los prerequisitos de CC0C2?",
        "expected_keywords": ["CC421"],
        "should_not_contain": []
    },
    {
        "question": "¿Cuántos créditos tiene el curso de NLP?",
        "expected_keywords": ["4"],
        "should_not_contain": ["3", "5"]
    },
    {
        "question": "¿Qué temas cubre la Unidad 2 del curso de NLP?",
        "expected_keywords": ["RNN", "Transformer"],
        "should_not_contain": []
    },
    {
        "question": "¿Qué es RAG según el curso de NLP?",
        "expected_keywords": ["Recuperación", "RAG"],
        "should_not_contain": []
    },
    {
        "question": "¿Cuántos créditos tiene el curso de Minería de Datos?",
        "expected_keywords": ["4"],
        "should_not_contain": ["3", "5"]
    },
    {
        "question": "¿Qué herramientas se usan en el curso de NLP?",
        "expected_keywords": ["Hugging Face", "PyTorch"],
        "should_not_contain": []
    },
    {
        "question": "¿Cuántas unidades tiene el curso de Procesamiento de Lenguaje Natural?",
        "expected_keywords": ["cinco", "5"],
        "should_not_contain": ["3", "4", "6"]
    },
]


def evaluate_answer(answer: str, expected: dict) -> dict:
    """Evalúa una respuesta contra ground truth."""
    answer_lower = answer.lower()

    # Contar keywords encontrados
    found_keywords = []
    missing_keywords = []
    for kw in expected["expected_keywords"]:
        if kw.lower() in answer_lower:
            found_keywords.append(kw)
        else:
            missing_keywords.append(kw)

    # Verificar contenido incorrecto
    incorrect_content = []
    for bad in expected.get("should_not_contain", []):
        if bad.lower() in answer_lower:
            incorrect_content.append(bad)

    # Calcular score
    if expected["expected_keywords"]:
        keyword_score = len(found_keywords) / len(expected["expected_keywords"])
    else:
        keyword_score = 1.0

    has_hallucination = len(incorrect_content) > 0

    return {
        "keyword_score": keyword_score,
        "found_keywords": found_keywords,
        "missing_keywords": missing_keywords,
        "has_hallucination": has_hallucination,
        "incorrect_content": incorrect_content,
        "is_correct": keyword_score >= 0.5 and not has_hallucination
    }


def run_benchmark(use_critic: bool, use_self_consistency: bool = False):
    """Ejecuta benchmark con configuración específica."""
    results = []

    config_name = []
    if use_critic:
        config_name.append("Critic")
    if use_self_consistency:
        config_name.append("SelfConsistency")
    if not config_name:
        config_name = ["Baseline"]
    config_str = "+".join(config_name)

    print(f"\n{'='*60}")
    print(f"  Benchmark: {config_str}")
    print("="*60)

    total_time = 0
    correct_count = 0

    for i, test in enumerate(TEST_QUESTIONS, 1):
        q = test["question"]
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {q[:50]}...")

        start = time.time()
        result = run_query(q, save_logs=False, use_critic=use_critic,
                          use_self_consistency=use_self_consistency)
        elapsed = time.time() - start
        total_time += elapsed

        answer = result.get("final_answer", "")
        eval_result = evaluate_answer(answer, test)

        emoji = "✅" if eval_result["is_correct"] else "❌"
        print(f"   {emoji} Score: {eval_result['keyword_score']:.1%} | Time: {elapsed:.1f}s")
        if eval_result["missing_keywords"]:
            print(f"   Missing: {eval_result['missing_keywords']}")
        if eval_result["has_hallucination"]:
            print(f"   ⚠️ Hallucination: {eval_result['incorrect_content']}")

        if eval_result["is_correct"]:
            correct_count += 1

        results.append({
            "question": q,
            "answer": answer[:300],
            "eval": eval_result,
            "time": elapsed,
            "critic_review": result.get("critic_review"),
            "n_nodes": len(result.get("logs", []))
        })

    # Resumen
    accuracy = correct_count / len(TEST_QUESTIONS)
    avg_time = total_time / len(TEST_QUESTIONS)

    print(f"\n{'─'*60}")
    print(f"📊 RESUMEN {config_str}:")
    print(f"   Accuracy: {accuracy:.1%} ({correct_count}/{len(TEST_QUESTIONS)})")
    print(f"   Tiempo promedio: {avg_time:.1f}s")
    print(f"   Tiempo total: {total_time:.1f}s")

    return {
        "config": config_str,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(TEST_QUESTIONS),
        "avg_time": avg_time,
        "total_time": total_time,
        "results": results
    }


def main():
    print("\n" + "="*60)
    print("  BENCHMARK: Comparación con/sin Critic")
    print("="*60)

    # Ejecutar benchmarks
    results = {}

    # Sin Critic (baseline)
    results["baseline"] = run_benchmark(use_critic=False)

    # Con Critic
    results["critic"] = run_benchmark(use_critic=True)

    # Con Self-Consistency
    results["self_consistency"] = run_benchmark(use_critic=False, use_self_consistency=True)

    # Con ambos
    results["critic_sc"] = run_benchmark(use_critic=True, use_self_consistency=True)

    # Tabla comparativa
    print("\n" + "="*60)
    print("  COMPARACIÓN FINAL")
    print("="*60)
    print(f"\n{'Config':<25} {'Accuracy':<12} {'Tiempo':<12}")
    print("-"*50)
    for name, data in results.items():
        print(f"{data['config']:<25} {data['accuracy']:.1%} ({data['correct']}/{data['total']})  {data['avg_time']:.1f}s")

    # Guardar resultados
    out_path = Path(__file__).parent.parent / "out" / "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        # Simplificar para JSON
        summary = {k: {
            "config": v["config"],
            "accuracy": v["accuracy"],
            "correct": v["correct"],
            "total": v["total"],
            "avg_time": v["avg_time"]
        } for k, v in results.items()}
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Resultados guardados en {out_path}")


if __name__ == "__main__":
    main()
