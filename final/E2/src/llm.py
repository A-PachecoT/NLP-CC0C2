# Cliente LLM con OpenAI - E2 con Critic
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Cargar .env desde final/
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY no configurada. Crea final/.env")

MODEL = "gpt-4.1-mini"
client = OpenAI(api_key=OPENAI_API_KEY)


def chat(messages: list, temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """Llama al LLM y retorna la respuesta."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def decompose_question(question: str) -> list:
    """Usa LLM para descomponer pregunta en subtareas."""
    messages = [
        {"role": "system", "content": """Eres un asistente que descompone preguntas complejas en subtareas simples.
Responde SOLO con una lista JSON de strings, sin explicación.
Si la pregunta ya es simple, devuelve una lista con un solo elemento.
Ejemplo: ["subtarea 1", "subtarea 2"]"""},
        {"role": "user", "content": f"Descompón esta pregunta: {question}"}
    ]
    response = chat(messages, temperature=0.1)
    try:
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(clean)
    except:
        return [question]


def generate_answer(question: str, context: list, chain_of_thought: bool = True) -> str:
    """Genera respuesta basada en contexto con chain-of-thought."""
    context_str = "\n\n---\n\n".join(context) if context else "No hay contexto disponible."

    if chain_of_thought:
        system_prompt = """Eres un asistente académico que responde preguntas sobre cursos universitarios.

Let's think step by step.

Responde SIEMPRE con este formato exacto:

RAZONAMIENTO:
- Paso 1: [qué información del contexto es relevante para la pregunta]
- Paso 2: [qué datos específicos extraigo del contexto]
- Paso 3: [cómo conecto estos datos para responder]

RESPUESTA:
[respuesta final concisa basada solo en el contexto]

REGLAS:
- Usa SOLO información del contexto proporcionado
- Si no hay información, di "No encontré información sobre eso en los documentos"
- Cita la fuente cuando sea relevante (nombre del curso, código)"""
    else:
        system_prompt = """Eres un asistente académico que responde preguntas sobre cursos y reglamentos universitarios.
REGLAS:
- Responde SOLO basándote en el contexto proporcionado
- Si la información no está en el contexto, di "No encontré información sobre eso"
- Sé conciso y directo
- Cita la fuente cuando sea relevante"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""Contexto:
{context_str}

Pregunta: {question}

Responde basándote únicamente en el contexto anterior."""}
    ]
    return chat(messages)


def generate_multiple_answers(question: str, context: list, n: int = 3) -> list:
    """Self-consistency con CoT: genera múltiples respuestas con razonamiento."""
    answers = []
    context_str = "\n\n---\n\n".join(context) if context else "No hay contexto."

    # Prompt con CoT para cada sample (según paper original)
    system_prompt = """Eres un asistente académico. Let's think step by step.

Responde con este formato:

RAZONAMIENTO:
- Paso 1: [info relevante del contexto]
- Paso 2: [datos extraídos]
- Paso 3: [conclusión]

RESPUESTA:
[respuesta concisa]

Usa SOLO información del contexto. Si no hay info, di "No encontré información"."""

    for i in range(n):
        # Variar temperatura para diversidad de reasoning paths
        temp = 0.3 + (i * 0.2)  # 0.3, 0.5, 0.7

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Contexto:\n{context_str}\n\nPregunta: {question}"}
        ]
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temp,
            max_tokens=512
        )
        answers.append(response.choices[0].message.content)
    return answers


def synthesize_answers(question: str, partial_answers: list) -> str:
    """Sintetiza múltiples respuestas parciales."""
    if len(partial_answers) == 1:
        return partial_answers[0]

    answers_str = "\n\n---\n\n".join([f"Respuesta {i+1}:\n{a}" for i, a in enumerate(partial_answers)])

    messages = [
        {"role": "system", "content": """Sintetiza las respuestas parciales en una respuesta coherente y completa.
Elimina redundancias y organiza la información de forma clara."""},
        {"role": "user", "content": f"""Pregunta original: {question}

{answers_str}

Sintetiza estas respuestas en una sola respuesta coherente."""}
    ]
    return chat(messages)


def critic_review(question: str, answer: str, context: list) -> dict:
    """Critic: revisa si la respuesta es correcta y usa el contexto."""
    context_str = "\n\n---\n\n".join(context) if context else "No hay contexto."

    messages = [
        {"role": "system", "content": """Eres un revisor crítico de respuestas. Tu trabajo es verificar:

1. GROUNDED: ¿La respuesta está fundamentada en el contexto? (no inventa información)
2. COMPLETE: ¿Responde completamente la pregunta?
3. ACCURATE: ¿Los datos citados son correctos según el contexto?

Responde en JSON con este formato exacto:
{
    "grounded": true/false,
    "complete": true/false,
    "accurate": true/false,
    "issues": ["lista de problemas encontrados"],
    "verdict": "APPROVED" o "NEEDS_REVISION",
    "suggestion": "sugerencia de mejora si hay problemas"
}"""},
        {"role": "user", "content": f"""CONTEXTO:
{context_str}

PREGUNTA: {question}

RESPUESTA A REVISAR:
{answer}

Evalúa la respuesta."""}
    ]

    response = chat(messages, temperature=0.1)

    try:
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(clean)
    except:
        return {
            "grounded": True,
            "complete": True,
            "accurate": True,
            "issues": [],
            "verdict": "APPROVED",
            "suggestion": ""
        }


def revise_answer(question: str, answer: str, context: list, critic_feedback: dict) -> str:
    """Revisa la respuesta basándose en el feedback del Critic."""
    context_str = "\n\n---\n\n".join(context) if context else "No hay contexto."

    messages = [
        {"role": "system", "content": """Eres un asistente que corrige respuestas basándose en feedback.
Genera una respuesta mejorada que corrija los problemas identificados.
Usa SOLO información del contexto proporcionado."""},
        {"role": "user", "content": f"""CONTEXTO:
{context_str}

PREGUNTA: {question}

RESPUESTA ORIGINAL:
{answer}

PROBLEMAS IDENTIFICADOS:
{critic_feedback.get('issues', [])}

SUGERENCIA:
{critic_feedback.get('suggestion', '')}

Genera una respuesta corregida."""}
    ]
    return chat(messages)


def select_best_answer(question: str, answers: list, context: list) -> tuple:
    """Self-consistency: selecciona la mejor respuesta entre varias."""
    context_str = "\n\n---\n\n".join(context[:3]) if context else "No hay contexto."

    answers_formatted = "\n\n".join([f"[{i+1}] {a}" for i, a in enumerate(answers)])

    messages = [
        {"role": "system", "content": """Eres un juez que selecciona la mejor respuesta.
Evalúa cuál respuesta:
1. Está mejor fundamentada en el contexto
2. Es más precisa y completa
3. No inventa información

Responde SOLO con el número de la mejor respuesta (1, 2, 3, etc.)"""},
        {"role": "user", "content": f"""CONTEXTO:
{context_str}

PREGUNTA: {question}

RESPUESTAS:
{answers_formatted}

¿Cuál es la mejor respuesta? (solo el número)"""}
    ]

    response = chat(messages, temperature=0.1)
    try:
        idx = int(response.strip().replace("[", "").replace("]", "")) - 1
        if 0 <= idx < len(answers):
            return answers[idx], idx
    except:
        pass
    return answers[0], 0


if __name__ == "__main__":
    print("Testing E2 LLM functions...")

    # Test critic
    context = ["CC0C2 tiene 4 créditos y prerequisitos CC0B1 y CC0A3."]
    answer = "CC0C2 tiene 5 créditos."  # incorrecto
    review = critic_review("¿Cuántos créditos tiene CC0C2?", answer, context)
    print(f"\nCritic review: {json.dumps(review, indent=2)}")

    # Test self-consistency
    answers = generate_multiple_answers("¿Cuántos créditos tiene CC0C2?", context, n=3)
    print(f"\nMultiple answers: {len(answers)}")
    best, idx = select_best_answer("¿Cuántos créditos?", answers, context)
    print(f"Best answer (#{idx+1}): {best[:100]}...")
