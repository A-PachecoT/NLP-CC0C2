# Cliente LLM con OpenAI
import os
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

# Modelo
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
    # Parsear JSON
    import json
    try:
        # Limpiar respuesta
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(clean)
    except:
        return [question]


def generate_answer(question: str, context: list) -> str:
    """Genera respuesta basada en contexto."""
    context_str = "\n\n---\n\n".join(context) if context else "No hay contexto disponible."

    messages = [
        {"role": "system", "content": """Eres un asistente académico que responde preguntas sobre cursos y reglamentos universitarios.
REGLAS:
- Responde SOLO basándote en el contexto proporcionado
- Si la información no está en el contexto, di "No encontré información sobre eso"
- Sé conciso y directo
- Cita la fuente cuando sea relevante (nombre del curso, sección del reglamento)"""},
        {"role": "user", "content": f"""Contexto:
{context_str}

Pregunta: {question}

Responde basándote únicamente en el contexto anterior."""}
    ]
    return chat(messages)


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


if __name__ == "__main__":
    # Test rápido
    print("Testing LLM...")

    # Test decompose
    q = "¿Cuáles son los prerequisitos de CC0C2 y cuántos créditos tiene?"
    subtasks = decompose_question(q)
    print(f"\nPregunta: {q}")
    print(f"Subtareas: {subtasks}")

    # Test generate
    context = ["CC0C2 tiene 4 créditos y requiere CC0B1 y CC0A3 como prerequisitos."]
    answer = generate_answer("¿Cuántos créditos tiene CC0C2?", context)
    print(f"\nRespuesta: {answer}")
