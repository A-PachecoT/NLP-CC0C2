# Informe E2: Multi-Agent QA System con Critic (R8)

**Curso:** CC0C2 - Procesamiento de Lenguaje Natural
**Proyecto:** R8 - Multi-Agent QA con RAG + Critic
**Fecha:** Diciembre 2025

---

## 1. Resumen Ejecutivo

Este proyecto implementa un sistema multi-agente para QA académico usando **LangGraph**, **RAG** y un **agente Critic** que valida las respuestas. El sistema responde preguntas sobre sílabos y reglamentos universitarios, con capacidad de detectar y corregir respuestas no fundamentadas en el contexto.

**Mejoras sobre E1:**
- Nodo **Critic** que verifica groundedness, completeness y accuracy
- **Chain-of-thought** en el Worker para razonamiento explícito
- **Self-consistency**: genera múltiples respuestas y selecciona la mejor
- **Benchmark** comparativo con métricas de evaluación

## 2. Arquitectura E2

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Manager   │ ──▶ │  Retriever  │ ──▶ │   Worker    │ ──▶ │ Synthesizer │ ──▶ │   Critic    │
│ (descompone)│     │   (RAG)     │     │ (responde)  │     │ (consolida) │     │  (valida)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼                   ▼
   Subtareas          ChromaDB          Chain-of-         Respuesta          APPROVED o
                    + Embeddings         Thought            final           NEEDS_REVISION
                                                                                  │
                                                                                  ▼
                                                                           (si revision)
                                                                           Revise Answer
```

### 2.1 Nuevo Componente: Critic

El Critic evalúa cada respuesta en 3 dimensiones:

| Dimensión | Descripción |
|-----------|-------------|
| **Grounded** | ¿La respuesta está fundamentada en el contexto? (no inventa) |
| **Complete** | ¿Responde completamente la pregunta? |
| **Accurate** | ¿Los datos citados son correctos según el contexto? |

**Output del Critic:**
```json
{
    "grounded": true,
    "complete": true,
    "accurate": true,
    "issues": [],
    "verdict": "APPROVED",
    "suggestion": ""
}
```

Si `verdict == "NEEDS_REVISION"`, el sistema genera una respuesta corregida.

### 2.2 Técnicas de Razonamiento

#### Chain-of-Thought (CoT)
El Worker usa un prompt que fuerza razonamiento explícito:
1. Identificar información relevante del contexto
2. Extraer datos específicos
3. Formular respuesta clara

#### Self-Consistency
Genera 3 respuestas con diferentes temperaturas (0.3, 0.5, 0.7) y un "juez" selecciona la mejor basándose en:
- Fundamentación en contexto
- Precisión
- No inventa información

## 3. Implementación

### 3.1 Estructura del Código

```
E2/src/
├── state.py      # Estado con critic_review
├── nodes.py      # +critic_node, +worker_self_consistency_node
├── graph.py      # build_graph(use_critic, use_self_consistency)
├── llm.py        # +critic_review, +revise_answer, +generate_multiple_answers
├── rag.py        # Sin cambios desde E1
├── demo.py       # Demo con Critic
└── benchmark.py  # Comparación con/sin Critic
```

### 3.2 Grafo Configurable

```python
def build_graph(use_critic: bool = True, use_self_consistency: bool = False):
    graph = StateGraph(AgentState)
    graph.add_node("manager", manager_node)
    graph.add_node("retriever", retriever_node)

    if use_self_consistency:
        graph.add_node("worker", worker_self_consistency_node)
    else:
        graph.add_node("worker", worker_node)

    graph.add_node("synthesizer", synthesizer_node)

    if use_critic:
        graph.add_node("critic", critic_node)
        graph.add_edge("synthesizer", "critic")
        graph.add_edge("critic", END)
    else:
        graph.add_edge("synthesizer", END)
```

## 4. Experimentos y Resultados

### 4.1 Benchmark

Se evaluaron 7 preguntas con ground truth conocido sobre los sílabos reales (NLP, Minería de Datos, Cloud, Robótica):

| Configuración | Accuracy | Tiempo Promedio |
|---------------|----------|-----------------|
| Baseline (sin Critic) | 71.4% (5/7) | 9.6s |
| Con Critic | 71.4% (5/7) | 10.6s |
| Self-Consistency | 57.1% (4/7) | 7.5s |
| Critic + SC | 57.1% (4/7) | 8.9s |

### 4.2 Análisis de Resultados

**Hallazgos principales:**

1. **Baseline y Critic tienen igual accuracy (71.4%)**: El Critic valida correctamente pero no mejora respuestas cuando el retrieval falla.

2. **Self-Consistency tiene peor accuracy (57.1%)**: Generar múltiples respuestas con temperaturas altas (0.5, 0.7) introduce más alucinaciones. El juez LLM no siempre selecciona la mejor.

3. **Self-Consistency es más rápido (7.5s vs 9.6s)**: Las respuestas son más cortas sin Chain-of-Thought extenso.

**Preguntas que fallan consistentemente:**
1. "¿Cuántos créditos tiene el curso de NLP?" - El chunk no contiene "4 créditos" explícitamente
2. "¿Cuántas unidades tiene el curso?" - Responde "4" en vez de "5" (alucinación)

**Conclusión:** El Critic detecta correctamente cuando las respuestas están fundamentadas, pero no puede mejorar respuestas si el contexto recuperado es incorrecto. Self-Consistency introduce variabilidad que puede ser contraproducente en RAG.

### 4.3 Ejemplos de Critic en Acción

**Respuesta APPROVED:**
```
Q: ¿Cuáles son los prerequisitos de CC0C2?
R: El prerequisito es el curso CC421.
Critic: APPROVED | Grounded: True | Accurate: True
```

**Respuesta con Issues (hipotético):**
```
Q: ¿Cuántos créditos tiene CC0C2?
R: CC0C2 tiene 5 créditos.  # Incorrecto - son 4
Critic: NEEDS_REVISION
Issues: ["El contexto indica 4 créditos, no 5"]
```

## 5. Tipos de Razonamiento

El sistema exhibe diferentes tipos de razonamiento según la pregunta:

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **Deductivo** | Combinar hechos explícitos | "CC0C2 requiere CC421 → Extraer info del sílabo" |
| **Inductivo** | Detectar patrones | "Los cursos tienen estructura de unidades → Patrón de organización" |
| **Abductivo** | Explicar con hipótesis | "No hay info de inglés → Posiblemente en otro documento" |

El Critic ayuda a identificar cuando el razonamiento se sale del contexto (alucinación).

## 6. Relación con Técnicas del Curso

### 6.1 RAG (Retrieval-Augmented Generation)
- Reduce alucinaciones al anclar respuestas en documentos
- Trade-off: costo de retrieval vs calidad
- Limitación: depende de la calidad del chunking

### 6.2 Multi-Agent Systems
- Manager: descomposición de tareas (divide & conquer)
- Workers: especialización (cada uno responde una subtarea)
- Critic: verificación (quality assurance)

### 6.3 Alineamiento (RLHF-lite)
El Critic actúa como un "reward model" simplificado:
- Evalúa groundedness (similar a verificar factualidad)
- Puede solicitar revisión (similar a iteración de preferencias)
- Sin entrenamiento, pero con evaluación en tiempo de inferencia

## 7. Métricas de Costo

| Métrica | Baseline | Con Critic | Self-Consistency |
|---------|----------|------------|------------------|
| Llamadas LLM/query | 3-4 | 4-5 | 5-6 |
| Tokens promedio | ~2000 | ~3000 | ~2500 |
| Latencia | 9.6s | 10.6s | 7.5s |
| Overhead vs Baseline | - | +10% | -22% |

## 8. Limitaciones y Trabajo Futuro

### Limitaciones Actuales
1. **Retrieval:** El RAG no siempre recupera los chunks correctos
2. **Chunking:** División por secciones puede separar info relacionada
3. **Sin reintento de retrieval:** El Critic no puede pedir más contexto

### Mejoras Propuestas
1. **Hybrid retrieval:** Combinar BM25 + embeddings
2. **Query rewriting:** Reformular consultas para mejor retrieval
3. **Critic con re-retrieval:** Si falla validación, buscar más contexto
4. **Fine-tuning del retriever:** Adaptar embeddings al dominio

## 9. Conclusiones

1. **LangGraph** facilita la orquestación de agentes como grafo de estados
2. El **Critic** añade una capa de verificación que detecta respuestas no fundamentadas (mismo accuracy, +10% latencia)
3. El **cuello de botella** está en el retrieval, no en la generación
4. **Self-Consistency** reduce latencia (-22%) pero introduce alucinaciones por temperaturas altas (-14% accuracy)
5. **Chain-of-Thought** mejora la interpretabilidad mostrando el razonamiento paso a paso
6. El sistema demuestra el patrón **Manager + Workers + Critic** aplicable a QA empresarial

## 10. Referencias

- LangGraph Documentation: https://python.langchain.com/docs/langgraph
- ChromaDB: https://docs.trychroma.com
- OpenAI API: https://platform.openai.com/docs
- RAG Paper: Lewis et al. (2020) "Retrieval-Augmented Generation"
