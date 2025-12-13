# Informe E1: Multi-Agent QA System (R8)

**Curso:** CC0C2 - Procesamiento de Lenguaje Natural
**Proyecto:** R8 - Multi-Agent QA con RAG
**Fecha:** Diciembre 2025

---

## 1. Objetivo

Construir un sistema multi-agente para QA académico que permita responder preguntas sobre sílabos universitarios. El sistema utiliza:

- **LangGraph** para orquestar el flujo de agentes
- **RAG (Retrieval-Augmented Generation)** con ChromaDB para recuperar contexto relevante
- **GPT-4.1-mini** como modelo generativo

## 2. Arquitectura

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Manager   │ ──▶ │  Retriever  │ ──▶ │   Worker    │ ──▶ │ Synthesizer │
│ (descompone)│     │   (RAG)     │     │ (responde)  │     │ (consolida) │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
   Subtareas          ChromaDB            LLM calls           Respuesta
                    + Embeddings                               final
```

### 2.1 Componentes

| Componente | Descripción | Tecnología |
|------------|-------------|------------|
| **Manager** | Descompone preguntas complejas en subtareas | GPT-4.1-mini |
| **Retriever** | Recupera documentos relevantes del corpus | ChromaDB + all-MiniLM-L6-v2 |
| **Worker** | Genera respuestas basadas en contexto | GPT-4.1-mini |
| **Synthesizer** | Consolida respuestas parciales | GPT-4.1-mini |

### 2.2 Estado del Grafo (TypedDict)

```python
class AgentState(TypedDict):
    question: str           # Pregunta original
    subtasks: List[str]     # Subtareas del Manager
    context: List[str]      # Documentos recuperados
    partial_answers: List[str]  # Respuestas de Workers
    final_answer: str       # Respuesta consolidada
    logs: List[dict]        # Trazas de ejecución
```

## 3. Implementación

### 3.1 Corpus

PDFs de sílabos reales procesados con PyMuPDF:

| Documento | Chunks |
|-----------|--------|
| CC0C2 - Procesamiento de Lenguaje Natural | 18 |
| CC842 - Minería de Datos | 8 |
| Cloud Computing | 3 |
| Robótica | 2 |
| **Total** | **31** |

### 3.2 RAG Pipeline

1. **Ingesta**: Extracción de texto de PDFs con PyMuPDF
2. **Chunking**: División por párrafos (~800 chars) con overlap de 150
3. **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
4. **Almacenamiento**: ChromaDB con distancia coseno
5. **Retrieval**: Top-k (k=3) documentos por subtarea

### 3.3 Flujo de Ejecución

1. Usuario envía pregunta
2. **Manager** analiza y descompone (si es compleja)
3. **Retriever** busca contexto para cada subtarea
4. **Worker** genera respuesta por subtarea usando contexto
5. **Synthesizer** consolida en respuesta final
6. Se guarda log JSON de la interacción

## 4. Resultados

### 4.1 Ejemplos de Ejecución

**Pregunta 1:** ¿Cuáles son los prerequisitos de CC0C2?

> El curso CC0C2, denominado "Procesamiento de Lenguaje Natural" de la Escuela Profesional de Ciencia de la Computación de la UNI, tiene como prerequisito el curso CC421. Es un curso electivo de 4 créditos (2h teoría, 4h práctica).

**Pregunta 2:** ¿Qué temas cubre el curso de NLP?

> El curso está dividido en cinco unidades: técnicas básicas de NLP (tokenización, normalización, n-gramas), redes neuronales (RNNs, Transformers), arquitecturas avanzadas de LLMs, y aplicaciones prácticas con Hugging Face.

### 4.2 Métricas de Ejecución

| Métrica | Valor |
|---------|-------|
| Documentos en corpus | 31 chunks |
| Nodos por query | 4 |
| Docs recuperados (promedio) | 3-6 |
| Tiempo respuesta | ~3-5 segundos |

### 4.3 Limitaciones Observadas

1. **Retrieval impreciso**: A veces recupera chunks de cursos incorrectos
2. **Sin validación**: No hay verificación de que la respuesta use el contexto correctamente
3. **Alucinaciones**: El modelo puede inventar información no presente en el contexto

## 5. Trabajo Futuro (E2)

Para el Entregable 2 se planea:

1. **Nodo Critic**: Validar que las respuestas usen el contexto recuperado
2. **Self-consistency**: Generar múltiples respuestas y elegir la más coherente
3. **Chain-of-thought**: Documentar el razonamiento paso a paso
4. **Métricas**: Comparar tasa de errores con/sin Critic

## 6. Estructura del Proyecto

```
final/E1/
├── src/
│   ├── state.py        # Estado TypedDict
│   ├── nodes.py        # Nodos del grafo
│   ├── graph.py        # Construcción del grafo LangGraph
│   ├── rag.py          # RAG con ChromaDB
│   ├── llm.py          # Cliente OpenAI
│   ├── ingest_pdfs.py  # Ingesta de PDFs
│   └── demo.py         # Script de demostración
├── data/
│   └── raw/            # PDFs originales
├── out/
│   ├── chroma_db/      # Base vectorial
│   └── query_logs.jsonl
└── docs/
    └── informe_e1.md
```

## 7. Ejecución

```bash
# Instalar dependencias
cd final/E1
uv venv && uv pip install -e .

# Ingestar PDFs (primera vez)
cd src && python ingest_pdfs.py

# Opción 1: LangGraph Studio (recomendado)
langgraph dev --tunnel

# Opción 2: Demo por terminal
python -m src.demo --verbose
```

### LangGraph Studio

El sistema se puede visualizar en tiempo real usando LangGraph Studio, que muestra:
- Grafo de nodos interactivo
- Estado en cada paso de ejecución
- Logs detallados de cada agente
- Chunks recuperados del RAG

## 8. Conclusiones

Se logró implementar un sistema multi-agente funcional que:

- Procesa PDFs de sílabos reales
- Descompone preguntas complejas en subtareas
- Recupera contexto relevante usando RAG
- Genera respuestas coherentes con GPT-4.1-mini
- Registra trazas de ejecución para auditoría

El sistema demuestra el flujo básico Manager → Retriever → Worker → Synthesizer, sentando las bases para añadir el Critic en E2.
