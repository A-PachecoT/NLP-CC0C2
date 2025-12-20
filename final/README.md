# Examen Final CC0C2 - R8: Multi-Agent QA

Sistema multi-agente para QA académico usando **RAG + LangGraph + GPT-4.1-mini**.

## Arquitectura

```
Pregunta → [Manager] → [Retriever] → [Worker] → [Synthesizer] → Respuesta
                            ↓
                       ChromaDB
                    (PDFs reales)
```

**E2 añade:**
```
... → [Synthesizer] → [Critic] → Respuesta validada
                         ↓
                  (revisión si error)
```

## Corpus

PDFs de sílabos reales en `data/raw/`:
- CC0C2 - Procesamiento de Lenguaje Natural (18 chunks)
- CC842 - Minería de Datos (8 chunks)
- Cloud Computing (3 chunks)
- Robótica (2 chunks)

**Total: 31 chunks indexados en ChromaDB**

## Instalación

```bash
cd final
uv venv
uv sync
```

## Uso

```bash
# 1. Ingestar PDFs (solo primera vez)
cd E1/src && uv run python ingest_pdfs.py

# 2. Ejecutar demo E1
uv run python demo.py

# 3. Para E2 (con Critic)
cd ../../E2/src
uv run python ingest_pdfs.py
uv run python demo.py
```

## Ejemplo

```
Q: ¿Cuáles son los prerequisitos de CC0C2?
R: El prerequisito es el curso CC421. Es un curso electivo
   de 4 créditos (2h teoría, 4h práctica).

Critic: ✅ APPROVED (grounded: true, accurate: true)
```

## Entregables

### E1 (13 dic)
- [x] Estructura proyecto
- [x] RAG con PDFs reales
- [x] Grafo Manager → Retriever → Worker → Synthesizer
- [x] Logging JSON
- [x] Informe 2-3 págs
- [x] Video 5-8 min

### E2 (20 dic)
- [x] Nodo Critic
- [x] Chain-of-thought + Self-consistency
- [x] Benchmark comparativo
- [x] Informe 6-10 págs
- [ ] Video + exposición

## Estructura

```
final/
├── E1/
│   ├── src/           # graph.py, nodes.py, rag.py, llm.py
│   ├── data/raw/      # PDFs originales
│   ├── out/           # ChromaDB, logs
│   └── docs/          # informe_e1.md
├── E2/
│   ├── src/           # + critic_node, benchmark.py
│   ├── data/raw/      # PDFs
│   ├── out/           # ChromaDB, benchmark_results.json
│   └── docs/          # informe_e2.md
└── experiments/docs/  # scripts de video, exposición
```
