## **Práctica calificada 2-CC0C2**

#### **Instrucciones generales**
- **Plazo**: 10 días.
- **Tiempo estimado por proyecto**: ~6 horas (3 h implementación, 1.5 h teoría, 1 h video, 0.5 h exposición).
- **Formato**: Trabajo **individual**.
- **Entregables (obligatorios)**:
  - **Repositorio Git**:
    - `README.md`: Instrucciones claras para ejecutar `make data`, dependencias, tiempos de ejecución, y diseño del generador de texto.
    - `scripts/make_dataset.py`: Script para generar un corpus de ~5,000 oraciones en español (dominio: reseñas de productos) con etiquetas {Positivo, Negativo, Neutral}.
    - `data/nlp_prueba_cc0c2.csv`: CSV generado por `make data` con columnas `Texto,Categoría`.
    - `out/`: Gráficos/tablas generadas por cada proyecto.
    - `video.mp4`: Video de 5-8 min en formato sprint.
    - `requirements.txt`: Dependencias (Python 3.x, NumPy, Pandas, Matplotlib, PyTorch, scikit-learn; opcional Hugging Face `transformers`, `sentence-transformers`).
    - `SEEDS_AND_VERSIONS.md`: Semillas usadas, versiones de bibliotecas, y hash SHA-256 del CSV.
    - Mínimo **4 commits** por proyecto con mensajes en español (ej.: "Implementar transformer PC2").
  - **Video (5-8 min, formato sprint)**:
    - 0:00-0:30: Objetivo y estructura.
    - 0:30-3:00: Demo en vivo de `make data`, verificación de hash, y ejecución del Notebook.
    - 3:00-4:30: Resultados, métricas, y gráficos.
    - 4:30-5:00+: Riesgos, limitaciones, y próximos pasos.
  - **Exposición (8 min + 10 min preguntas)**: Usa el Notebook como apoyo. Responde preguntas orales.
- **Restricciones antiplagio**: No uses IA generativa ni copies de internet. Verificación con MOSS/Turnitin y coherencia de commits.
- Corpus generado por tu script con semilla fija.
- **Formato del Notebook**:
  - Objetivo e historias de usuario.
  - Setup reproducible (semillas, versiones).
  - Implementación con celdas comentadas.
  - Métricas y gráficos en `out/`.
  - Preguntas teóricas (1-2 párrafos por pregunta).
  - Trade-offs, riesgos, y conclusiones.

**Dataset**: Genera `data/nlp_prueba_cc0c2.csv` con `make data` (~5,000 oraciones en español, etiquetas {Positivo, Negativo, Neutral}, dominio: reseñas de productos). Ejemplo:
```
Texto,Categoría
La cámara del celular es espectacular,Positivo
El software es confuso y se bloquea,Negativo
El diseño es moderno pero el precio es alto,Neutral
```

#### **Requisito de automatización: Uso de `make data`**
Incluye un **Makefile**  `make data` que:
- Crea `data/` si no existe.
- Ejecuta `scripts/make_dataset.py` con semilla fija y parámetros.
- Genera el hash SHA-256 y lo guarda en `data/nlp_prueba_cc0c2.sha256`.
- Permite verificar reproducibilidad con `make verify-repro`.

#### Código de ejemplo (Makefile)
```make
# Makefile para generación reproducible del corpus
PYTHON  ?= python3
SEED    ?= 42
N       ?= 5000
DATA_DIR := data
RAW      := $(DATA_DIR)/nlp_prueba_cc0c2.csv
HASHFILE := $(DATA_DIR)/nlp_prueba_cc0c2.sha256

ifeq ($(shell command -v sha256sum >/dev/null 2>&1; echo $$?),0)
  SHA256 = sha256sum
else
  SHA256 = shasum -a 256
endif

.PHONY: data hash verify-repro clean-data help

$(DATA_DIR):
	@mkdir -p $(DATA_DIR)

data: $(DATA_DIR)
	$(PYTHON) scripts/make_dataset.py --seed $(SEED) --n-samples $(N) --out $(RAW) --balance
	@$(SHA256) $(RAW) | tee $(HASHFILE)

hash:
	@$(SHA256) $(RAW)

verify-repro:
	@test -f $(RAW) || (echo "No se encontró $(RAW)"; exit 1)
	@test -f $(HASHFILE) || (echo "No se encontró $(HASHFILE)"; exit 1)
	@$(SHA256) --check $(HASHFILE) || (echo "El hash NO coincide"; exit 1)
	@echo "Reproducible: OK"

clean-data:
	@rm -f $(RAW) $(HASHFILE)

help:
	@echo "Targets: data, hash, verify-repro, clean-data"
```

#### Guía para `make_dataset.py`
- **Dominio**: Reseñas de productos (celulares, laptops, etc.).
- **Método**: Usa plantillas (ej.: "El [producto] es [adjetivo]") con listas de palabras polarizadas:
  - Positivas: "excelente", "rápido", "confiable".
  - Negativas: "defectuoso", "lento", "frágil".
  - Neutrales: "estándar", "promedio", "funcional".
- **Ejemplo**:
  ```python
  import argparse
  import random
  import pandas as pd

  parser = argparse.ArgumentParser(description="Generar corpus de reseñas")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--n-samples", type=int, default=5000)
  parser.add_argument("--out", type=str, default="data/nlp_prueba_cc0c2.csv")
  parser.add_argument("--balance", action="store_true")
  args = parser.parse_args()

  random.seed(args.seed)
  productos = ["celular", "laptop", "auriculares"]
  adj_pos = ["excelente", "rápido", "increíble"]
  adj_neg = ["defectuoso", "lento", "caro"]
  adj_neu = ["normal", "estándar", "promedio"]
  data = []
  for _ in range(args.n_samples):
      prod = random.choice(productos)
      cat = random.choice(["Positivo", "Negativo", "Neutral"])
      adj = random.choice(adj_pos if cat == "Positivo" else adj_neg if cat == "Negativo" else adj_neu)
      texto = f"El {prod} es {adj}"
      data.append([texto, cat])
  pd.DataFrame(data, columns=["Texto", "Categoría"]).to_csv(args.out, index=False)
  ```

### **Lista de proyectos**

### **Proyecto 1: N-gramas con Back-off y smoothing avanzado**
- **Temas**: Modelos de lenguaje avanzados.
- **Implementación**: Entrena unigrama, bigrama, y trigrama en 4,000 oraciones. Implementa **Katz back-off** y **Kneser-Ney smoothing**. Maneja `<UNK>` con umbral (frecuencia < 3). Compara tokenización por palabras vs subpalabras (BPE, vocab=2,000). Evalúa en 1,000 oraciones.
- **Teoría**:
  - Diferencias entre Katz y Kneser-Ney.
  - Impacto de `<UNK>` en perplejidad.
  - Subpalabras vs palabras en n-gramas.
  - Limitaciones de n-gramas largos.
- **Métricas**: Perplejidad por técnica, tamaño de vocabulario, tiempo de entrenamiento.
- **Entregables**:
  - Notebook y teoría. Commit: "N-gramas Katz y Kneser-Ney PC2".
  - Video: Demo de entrenamiento y comparación.

### **Proyecto 2: Clasificación con Transformers preentrenados**
- **Temas**: Modelos preentrenados y fine-tuning.
- **Implementación**: Tarea binaria (Positivo vs No Positivo) con un transformer ligero (ej.: `distilbert-base-multilingual-cased`). Fine-tune en 4,000 oraciones con validación cruzada (5 folds). Compara con BoW + logística L2. Traza curvas ROC y PR.
- **Teoría**:
  -  Ventajas de transformers sobre BoW.
  -  Trade-offs de fine-tuning vs feature extraction.
  -  Impacto del desbalance en F1.
  -  Regularización implícita en transformers.
- **Métricas**: F1 macro, ROC-AUC, tiempo de entrenamiento, memoria pico.
- **Entregables**:
  - Notebook y teoría. Commit: "Transformer clasificación PC2".
  - Video: Curvas ROC/PR y resultados.

### **Proyecto 3: Embeddings contextuales y anisotropía**
- **Temas**: Representaciones contextuales.
- **Implementación**: Compara embeddings estáticos (word2vec promedio) vs contextuales (`sentence-transformers`, modelo `paraphrase-multilingual-MiniLM-L12-v2`). Aplica **CSLS** para mitigar anisotropía. Recupera 10 consultas con **Recall@10** en 4,000 oraciones. Visualiza con t-SNE.
- **Teoría**:
  - Anisotropía en embeddings contextuales.
  - CSLS vs coseno.
  - Limitaciones de t-SNE vs PCA.
  - Casos donde word2vec supera a transformers.
- **Métricas**: Recall@10, distribución de cosenos, tiempo de consulta.
- **Entregables**:
  - Notebook y teoría. Commit: "Embeddings contextuales PC2".
  - Video: Visualización t-SNE y resultados.

### **Proyecto 4: LM Neuronal con Transformer**
- **Temas**: Modelos de lenguaje transformer.
- **Implementación**: Implementa un transformer pequeño (2 capas, 4 cabeceras) con **weight tying** y **dropout** (0.2). Entrena en 4,000 oraciones (nivel palabra) y valida en 1,000. Compara con trigrama Kneser-Ney. Aplica **gradient clipping** (norma=1.0).
- **Teoría**:
  -  Beneficios del [weight tying](https://tomroth.dev/weight_tying/).
  -  Transformers vs RNN en modelado de lenguaje.
  -  Exploding gradients y clipping.
  -  Impacto de la profundidad del modelo.
- **Métricas**: Perplejidad, curva de entrenamiento, norma de gradiente, tokens por segundo.
- **Entregables**:
  - Notebook y teoría. Commit: "LM Transformer PC2".
  - Video: Comparación con trigrama.

#### **Proyecto 5: Seq2Seq con atención y scheduled sampling**
- **Temas**: Modelos secuencia a secuencia avanzados.
- **Implementación**: Genera pares sintéticos (oraciones ruidosas -> originales). Implementa un encoder-decoder con **atención multiplicativa** y **scheduled sampling** (probabilidad inicial 0.5, decaimiento lineal). Compara **beam search** (beam=5) vs **teacher forcing**. Evalúa 1,000 pares.
- **Teoría**:
  - Atención multiplicativa vs aditiva.
  - Scheduled sampling y exposición a errores.
  - Trade-offs de beam search.
  - Denoising como preentrenamiento.
- **Métricas**: BLEU, chrF, tiempo de decodificación, tasa de reconstrucción exacta.
- **Entregables**:
  - Notebook y teoría. Commit: "Seq2Seq atención PC2".
  - Video: Mapas de atención y ejemplos.

#### **Proyecto 6: BPTT con longitud adaptativa**
- **Temas**: Retropropagación a través del tiempo.
- **Implementación**: Reusa el LM del Proyecto 4 (Transformer). Compara **BPTT truncado** con longitudes 32, 64, 128, ajustando dinámicamente según la norma de gradiente (umbral=1.0). Traza curvas de pérdida y normas de gradiente. Evalúa en 1,000 oraciones.
- **Teoría**:
  - BPTT truncado vs completo.
  - Impacto de la longitud de contexto.
  - Estrategias para manejar exploding gradients.
  - Adaptación dinámica de BPTT.
- **Métricas**: Perplejidad, norma de gradiente, % de pasos con clip, throughput (tokens/s).
- **Entregables**:
  - Notebook y teoría. Commit: "BPTT adaptativo PC2".
  - Video: Análisis de curvas y elección de longitud.

#### **Proyecto 7: Generación de texto con regularización avanzada**
- **Temas**: Generación de texto y regularización.
- **Implementación**: Entrena un LM transformer (reusa Proyecto 4) para generar reseñas. Aplica **label smoothing** (0.1) y **mixup** (interpolación de embeddings). Compara generación con **top-k** (k=50) y **nucleus sampling** (p=0.9). Evalúa 1,000 muestras generadas con métricas automáticas y 10 ejemplos cualitativos.
- **Teoría**:
  - Impacto del label smoothing.
  - Mixup en NLP.
  - Top-k vs nucleus sampling.
  -  Evaluación automática vs humana.
- **Métricas**: Perplejidad, diversidad (distinct-1, distinct-2), tiempo de generación.
- **Entregables**:
  - Notebook y teoría. Commit: "Generación regularizada PC2".
  - Video: Ejemplos generados y métricas.


#### **Rúbrica de evaluación (20 puntos por proyecto)**

| **Criterio**               | **Puntos** | **Detalles**                                                                 |
| -------------------------- | ---------- | ---------------------------------------------------------------------------- |
| **Trabajo (Notebook)**     | **6**      | Código funcional (3), reproducibilidad (2), teoría escrita (1).              |
| **Video de ejecución**     | **4**      | Guion sprint (1), demo en vivo (1.5), métricas (1), cierre (0.5).            |
| **Exposición y preguntas** | **10**     | Estructura (2), profundidad técnica (3), respuestas orales (4), gráficos (1). |

**Condiciones adicionales**:
- `make_dataset.py` debe ser determinista con `--seed` y aceptar `--help`.
- `make data` debe ser reproducible (verificado con `make verify-repro`).
- Registra el hash SHA-256 en `SEEDS_AND_VERSIONS.md`.


#### **Cronograma recomendado (10 días)**
- **Día 1**: Configura `make_dataset.py` y `make data`. Prueba `verify-repro`. Commit: "Inicializar PC2".
- **Días 2-8**: Un proyecto por día (~6 horas cada uno). Commits: "Proyecto X PC2".
- **Día 9**: Genera gráficos/tablas en `out/`, graba video. Commit: "Video y visualizaciones PC2".
- **Día 10**: Finaliza Notebook, prepara exposición, ensaya preguntas. Commit: "Entregables finales PC2".
