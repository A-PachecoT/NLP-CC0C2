# Proyecto Parcial CC0C2: Mini-Transformer + Cuantizacion

## Proyectos implementados

- **Proyecto 3 (obligatorio):** Mini-Transformer decoder-only con 1 bloque
- **Proyecto 6:** Escalado de inferencia y cuantizacion (Int8/Int4) con KV-cache

## Dependencias

**Permitidas (preinstaladas):**
- Python stdlib
- numpy
- torch (opcional, fallback a NumPy si no disponible)

**Verificacion:**
```bash
make deps
```

## Uso rapido

```bash
# 1. Verificar dependencias
make deps

# 2. Generar corpus sintetico reproducible
make data

# 3. Verificar hash del corpus
make verify-corpus

# 4. Tokenizar
make tokenize

# 5. Entrenar Mini-Transformer
make train

# 6. Evaluar metricas
make eval

# 7. Benchmarks (3 repeticiones con sigma)
make bench

# 8. Generar graficos
make plot

# 9. Ejecutar tests
make test

# 10. Empaquetar artefactos reproducibles
make pack

# 11. Verificar hash del paquete
make verify

# 12. Test de idempotencia
make test-idem
```

## Contrato de variables

| Variable | Descripcion | Efecto | Valores por defecto |
|----------|-------------|--------|---------------------|
| `SEED` | Semilla RNG decimal | Reproducibilidad del corpus y entrenamiento | `42` |
| `SALT` | Salt hexadecimal (>=16 chars) | Unicidad del corpus (hash SHA-256) | `1a2b3c4d5e6f7890abcdef1234567890` |
| `CONTEXT` | Longitud de contexto | Aumenta memoria/latencia; permite secuencias mas largas | `512` |
| `LR` | Learning rate | Alto=rapido/inestable; bajo=lento/estable | `0.001` |
| `HEADS` | Numero de attention heads | +capacidad y +costo O(n^2) | `4` |
| `DIM` | Dimension del modelo | +capacidad, +parametros, +costo | `128` |
| `SEED_BENCH` | Semilla para benchmarks | Reproducibilidad de mediciones | `42` |
| `SOURCE_DATE_EPOCH` | Timestamp para empaquetado | Builds deterministas (tar.gz) | `1700000000` |

**Variables de decodificacion (a implementar):**
- `TOPK`: Top-K sampling (diversidad controlada)
- `TOPP`: Nucleus sampling (probabilidad acumulada)
- `TEMP`: Temperatura (suavizado de distribucion)
- `BEAM`: Tamano de beam search
- `LENGTH_PENALTY`: Penalizacion por longitud

**Variables de cuantizacion (Proyecto 6):**
- `QUANT_BITS`: Bits de cuantizacion (32=FP32, 8=Int8, 4=Int4)

## Ejemplo de uso con variables personalizadas

```bash
# Entrenar con contexto mas largo y learning rate menor
make train CONTEXT=1024 LR=0.0005 HEADS=8 DIM=256

# Benchmark con contexto especifico
make bench CONTEXT=256 SEED_BENCH=123
```

## Verificacion de reproducibilidad

```bash
# 1. Verificar que el corpus es reproducible
make verify-corpus

# 2. Verificar que make pack genera el mismo hash
make distclean
make pack
sha256sum dist/proy-v1.0.0.tar.gz  # comparar con out/HASHES.md

# 3. Verificar idempotencia (2 ejecuciones -> mismos hashes)
make test-idem
```

## Estructura del proyecto

```
parcial/
├── src/           # Codigo fuente (tokenizer, train, eval, bench, plot)
├── tools/         # Scripts de generacion (gen_corpus.sh, capture_env.py)
├── tests/         # Pruebas unitarias (pytest, cobertura >=70%)
├── docs/          # Documentacion (bitacoras, reporte, autoria, cobertura)
├── out/           # Artefactos generados (corpus, metricas, graficos, logs)
├── dist/          # Paquetes finales (proy-v1.0.0.tar.gz)
├── Makefile       # Pipeline completo de reproducibilidad
└── README.md      # Este archivo
```

## Documentacion adicional

- **Bitacoras de sprints:** `docs/bitacora-sprint-{1..N}.md`
- **Reporte final:** `docs/reporte.md`
- **Cobertura de tests:** `docs/cobertura.md`
- **Autoria:** `docs/autoria.md`
- **Video:** `docs/video.md`

## Entregables finales

- `dist/proy-v1.0.0.tar.gz` (empaquetado determinista)
- `out/HASHES.md` (SHA-256 del paquete)
- Video de 5 min mostrando `git log`, `make verify`, experimento clave
- Exposicion oral de 7 min + 5 min de preguntas

## Gates de evaluacion

- Repositorio trazable
- Corpus unico (hash verificado con `make verify-corpus`)
- `make verify` pasa limpio
- Pruebas + cobertura >=70%
- Video subido

**Si algun gate falla -> no se evalua el proyecto**
