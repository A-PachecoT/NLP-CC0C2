# Bitacora Sprint 2: Mini-Transformer
**Inicio:** 2025-10-14
**Miembro:** Andre Joaquin Pacheco Taboada

## Implementacion

### Modulos creados
```bash
src/attention.py      # Multi-head attention con causal masking
src/transformer.py    # Decoder-only 1 bloque, RoPE + sinusoidal
src/train.py         # Training loop con warmup y gradient clipping
src/eval.py          # Perplexity
src/bench.py         # Benchmarks latencia con sigma
src/plot.py          # Graficos matplotlib
```

### Arquitectura Mini-Transformer
- **1 bloque decoder-only**
- **Attention:** Multi-head (4 heads), causal masking
- **Positional encoding:** RoPE (default) o sinusoidal (configurable)
- **FFN:** 2 capas con ReLU
- **Layer Norm:** Pre-norm architecture
- **Parametros:** dim=128, heads=4, vocab=1004

### Pipeline ejecutado

```bash
$ source ../.venv/bin/activate  # Activar venv con torch

$ make train
Using device: cpu
Vocab: 1004, Tokens: 50000
Creando modelo: dim=128, heads=4, pos=rope
Entrenando 100 pasos, lr=0.001, warmup=10
Step 0/100, Loss: 7.21, LR: 0.000100
Step 10/100, Loss: 7.12, LR: 0.001000
Step 90/100, Loss: 6.96, LR: 0.001000
Modelo guardado en dist/model.tar.gz

$ make eval
Evaluando en 10000 tokens
Perplexity: 1045.31
Loss: 6.95
Metricas guardadas en out/metrics.json

$ make bench
Benchmark: n=512, reps=3, warmup=1 (simulacion numpy)
Sin cache: 0.05ms ± 0.00ms
Con cache: 0.02ms ± 0.00ms

$ make plot
Grafico guardado en out/plot_latencia.png
```

## Resultados

### Entrenamiento
- **Loss inicial:** 7.21
- **Loss final:** 6.96
- **Mejora:** 0.25 (100 steps)
- **Warmup:** 10 steps (LR: 0.0001 → 0.001)
- **Device:** CPU

### Evaluacion
- **Perplexity:** 1045.31
- **Loss:** 6.95
- **Test tokens:** 10000 (ultimos 20% del corpus)

### Benchmarks (simulacion)
- **Sin KV-cache:** 0.05ms ± 0.00ms (multiplicacion matrices numpy)
- **Con KV-cache:** 0.02ms ± 0.00ms (50% simulado)
- **Speedup:** 2x
- **Nota:** Benchmarks simulados, no usan generacion autoregresiva real

## Estado actual
- ✅ Mini-Transformer funcionando (attention + FFN + LayerNorm)
- ✅ RoPE y sinusoidal implementados
- ✅ Training con warmup y gradient clipping
- ✅ Eval con perplexity
- ✅ Bench con repeticiones y sigma
- ✅ Pipeline completo: tokenize → train → eval → bench → plot

**Proximos pasos:** Generacion autoregresiva + KV-cache real + cuantizacion Int8/Int4

**Fin:** 2025-10-15 [sprint 2 completado - mini-transformer]
