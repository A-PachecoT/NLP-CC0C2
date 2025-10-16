# Bitacora Sprint 3: KV-cache + Cuantizacion
**Inicio:** 2025-10-15
**Miembro:** Andre Joaquin Pacheco Taboada

## Implementacion

### KV-cache real

**Modulos creados:**
```bash
src/generate.py   # Generacion autoregresiva con/sin cache
```

**Modificaciones:**
- `bench.py` - Benchmark REAL usando generacion autoregresiva (no simulacion)
- `Makefile` - bench depende de train, pasa --model

**Funcionamiento KV-cache:**
1. **Con cache:** Solo procesa ultimo token nuevo, reutiliza K/V anteriores
2. **Sin cache:** Reprocesa toda la secuencia en cada paso

### Pipeline ejecutado

```bash
$ source ../.venv/bin/activate

$ make bench
Using device: cpu
Vocab: 1004, Tokens: 50000
Creando modelo: dim=128, heads=4, pos=rope
Entrenando 100 pasos, lr=0.001, warmup=10
Step 0/100, Loss: 7.09, LR: 0.000100
Step 90/100, Loss: 6.83, LR: 0.001000
Modelo guardado en dist/model.tar.gz

Benchmark: length=50, reps=3, warmup=1
Con cache:  8.98ms ± 0.43ms
Sin cache:  17.47ms ± 0.52ms
Speedup: 1.95x

$ make plot
Grafico guardado en out/plot_latencia.png
```

## Resultados KV-cache

### Generacion autoregresiva (50 tokens)
- **Con KV-cache:** 8.98ms ± 0.43ms
- **Sin KV-cache:** 17.47ms ± 0.52ms
- **Speedup:** 1.95x

## Cuantizacion Int8/Int4

### Implementacion

**Modulos creados:**
```bash
src/quant.py         # Cuantizacion simetrica Int8/Int4
src/bench_quant.py   # Benchmark FP32 vs Int8 vs Int4
src/plot_quant.py    # Graficos de compresion/accuracy/latencia
```

**Tecnicas aplicadas:**
- **Cuantizacion simetrica** con scale factor (no zero-point)
- **Int8:** rango [-128, 127], scale = abs_max / 127.0
- **Int4:** rango [-8, 7], scale = abs_max / 7.0 (guardado como int8)
- **QuantizedLinear:** Descuantiza pesos en forward pass

### Pipeline ejecutado

```bash
$ make bench-quant
Cargando modelo base FP32...
[1/3] Benchmark FP32
  Calculando perplexity...
  Midiendo latencia...
  Size: 1.99 MB
  Perplexity: 1052.43
  Latency: 9.23ms ± 0.51ms

[2/3] Benchmark Int8
  Size: 1.06 MB (1.88x)
  Perplexity: 1052.49 (delta: +0.06)
  Latency: 22.68ms ± 0.59ms

[3/3] Benchmark Int4
  Size: 1.06 MB (1.88x)
  Perplexity: 1054.75 (delta: +2.32)
  Latency: 22.74ms ± 0.64ms

$ make plot-quant
Grafico guardado en out/plot_quant.png
```

## Resultados Cuantizacion

| Config | Tamaño (MB) | Compresion | Perplexity | Accuracy Drop | Latencia (ms) |
|--------|-------------|------------|------------|---------------|---------------|
| FP32   | 1.99        | 1.00x      | 1052.43    | baseline      | 9.23 ± 0.51   |
| Int8   | 1.06        | 1.88x      | 1052.49    | +0.06         | 22.68 ± 0.59  |
| Int4   | 1.06        | 1.88x      | 1054.75    | +2.32         | 22.74 ± 0.64  |

### Analisis

**Trade-offs observados:**
1. **Compresion:** Int8/Int4 reducen tamaño ~1.88x (esperado ~4x teorico, overhead de scales/buffers)
2. **Accuracy:** Degradacion minima en Int8 (+0.06 ppl), aceptable en Int4 (+2.32 ppl)
3. **Latencia:** **Incremento** 2.5x por overhead de descuantizacion en forward pass (Python/CPU)

**Nota:** En produccion con kernels optimizados (CUDA, quantized GEMM), Int8/Int4 suelen ser mas rapidos que FP32.

**Fin:** 2025-10-16
