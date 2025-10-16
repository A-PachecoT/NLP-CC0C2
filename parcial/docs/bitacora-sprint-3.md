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

## Cuantizacion (pendiente)

**Proximos pasos:**
- Implementar `src/quant.py` con Int8/Int4
- Benchmark FP32 vs Int8 vs Int4
- Medir accuracy drop (perplexity degradation)
- Comparar memoria y latencia

**Fin:** [sprint 3 en progreso - KV-cache completado]
