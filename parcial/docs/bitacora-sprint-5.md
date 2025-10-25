# Bitacora Sprint 5
**Fecha:** 2025-10-25
**Miembro:** Andre Joaquin Pacheco Taboada

## Problema

Revisando parcial.md me di cuenta que faltan archivos:
- `memory_usage.csv`
- `accuracy_drop.json`
- `bench_latency.csv`
- `plot_memoria.png`

Mi script bench_quant.py solo generaba bench_quant.csv.

## Solución

### 1. Modificar bench_quant.py
Agregué código para generar los 3 CSVs y JSON automáticamente al final del script (líneas 114-145). Ahora cuando corre genera todo.

### 2. Crear plot_memoria.py
Script nuevo que lee memory_usage.csv y hace gráfico de barras. Copié la estructura de plot_quant.py y adapté.

### 3. Actualizar Makefile
Agregué target `plot-memoria` y lo puse en dependencias de `pack`.

## Comandos ejecutados

```bash
make verify-corpus  # ✅ OK
make pack           # ✅ generó dist/ + HASHES.md
make verify         # ✅ OK
```

## Problema 2: test-idem fallaba

Ejecuté `make test-idem` y falló porque el training daba resultados diferentes cada vez:
- Primera corrida: perplexity 1065.99
- Segunda corrida: perplexity 1017.62

Googleé y encontré que PyTorch necesita configuración extra para ser determinista.

### Solución determinismo

Agregué a train.py, eval.py, bench.py, bench_quant.py:
```python
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
```

Probé 2 veces y ahora losses son idénticos:
```
Corrida 1: Step 0: 7.1569, Step 10: 7.0837
Corrida 2: Step 0: 7.1569, Step 10: 7.0837
```

Perplexity: 1019.74 (ambas veces). Hash de metrics.json idéntico.

**Nota:** bench.csv sigue variando un poco (6.36ms vs 6.48ms) porque mide tiempo real de CPU. Eso es normal.

## Pendiente

- Grabar video 5min
- Subir y poner link en video.md

**Fin:** 2025-10-25
