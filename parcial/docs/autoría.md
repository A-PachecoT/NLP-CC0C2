# Declaracion de Autoria

**Autor:** Andre Joaquin Pacheco Taboada
**Curso:** CC0C2A Procesamiento del Lenguaje Natural
**Fecha:** 2025-10-25

## Declaracion

Declaro que este proyecto es mi trabajo individual para el parcial del curso.

Elegí implementar Mini-Transformer (Proyecto 3) más KV-cache y Cuantizacion (Proyecto 6). Decidí usar PyTorch porque ya lo conocía, RoPE para positional encoding porque leí que funciona mejor que sinusoidal, y cuantización simétrica.

Hiperparámetros: dim=128, heads=4, lr=0.001 con warmup=10, SEED=42.

Ejecuté todo el pipeline (data, tokenize, train, eval, bench, test) varias veces y validé los resultados manualmente.

## Librerías y Referencias

**Librerías usadas:**
- Python stdlib (json, argparse, pathlib, tempfile, time)
- NumPy - operaciones numéricas
- PyTorch - nn.Module, autograd, optimizers
- Matplotlib - gráficos
- Pytest + Coverage - tests

**Papers consultados:**
- "Attention is All You Need" (Vaswani et al., 2017) - arquitectura transformer
- "RoFormer" (Su et al., 2021) - RoPE positional encoding
- "LLM.int8()" (Dettmers et al., 2022) - técnicas de cuantización

## Compromiso Académico

Confirmo que:
- Ejecuté y validé todo el pipeline personalmente
- Entiendo cómo funciona cada módulo implementado
- No copié código de otros estudiantes ni repos externos
- Este es mi trabajo individual

Puedo explicar: cómo funciona multi-head attention, por qué KV-cache reduce de O(n²) a O(n), trade-offs de Int8 vs Int4.

## Reproducibilidad

El proyecto es reproducible con `make data && make tokenize && make train && make eval && make bench && make test && make pack && make verify`.

**Fecha:** 2025-10-25
