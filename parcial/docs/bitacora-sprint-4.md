# Bitacora Sprint 4: Tests y Docs
**Fecha:** 2025-10-16 al 18
**Miembro:** Andre Joaquin Pacheco Taboada

## Tests

Creé tests para todos los módulos (42 tests total):
- test_tokenizer.py, test_attention.py, test_transformer.py
- test_train.py, test_eval.py, test_generate.py
- test_quant.py, test_bench.py
- conftest.py con fixtures compartidos

Instalé pytest y coverage:
```bash
uv pip install pytest pytest-cov
```

Resultado: 42 passed, 97% coverage total
- 6 módulos con 100%: attention, transformer, tokenizer, eval, generate, bench
- train.py: 98% (solo falta __main__)
- quant.py: 87%

Configuré .coveragerc para excluir scripts de plotting.

## Documentación

Escribí:
- reporte.md: todo el proyecto
- cobertura.md: justificación tests
- autoría.md: declaración académica
- video.md: guía para grabar

## Pendiente

- Ajustes según parcial.md
- Grabar video

**Fin:** 2025-10-18
