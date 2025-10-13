# Bitácora Sprint 1: Setup y configuración inicial
**Inicio:** 2025-10-13
**Miembro:** Andre Joaquin Pacheco Taboada

## Comandos

### Setup inicial
```bash
# Estructura creada
mkdir -p parcial/{src,tools,tests,docs,out,dist}
```

### Configuración Makefile
**Problema:** Heredoc Python en Makefile causaba error de sintaxis (línea 78: missing separator)
**Causa:** Líneas dentro de heredoc `<<'PY'` tenían tabs, Makefile las interpretaba como comandos
**Solución:** Extraer script Python a `tools/capture_env.py` separado

```python
# tools/capture_env.py - Captura entorno de forma limpia
# Evita problemas de indentación en Makefile
```

**Resultado:** Makefile simplificado, más mantenible

### Generación de corpus
```bash
$ make data
Generando corpus sintético
# Corpus generado: out/corpus.txt (50,000 palabras)
# Hash guardado: out/corpus_sha256.txt

$ make verify-corpus
Verificando hash del corpus
# ✅ Hash verificado correctamente
```

**Hash SHA-256:** Ver `out/corpus_sha256.txt`

## Estado actual
- ✅ Estructura de carpetas creada
- ✅ Makefile funcional (workaround: capture_env.py separado)
- ✅ gen_corpus.sh implementado con SEED+SALT
- ✅ .gitattributes configurado (LF normalizado)
- ✅ Corpus generado y verificado reproducible

**Próximos pasos:** Implementar tokenizer básico, luego Mini-Transformer

**Fin:** 2025-10-13 [sprint 1 completado - setup]
