# Video Demostracion (5 minutos)

**Autor:** Andre Joaquin Pacheco Taboada
**Proyecto:** Mini-Transformer con KV-cache y Cuantizacion
**Duracion:** ~5 minutos

---

## Link al Video

**URL:** [PENDIENTE - Grabar y subir a YouTube/Drive]

**Alternativas de hosting:**
- YouTube (unlisted)
- Google Drive (compartido con permisos)
- Vimeo

---

## Contenido del Video

### 1. Introduccion (30 segundos)
- Presentacion personal
- Proyecto: Mini-Transformer + KV-cache + Cuantizacion
- Objetivos: reproducibilidad, benchmarks, tests

### 2. Git Log y Commits (1 minuto)
```bash
git log --oneline --graph --all
```

**Mostrar:**
- Commits cronologicos (corpus ’ tokenizer ’ transformer ’ kv-cache ’ cuantizacion)
- Mensajes descriptivos
- Estructura de branches (si aplica)

### 3. Verificacion de Reproducibilidad (1.5 minutos)

**Opcion A - Verificar hash del corpus:**
```bash
make verify-corpus
# Muestra: hash generado = hash guardado
```

**Opcion B - Test de idempotencia:**
```bash
make test-idem
# Ejecuta pipeline 2 veces, compara hashes
```

**Opcion C - Verificar paquete:**
```bash
make pack
make verify
# Verifica SHA-256 de dist/proy-v1.0.0.tar.gz
```

### 4. Experimento Clave: KV-cache o Cuantizacion (2 minutos)

**Opcion A - KV-cache speedup:**
```bash
make bench
cat out/bench.csv
```

**Explicar:**
- Sin cache: 17.47ms ± 0.52ms
- Con cache: 8.98ms ± 0.43ms
- **Speedup: 1.95x**
- Por que: O(n²) ’ O(n), reutiliza keys/values

**Mostrar grafico:**
```bash
open out/plot_latencia.png
```

**Opcion B - Cuantizacion trade-offs:**
```bash
make bench-quant
cat out/bench_quant.csv
```

**Explicar:**
- FP32: 1.99 MB, 1045.31 ppl
- Int8: 1.06 MB (1.88x), +0.06 ppl (casi gratis)
- Int4: 1.06 MB (1.88x), +2.32 ppl (aceptable)
- Trade-off: compresion vs accuracy

**Mostrar grafico:**
```bash
open out/plot_quant.png
```

### 5. Tests y Cobertura (30 segundos)
```bash
make test
# Mostrar: 42 passed, 97% coverage
```

**Mencionar:**
- 6 modulos con 100% cobertura
- Tests de unidad, integracion, regresion

### 6. Cierre (30 segundos)
- Resumen: modelo funcional, reproducible, bien testeado
- Resultados clave (speedup, compresion, cobertura)
- Gracias

---

## Checklist Pre-Grabacion

- [ ] Repo limpio (`git status` sin cambios uncommitted)
- [ ] Entorno activado (`source ../.venv/bin/activate`)
- [ ] Pipeline ejecutado al menos una vez
- [ ] Graficos generados (plot_latencia.png, plot_quant.png)
- [ ] Terminal con fuente legible (tamaño e14pt)
- [ ] Grabacion de pantalla + audio (explicacion verbal)

---

## Consejos Tecnicos

**Herramientas de grabacion:**
- macOS: QuickTime Player (Cmd+Shift+5)
- Windows: Xbox Game Bar (Win+G)
- Linux: SimpleScreenRecorder, OBS Studio
- Multiplataforma: OBS Studio (gratuito)

**Calidad:**
- Resolucion: 1920x1080 (Full HD)
- FPS: 30 fps minimo
- Audio: Microfono claro, sin ruido de fondo
- Duracion: ~5 minutos (max 6 minutos)

**Edicion (opcional):**
- Cortar pausas largas
- Acelerar comandos lentos (make train)
- Agregar titulos/captions si ayuda

---

## Script Sugerido

```
[00:00] Hola, soy Andre Pacheco. Les presento mi proyecto de NLP:
        Mini-Transformer con KV-cache y cuantizacion.

[00:30] Primero, veamos el historial de commits:
        [Ejecuta: git log --oneline]
        Como pueden ver, el proyecto tiene commits por cada feature...

[01:30] Ahora verificamos reproducibilidad:
        [Ejecuta: make verify-corpus]
        El hash del corpus coincide, demostrando determinismo...

[02:00] El experimento clave es KV-cache:
        [Ejecuta: make bench]
        Sin cache: 17ms, con cache: 9ms. Speedup de 1.95x.
        Esto es porque reducimos complejidad de O(n²) a O(n)...
        [Muestra: out/plot_latencia.png]

[04:00] Tambien implementamos cuantizacion:
        [Ejecuta: make bench-quant]
        Int8 reduce tamaño 1.88x con solo +0.06 perplexity...
        [Muestra: out/plot_quant.png]

[04:30] Finalmente, tenemos 97% cobertura de tests:
        [Ejecuta: make test]
        42 tests, todos pasando...

[05:00] En resumen: modelo funcional, reproducible, y optimizado.
        Gracias por su atencion.
```

---

**PENDIENTE:** Grabar video y actualizar link arriba.

**Fecha limite:** [Consultar syllabus del curso]

---

**Fin del documento**
