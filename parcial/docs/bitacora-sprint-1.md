# Bitácora Sprint 1: Setup y configuración inicial
**Inicio:** 2025-10-13
**Miembro:** Andre Joaquin Pacheco Taboada

## Comandos ejecutados

### Setup inicial
```bash
mkdir -p parcial/{src,tools,tests,docs,out,dist}
```

### Configuración Makefile
**Problema:** Heredoc Python causaba error "missing separator" (línea 78)
**Solución:** Extraer a `tools/capture_env.py` separado

### Pipeline completo
```bash
# 1. Verificar dependencias
$ make deps
Verificando dependencias preinstaladas (stdlib, numpy, torch opcional)
numpy OK
torch OK

# 2. Generar corpus sintetico reproducible
$ make data
Generando corpus sintético
./tools/gen_corpus.sh 42 1a2b3c4d5e6f7890abcdef1234567890 > out/corpus.txt
echo "Comando: ./tools/gen_corpus.sh 42 1a2b3c4d5e6f7890abcdef1234567890" > out/seed.txt
sha256sum out/corpus.txt | awk '{print $1}' > out/corpus_sha256.txt

# 3. Verificar hash del corpus
$ make verify-corpus
Verificando hash del corpus
HGEN="$(./tools/gen_corpus.sh 42 1a2b3c4d5e6f7890abcdef1234567890 | sha256sum | awk '{print $1}')"; \
        HSAVED="$(cat out/corpus_sha256.txt)"; test "$HGEN" = "$HSAVED"

# 4. Tokenizar
$ make tokenize
Generando corpus sintético
./tools/gen_corpus.sh 42 1a2b3c4d5e6f7890abcdef1234567890 > out/corpus.txt
echo "Comando: ./tools/gen_corpus.sh 42 1a2b3c4d5e6f7890abcdef1234567890" > out/seed.txt
sha256sum out/corpus.txt | awk '{print $1}' > out/corpus_sha256.txt
Tokenizando corpus
python src/tokenizer.py out/corpus.txt --output out/tokens.jsonl --vocab out/vocab.txt
Vocab size: 1004
Total tokens: 50000
Unique tokens: 1000
```

### Tokenizer básico
**Problema 1:** Alias `cd="z"` (zoxide) rompía comandos Bash no-interactivos
**Solución:** Modificar `~/.zshrc` con `[[ -o interactive ]] && alias cd="z"`

**Problema 2:** Makefile usa `python` pero sistema tiene `python3`
**Solución:** Variable `PYTHON ?= python3` en Makefile, usar `$(PYTHON)` en todos los targets

```bash
$ make tokenize
Vocab size: 1004
Total tokens: 50000
Unique tokens: 1000
```

## Estado actual
- ✅ Setup completo (estructura, Makefile, gitattributes)
- ✅ Corpus reproducible (SEED+SALT, hash verificado)
- ✅ README.md con tabla variable→efecto
- ✅ Tokenizer básico funcionando (vocab + jsonl)
- ✅ Fix dotfiles zoxide para shells no-interactivos
- ✅ Fix Makefile python→python3

**Próximos pasos:** Mini-Transformer (attention, RoPE, training loop)

**Fin:** 2025-10-13 [sprint 1 completado - setup + tokenizer]
