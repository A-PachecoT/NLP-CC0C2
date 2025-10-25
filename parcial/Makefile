# Makefile para el proyecto CC0c2
# Uso: make [target]

.PHONY: deps build data tokenize train eval bench plot pack verify verify-corpus tag test test-idem clean distclean

# Python command
PYTHON ?= python3

# Reproducibilidad
SOURCE_DATE_EPOCH ?= 1700000000
SEED ?= 42
SALT ?= 1a2b3c4d5e6f7890abcdef1234567890
SEED_BENCH ?= 42

# Hiperparámetros por defecto
CONTEXT ?= 512
LR ?= 0.001
HEADS ?= 4
DIM ?= 128

deps:
	@echo "Verificando dependencias preinstaladas (stdlib, numpy, torch opcional)"
	@$(PYTHON) -c "import numpy; print('numpy OK')" || echo "ERROR: numpy no encontrado"
	@$(PYTHON) -c "import torch; print('torch OK')" 2>/dev/null || echo "torch no disponible (opcional)"

build:
	@echo "Chequeos básicos"
	# shellcheck tools/*.sh || true
	# ruff check src/*.py || true
	mkdir -p out dist

data:
	@echo "Generando corpus sintético"
	./tools/gen_corpus.sh $(SEED) $(SALT) > out/corpus.txt
	echo "Comando: ./tools/gen_corpus.sh $(SEED) $(SALT)" > out/seed.txt
	sha256sum out/corpus.txt | awk '{print $$1}' > out/corpus_sha256.txt

verify-corpus:
	@echo "Verificando hash del corpus"
	HGEN="$$(./tools/gen_corpus.sh $(SEED) $(SALT) | sha256sum | awk '{print $$1}')"; \
	HSAVED="$$(cat out/corpus_sha256.txt)"; test "$$HGEN" = "$$HSAVED"

tokenize: data
	@echo "Tokenizando corpus"
	$(PYTHON) src/tokenizer.py out/corpus.txt --output out/tokens.jsonl --vocab out/vocab.txt

train: tokenize
	@echo "Entrenando modelo"
	$(PYTHON) src/train.py --lr $(LR) --heads $(HEADS) --dim $(DIM) --input out/tokens.jsonl --output dist/model.tar.gz

eval: train
	@echo "Evaluando métricas"
	$(PYTHON) src/eval.py dist/model.tar.gz --output out/metrics.json

bench: train
	@echo "Benchmarking KV-cache (3 repeticiones)"
	$(PYTHON) src/bench.py --model dist/model.tar.gz --n 50 --seed $(SEED_BENCH) --warmup 1 --reps 3 --output out/bench.csv

plot: bench
	@echo "Generando gráficos"
	$(PYTHON) src/plot.py out/bench.csv --output out/plot_latencia.png

bench-quant: train
	@echo "Benchmarking cuantizacion FP32/Int8/Int4"
	$(PYTHON) src/bench_quant.py --model dist/model.tar.gz --length 50 --reps 3 --output out/bench_quant.csv

plot-quant: bench-quant
	@echo "Generando graficos de cuantizacion"
	$(PYTHON) src/plot_quant.py out/bench_quant.csv --output out/plot_quant.png

plot-memoria: bench-quant
	@echo "Generando grafico de memoria"
	$(PYTHON) src/plot_memoria.py out/memory_usage.csv --output out/plot_memoria.png

test:
	@echo "Ejecutando tests"
	pytest tests/ --cov=src --cov-report=term-missing || bats tests/ || true

test-idem:
	@echo "Verificando idempotencia"
	rm -rf out/tmp && mkdir -p out/tmp
	$(MAKE) test eval bench plot
	rsync -a --delete out/ out/tmp/
	$(MAKE) test eval bench plot
	{ find out -type f ! -path 'out/tmp/*' ! -name 'hashes.txt' -exec sha256sum {} \; | sort > out/hashes.txt; }
	find out/tmp -type f -exec sha256sum {} \; | sort > out/tmp/hashes.txt
	diff -u out/tmp/hashes.txt out/hashes.txt

pack: eval bench plot bench-quant plot-quant plot-memoria
	@echo "Capturando entorno"
	$(PYTHON) tools/capture_env.py > out/env.txt
	@echo "Empaquetando artefactos reproducibles"
	find out -type f -print0 | xargs -0 touch -t 202311142213.20
	rm -f dist/proy-v1.0.0.tar.gz
	tar -czf dist/proy-v1.0.0.tar.gz out/
	sha256sum dist/proy-v1.0.0.tar.gz | awk '{print $$1"  "$$2}' > out/HASHES.md

verify:
	@echo "Verificando hash del paquete"
	sha256sum -c out/HASHES.md

tag:
	@echo "Creando tag simulado"
	echo "v1.0.0: Versión inicial" > CHANGELOG.md
	echo "Firma simulada: $(shell date -u +%FT%TZ)" > out/tag_signature.txt

clean:
	rm -rf out/tmp out/hashes.txt

distclean: clean
	rm -rf out/* dist/* CHANGELOG.md