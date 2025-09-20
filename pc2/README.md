# Proyecto 2: Clasificación con Transformers preentrenados

## Instalación
```bash
uv add pandas numpy matplotlib torch psutil transformers scikit-learn
```

## Generar dataset
```bash
make data           # Genera 5000 reseñas
make verify-repro   # Verifica reproducibilidad
```

## Ejecución
1. Abrir `notebook.ipynb`
2. Ejecutar todas las celdas
3. Resultados en `out/`

## Diseño
- Clasificación binaria: Positivo vs No Positivo
- DistilBERT multilingual fine-tuned
- Baseline: BoW + Logistic Regression L2
- Validación cruzada 5-fold
- Curvas ROC y PR