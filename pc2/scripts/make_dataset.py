#!/usr/bin/env python3
"""Generador de corpus de reseñas de productos"""
import argparse
import random
import pandas as pd

def generar_corpus(seed, n_samples, balance):
    """Genera reseñas sintéticas de productos"""
    random.seed(seed)

    # Vocabulario
    productos = ["celular", "laptop", "tablet", "auriculares", "smartwatch", "cámara", "teclado", "mouse"]

    adj_positivos = ["excelente", "increíble", "rápido", "confiable", "duradero", "elegante", "potente", "eficiente"]
    adj_negativos = ["defectuoso", "lento", "frágil", "caro", "inútil", "horrible", "malo", "pésimo"]
    adj_neutrales = ["normal", "estándar", "promedio", "básico", "funcional", "aceptable", "común", "regular"]

    # Plantillas variadas
    plantillas = [
        "El {producto} es {adjetivo}",
        "Mi {producto} resultó {adjetivo}",
        "Este {producto} me parece {adjetivo}",
        "El {producto} funciona de manera {adjetivo}",
        "Compré un {producto} {adjetivo}",
    ]

    data = []
    categorias = ["Positivo", "Negativo", "Neutral"]

    if balance:
        # Distribución balanceada
        samples_por_cat = n_samples // 3
        resto = n_samples % 3

        for cat in categorias:
            n = samples_por_cat + (1 if resto > 0 else 0)
            resto -= 1

            for _ in range(n):
                prod = random.choice(productos)
                plantilla = random.choice(plantillas)

                if cat == "Positivo":
                    adj = random.choice(adj_positivos)
                elif cat == "Negativo":
                    adj = random.choice(adj_negativos)
                else:
                    adj = random.choice(adj_neutrales)

                texto = plantilla.format(producto=prod, adjetivo=adj)
                data.append([texto, cat])
    else:
        # Distribución aleatoria
        for _ in range(n_samples):
            prod = random.choice(productos)
            plantilla = random.choice(plantillas)
            cat = random.choice(categorias)

            if cat == "Positivo":
                adj = random.choice(adj_positivos)
            elif cat == "Negativo":
                adj = random.choice(adj_negativos)
            else:
                adj = random.choice(adj_neutrales)

            texto = plantilla.format(producto=prod, adjetivo=adj)
            data.append([texto, cat])

    # Mezclar manteniendo seed
    random.shuffle(data)
    return data

def main():
    parser = argparse.ArgumentParser(description="Generar corpus de reseñas de productos")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument("--n-samples", type=int, default=5000, help="Número de muestras")
    parser.add_argument("--out", type=str, default="data/nlp_prueba_cc0c2.csv", help="Archivo de salida")
    parser.add_argument("--balance", action="store_true", help="Balancear categorías")

    args = parser.parse_args()

    # Generar datos
    data = generar_corpus(args.seed, args.n_samples, args.balance)

    # Guardar CSV
    df = pd.DataFrame(data, columns=["Texto", "Categoría"])
    df.to_csv(args.out, index=False)

    print(f"Generadas {len(data)} reseñas en {args.out}")
    print(f"Distribución: {df['Categoría'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()