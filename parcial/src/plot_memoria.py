#!/usr/bin/env python3
"""Grafico de uso de memoria (FP32 vs Int8 vs Int4)"""
import argparse
import csv
import matplotlib.pyplot as plt

def plot_memory(input_path, output_path):
    """Lee memory_usage.csv y genera grafico de barras"""
    configs = []
    memory_mb = []

    with open(input_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            configs.append(row['config'])
            memory_mb.append(float(row['memory_mb']))

    # Grafico de barras
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(configs, memory_mb, color=['#3498db', '#2ecc71', '#e74c3c'])

    # Labels y titulo
    ax.set_ylabel('Memoria (MB)', fontsize=12)
    ax.set_xlabel('Configuracion', fontsize=12)
    ax.set_title('Uso de Memoria: FP32 vs Int8 vs Int4', fontsize=14, fontweight='bold')

    # Agregar valores encima de barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} MB',
                ha='center', va='bottom', fontsize=10)

    # Guardar
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Grafico guardado en {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grafico de memoria')
    parser.add_argument('input', help='Archivo memory_usage.csv')
    parser.add_argument('--output', required=True, help='Archivo de salida PNG')
    args = parser.parse_args()

    plot_memory(args.input, args.output)
