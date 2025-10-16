#!/usr/bin/env python3
"""Genera graficos de benchmarks"""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

def plot_benchmarks(input_path, output_path):
    """Lee CSV y genera grafico de barras con matplotlib"""
    data = []
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'config': row['config'],
                'mean': float(row['mean_ms']),
                'std': float(row['std_ms'])
            })

    # Crear grafico
    configs = [d['config'] for d in data]
    means = [d['mean'] for d in data]
    stds = [d['std'] for d in data]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(configs, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    plt.ylabel('Latencia (ms)')
    plt.title('Latencia por configuracion')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Grafico guardado en {output_path}")
    print("\nResultados:")
    for d in data:
        print(f"  {d['config']}: {d['mean']:.2f}ms ± {d['std']:.2f}ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Archivo bench.csv')
    parser.add_argument('--output', required=True, help='Archivo plot.png')
    args = parser.parse_args()

    plot_benchmarks(args.input, args.output)
