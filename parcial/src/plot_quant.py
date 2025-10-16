#!/usr/bin/env python3
"""Genera graficos para benchmarks de cuantizacion"""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

def plot_quant_benchmarks(input_path, output_path):
    """Lee CSV de cuantizacion y genera graficos"""
    data = []
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'config': row['config'],
                'size': float(row['size_mb']),
                'perplexity': float(row['perplexity']),
                'latency': float(row['latency_ms']),
                'latency_std': float(row['latency_std'])
            })

    configs = [d['config'] for d in data]
    sizes = [d['size'] for d in data]
    perplexities = [d['perplexity'] for d in data]
    latencies = [d['latency'] for d in data]
    latency_stds = [d['latency_std'] for d in data]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Crear figura con 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Tamaño (MB)
    ax1.bar(configs, sizes, color='steelblue', alpha=0.7)
    ax1.set_ylabel('Tamaño (MB)')
    ax1.set_title('Compresion de modelo')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Perplexity
    ax2.bar(configs, perplexities, color='coral', alpha=0.7)
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Accuracy (menor es mejor)')
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Latencia
    ax3.bar(configs, latencies, yerr=latency_stds, capsize=5, color='seagreen', alpha=0.7)
    ax3.set_ylabel('Latencia (ms)')
    ax3.set_title('Velocidad de generacion')
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Grafico guardado en {output_path}")
    print("\nResultados:")
    for d in data:
        print(f"  {d['config']:5s}: {d['size']:.2f}MB, {d['perplexity']:.2f} ppl, {d['latency']:.2f}ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Archivo bench_quant.csv')
    parser.add_argument('--output', required=True, help='Archivo plot.png')
    args = parser.parse_args()

    plot_quant_benchmarks(args.input, args.output)
