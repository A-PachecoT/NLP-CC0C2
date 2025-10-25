#!/usr/bin/env python3
"""Benchmark cuantizacion: FP32 vs Int8 vs Int4"""
import torch
import argparse
import time
import json
from pathlib import Path
from eval import load_model, load_tokens, calculate_perplexity
from quant import quantize_model, get_model_size
from generate import generate_with_cache

def benchmark_quantized_model(model, name, test_tokens, max_gen=50, reps=3, device='cpu'):
    """Benchmark de un modelo cuantizado"""
    model.eval()
    model.to(device)

    # 1. Medir tamaño
    size_mb = get_model_size(model)

    # 2. Medir perplexity
    print(f"  Calculando perplexity...")
    perplexity, loss = calculate_perplexity(model, test_tokens, device=device)

    # 3. Medir latencia de generacion
    print(f"  Midiendo latencia...")
    input_ids = torch.tensor([[1]], dtype=torch.long, device=device)

    # Warmup
    _ = generate_with_cache(model, input_ids, max_length=10, device=device)

    times = []
    for _ in range(reps):
        start = time.perf_counter()
        _ = generate_with_cache(model, input_ids, max_length=max_gen, device=device)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    mean_time = sum(times) / len(times)
    std_time = torch.tensor(times).std().item()

    return {
        'name': name,
        'size_mb': size_mb,
        'perplexity': perplexity,
        'loss': loss,
        'latency_ms': mean_time,
        'latency_std': std_time
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dist/model.tar.gz', help='Modelo .tar.gz')
    parser.add_argument('--length', type=int, default=50, help='Longitud generacion')
    parser.add_argument('--reps', type=int, default=3, help='Repeticiones')
    parser.add_argument('--output', required=True, help='Archivo CSV salida')
    parser.add_argument('--seed', type=int, default=42, help='Seed para reproducibilidad')
    args = parser.parse_args()

    # Configuracion determinista
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Cargando modelo base FP32...")
    model_fp32 = load_model(args.model, device=device)

    print("Cargando tokens de test...")
    tokens = load_tokens('out/tokens.jsonl')
    test_tokens = tokens[int(len(tokens)*0.8):]

    results = []

    # Benchmark FP32
    print("\n[1/3] Benchmark FP32")
    res_fp32 = benchmark_quantized_model(
        model_fp32, 'FP32', test_tokens,
        max_gen=args.length, reps=args.reps, device=device
    )
    results.append(res_fp32)
    print(f"  Size: {res_fp32['size_mb']:.2f} MB")
    print(f"  Perplexity: {res_fp32['perplexity']:.2f}")
    print(f"  Latency: {res_fp32['latency_ms']:.2f}ms ± {res_fp32['latency_std']:.2f}ms")

    # Benchmark Int8
    print("\n[2/3] Benchmark Int8")
    model_int8 = quantize_model(model_fp32, bits=8)
    res_int8 = benchmark_quantized_model(
        model_int8, 'Int8', test_tokens,
        max_gen=args.length, reps=args.reps, device=device
    )
    results.append(res_int8)
    print(f"  Size: {res_int8['size_mb']:.2f} MB ({res_fp32['size_mb']/res_int8['size_mb']:.2f}x)")
    print(f"  Perplexity: {res_int8['perplexity']:.2f} (delta: {res_int8['perplexity']-res_fp32['perplexity']:.2f})")
    print(f"  Latency: {res_int8['latency_ms']:.2f}ms ± {res_int8['latency_std']:.2f}ms")

    # Benchmark Int4
    print("\n[3/3] Benchmark Int4")
    model_int4 = quantize_model(model_fp32, bits=4)
    res_int4 = benchmark_quantized_model(
        model_int4, 'Int4', test_tokens,
        max_gen=args.length, reps=args.reps, device=device
    )
    results.append(res_int4)
    print(f"  Size: {res_int4['size_mb']:.2f} MB ({res_fp32['size_mb']/res_int4['size_mb']:.2f}x)")
    print(f"  Perplexity: {res_int4['perplexity']:.2f} (delta: {res_int4['perplexity']-res_fp32['perplexity']:.2f})")
    print(f"  Latency: {res_int4['latency_ms']:.2f}ms ± {res_int4['latency_std']:.2f}ms")

    # Guardar CSV principal
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write("config,size_mb,perplexity,loss,latency_ms,latency_std\n")
        for r in results:
            f.write(f"{r['name']},{r['size_mb']:.4f},{r['perplexity']:.4f},"
                   f"{r['loss']:.4f},{r['latency_ms']:.4f},{r['latency_std']:.4f}\n")

    print(f"\nResultados guardados en {args.output}")

    # Guardar archivos separados segun parcial.md
    output_dir = Path(args.output).parent

    # 1. memory_usage.csv
    memory_path = output_dir / 'memory_usage.csv'
    with open(memory_path, 'w') as f:
        f.write("config,memory_mb\n")
        for r in results:
            f.write(f"{r['name']},{r['size_mb']:.4f}\n")
    print(f"Memoria guardada en {memory_path}")

    # 2. accuracy_drop.json
    import json
    accuracy_path = output_dir / 'accuracy_drop.json'
    accuracy_data = {
        r['name']: {
            'perplexity': round(r['perplexity'], 2),
            'delta': round(r['perplexity'] - res_fp32['perplexity'], 2)
        }
        for r in results
    }
    with open(accuracy_path, 'w') as f:
        json.dump(accuracy_data, f, indent=2)
    print(f"Accuracy drop guardado en {accuracy_path}")

    # 3. bench_latency.csv (latencia con std)
    latency_path = output_dir / 'bench_latency.csv'
    with open(latency_path, 'w') as f:
        f.write("config,latency_ms,latency_std\n")
        for r in results:
            f.write(f"{r['name']},{r['latency_ms']:.4f},{r['latency_std']:.4f}\n")
    print(f"Latencia guardada en {latency_path}")

    # Resumen
    print("\n=== RESUMEN ===")
    print(f"Compresion Int8: {res_fp32['size_mb']/res_int8['size_mb']:.2f}x")
    print(f"Compresion Int4: {res_fp32['size_mb']/res_int4['size_mb']:.2f}x")
    print(f"Accuracy drop Int8: +{res_int8['perplexity']-res_fp32['perplexity']:.2f} perplexity")
    print(f"Accuracy drop Int4: +{res_int4['perplexity']-res_fp32['perplexity']:.2f} perplexity")
