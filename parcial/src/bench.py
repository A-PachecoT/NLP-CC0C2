#!/usr/bin/env python3
"""Benchmarks de latencia con KV-cache real"""
import torch
import argparse
import time
from pathlib import Path
from eval import load_model
from generate import generate_with_cache, generate_without_cache

def benchmark_generation(model, max_length=50, device='cpu', reps=3, warmup=1):
    """Benchmark generacion con/sin KV-cache"""
    # Input inicial
    input_ids = torch.tensor([[1]], dtype=torch.long, device=device)

    # Warmup
    for _ in range(warmup):
        _ = generate_with_cache(model, input_ids, max_length=10, device=device)
        _ = generate_without_cache(model, input_ids, max_length=10, device=device)

    # Benchmark con cache
    times_with_cache = []
    for _ in range(reps):
        start = time.perf_counter()
        _ = generate_with_cache(model, input_ids, max_length=max_length, device=device)
        end = time.perf_counter()
        times_with_cache.append((end - start) * 1000)  # ms

    # Benchmark sin cache
    times_without_cache = []
    for _ in range(reps):
        start = time.perf_counter()
        _ = generate_without_cache(model, input_ids, max_length=max_length, device=device)
        end = time.perf_counter()
        times_without_cache.append((end - start) * 1000)  # ms

    return times_with_cache, times_without_cache

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dist/model.tar.gz', help='Modelo .tar.gz')
    parser.add_argument('--n', type=int, default=50, help='Longitud a generar')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    print(f"Benchmark: length={args.n}, reps={args.reps}, warmup={args.warmup}")

    # Cargar modelo
    model = load_model(args.model, device=device)

    # Benchmark
    times_cache, times_no_cache = benchmark_generation(
        model, max_length=args.n, device=device, reps=args.reps, warmup=args.warmup
    )

    # Estadisticas
    mean_cache = sum(times_cache) / len(times_cache)
    std_cache = torch.tensor(times_cache).std().item()
    mean_no_cache = sum(times_no_cache) / len(times_no_cache)
    std_no_cache = torch.tensor(times_no_cache).std().item()

    speedup = mean_no_cache / mean_cache

    print(f"Con cache:  {mean_cache:.2f}ms ± {std_cache:.2f}ms")
    print(f"Sin cache:  {mean_no_cache:.2f}ms ± {std_no_cache:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")

    # Guardar CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write("config,mean_ms,std_ms\n")
        f.write(f"with_cache,{mean_cache:.4f},{std_cache:.4f}\n")
        f.write(f"no_cache,{mean_no_cache:.4f},{std_no_cache:.4f}\n")

    print(f"Resultados guardados en {args.output}")
