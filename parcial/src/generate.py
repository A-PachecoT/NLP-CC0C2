#!/usr/bin/env python3
"""Generacion autoregresiva con y sin KV-cache"""
import torch
import torch.nn.functional as F
import time

def generate_with_cache(model, input_ids, max_length=50, temperature=1.0, device='cpu'):
    """Generacion autoregresiva CON KV-cache"""
    model.eval()
    generated = input_ids.clone()
    kv_cache = {}

    with torch.no_grad():
        for _ in range(max_length):
            # Solo procesar el ultimo token
            if len(kv_cache) > 0:
                input_token = generated[:, -1:]
            else:
                input_token = generated

            # Forward con cache
            logits = model(input_token, kv_cache=kv_cache)
            next_token_logits = logits[:, -1, :] / temperature

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            if generated.size(1) >= max_length:
                break

    return generated

def generate_without_cache(model, input_ids, max_length=50, temperature=1.0, device='cpu'):
    """Generacion autoregresiva SIN KV-cache"""
    model.eval()
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_length):
            # Procesar toda la secuencia cada vez
            logits = model(generated)
            next_token_logits = logits[:, -1, :] / temperature

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            if generated.size(1) >= max_length:
                break

    return generated

def benchmark_generation(model, input_ids, max_length=50, device='cpu', reps=3):
    """Benchmark generacion con/sin cache"""
    times_with_cache = []
    times_without_cache = []

    # Warmup
    _ = generate_with_cache(model, input_ids, max_length=10, device=device)
    _ = generate_without_cache(model, input_ids, max_length=10, device=device)

    # Benchmark con cache
    for _ in range(reps):
        start = time.perf_counter()
        _ = generate_with_cache(model, input_ids, max_length=max_length, device=device)
        end = time.perf_counter()
        times_with_cache.append((end - start) * 1000)  # ms

    # Benchmark sin cache
    for _ in range(reps):
        start = time.perf_counter()
        _ = generate_without_cache(model, input_ids, max_length=max_length, device=device)
        end = time.perf_counter()
        times_without_cache.append((end - start) * 1000)  # ms

    return {
        'with_cache': {
            'mean': sum(times_with_cache) / len(times_with_cache),
            'std': torch.tensor(times_with_cache).std().item(),
            'times': times_with_cache
        },
        'without_cache': {
            'mean': sum(times_without_cache) / len(times_without_cache),
            'std': torch.tensor(times_without_cache).std().item(),
            'times': times_without_cache
        }
    }

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from eval import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Modelo .tar.gz')
    parser.add_argument('--length', type=int, default=50, help='Longitud a generar')
    parser.add_argument('--reps', type=int, default=3, help='Repeticiones')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    # Cargar modelo
    print(f"Cargando modelo desde {args.model}")
    model = load_model(args.model, device=device)

    # Input inicial (BOS token)
    input_ids = torch.tensor([[1]], dtype=torch.long, device=device)

    # Benchmark
    print(f"Benchmarking generacion (length={args.length}, reps={args.reps})")
    results = benchmark_generation(model, input_ids, max_length=args.length, device=device, reps=args.reps)

    print("\nResultados:")
    print(f"  Con cache:  {results['with_cache']['mean']:.2f}ms ± {results['with_cache']['std']:.2f}ms")
    print(f"  Sin cache:  {results['without_cache']['mean']:.2f}ms ± {results['without_cache']['std']:.2f}ms")

    speedup = results['without_cache']['mean'] / results['with_cache']['mean']
    print(f"  Speedup: {speedup:.2f}x")
