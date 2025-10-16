#!/usr/bin/env python3
"""Evaluacion: calcula perplexity del modelo"""
import json
import torch
import torch.nn as nn
import argparse
import tarfile
import tempfile
from pathlib import Path
from transformer import MiniTransformer

def load_model(path, device='cpu'):
    """Carga modelo desde tar.gz"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extraer tar.gz
        with tarfile.open(path, 'r:gz') as tar:
            tar.extractall(tmpdir)

        model_path = Path(tmpdir) / 'model.pt'
        checkpoint = torch.load(model_path, map_location=device)

    # Reconstruir modelo
    model = MiniTransformer(
        vocab_size=checkpoint['vocab_size'],
        dim=checkpoint['dim'],
        heads=4,  # hardcoded
        max_len=checkpoint.get('max_len', 512),
        pos_type=checkpoint.get('pos_type', 'rope')
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model

def load_tokens(path):
    """Carga tokens desde jsonl"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data['tokens']

def calculate_perplexity(model, tokens, seq_len=64, device='cpu'):
    """Calcula perplexity en tokens de evaluacion"""
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    count = 0

    tokens_t = torch.tensor(tokens, dtype=torch.long, device=device)

    # Evaluar en batches de seq_len
    with torch.no_grad():
        for i in range(0, len(tokens) - seq_len, seq_len):
            x = tokens_t[i:i+seq_len].unsqueeze(0)  # (1, seq_len)
            y = tokens_t[i+1:i+seq_len+1]  # (seq_len,)

            # Forward pass
            logits = model(x)[0]  # (seq_len, vocab)

            # Cross-entropy
            loss = criterion(logits, y)

            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity, avg_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Modelo .tar.gz')
    parser.add_argument('--output', required=True, help='Archivo metrics.json')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar modelo
    print(f"Cargando modelo desde {args.model}")
    model = load_model(args.model, device=device)

    # Cargar tokens (usar mismo corpus para simplificar)
    tokens = load_tokens('out/tokens.jsonl')
    test_tokens = tokens[int(len(tokens)*0.8):]  # Ultimos 20% para test

    # Calcular perplexity
    print(f"Evaluando en {len(test_tokens)} tokens")
    perplexity, loss = calculate_perplexity(model, test_tokens, device=device)

    # Guardar metricas
    metrics = {
        'perplexity': float(perplexity),
        'loss': float(loss),
        'test_tokens': len(test_tokens)
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Perplexity: {perplexity:.2f}")
    print(f"Loss: {loss:.4f}")
    print(f"Metricas guardadas en {args.output}")
