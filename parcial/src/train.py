#!/usr/bin/env python3
"""Entrenamiento Mini-Transformer con warmup y gradient clipping"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path
from transformer import MiniTransformer

def load_tokens(path):
    """Carga tokens desde jsonl"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data['tokens'], data['vocab_size']

def create_batches(tokens, batch_size, seq_len, device='cpu'):
    """Genera batches de secuencias"""
    tokens_t = torch.tensor(tokens, dtype=torch.long)
    n = len(tokens) - seq_len

    indices = torch.randperm(n)[:batch_size * 10]

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i+batch_size]
        x = torch.stack([tokens_t[j:j+seq_len] for j in batch_idx]).to(device)
        y = torch.stack([tokens_t[j+1:j+seq_len+1] for j in batch_idx]).to(device)
        yield x, y

def warmup_lr_schedule(optimizer, step, warmup_steps, base_lr):
    """Learning rate con warmup lineal"""
    if step < warmup_steps:
        lr = base_lr * (step + 1) / warmup_steps
    else:
        lr = base_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def train(model, tokens, lr=0.001, steps=100, batch_size=4, seq_len=64, warmup_steps=10, device='cpu'):
    """Loop de entrenamiento"""
    print(f"Entrenando {steps} pasos, lr={lr}, warmup={warmup_steps}")

    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        # Warmup learning rate
        current_lr = warmup_lr_schedule(optimizer, step, warmup_steps, lr)

        # Batch
        batches = list(create_batches(tokens, batch_size, seq_len, device))
        if not batches:
            break

        x, y = batches[0]

        # Forward pass
        logits = model(x)  # (batch, seq_len, vocab)

        # Loss
        loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}/{steps}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

    return model

def save_model(model, path):
    """Guarda modelo en tar.gz con torch.save"""
    import tarfile
    import tempfile
    import os

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Crear directorio temporal
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'model.pt')

        # Guardar modelo con torch.save
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': model.vocab_size,
            'dim': model.dim,
            'pos_type': model.pos_type,
            'max_len': model.max_len,
        }, model_path)

        # Empaquetar en tar.gz
        with tarfile.open(path, 'w:gz') as tar:
            tar.add(model_path, arcname='model.pt')

    print(f"Modelo guardado en {path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Archivo tokens.jsonl')
    parser.add_argument('--output', required=True, help='Archivo modelo .tar.gz')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--pos', default='rope', choices=['rope', 'sinusoidal'])
    parser.add_argument('--seed', type=int, default=42, help='Seed para reproducibilidad')
    args = parser.parse_args()

    # Configuracion determinista
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Cargar tokens
    print(f"Cargando tokens desde {args.input}")
    tokens, vocab_size = load_tokens(args.input)
    print(f"Vocab: {vocab_size}, Tokens: {len(tokens)}")

    # Crear modelo
    print(f"Creando modelo: dim={args.dim}, heads={args.heads}, pos={args.pos}")
    model = MiniTransformer(vocab_size, args.dim, args.heads, pos_type=args.pos)

    # Entrenar
    model = train(model, tokens, lr=args.lr, steps=args.steps, device=device)

    # Guardar
    save_model(model, args.output)
