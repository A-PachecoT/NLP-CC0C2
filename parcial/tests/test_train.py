#!/usr/bin/env python3
"""Tests para train"""
import pytest
import torch
import tempfile
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from train import load_tokens, create_batches, warmup_lr_schedule, train
from transformer import MiniTransformer

def test_load_tokens():
    """Test carga de tokens"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens_path = Path(tmpdir) / 'tokens.jsonl'
        data = {'tokens': [1, 2, 3, 4], 'vocab_size': 100}
        with open(tokens_path, 'w') as f:
            json.dump(data, f)

        tokens, vocab_size = load_tokens(str(tokens_path))
        assert tokens == [1, 2, 3, 4]
        assert vocab_size == 100

def test_create_batches():
    """Test creacion de batches"""
    tokens = list(range(100))
    batches = list(create_batches(tokens, batch_size=2, seq_len=10))

    assert len(batches) > 0
    x, y = batches[0]
    assert x.shape == (2, 10)
    assert y.shape == (2, 10)

def test_warmup_lr_schedule():
    """Test warmup de learning rate"""
    model = MiniTransformer(vocab_size=100, dim=64, heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Paso 0 (warmup)
    lr = warmup_lr_schedule(optimizer, step=0, warmup_steps=10, base_lr=0.001)
    assert lr < 0.001

    # Paso 5 (warmup medio)
    lr = warmup_lr_schedule(optimizer, step=5, warmup_steps=10, base_lr=0.001)
    assert 0 < lr < 0.001

    # Paso 10 (post-warmup)
    lr = warmup_lr_schedule(optimizer, step=10, warmup_steps=10, base_lr=0.001)
    assert lr == 0.001

def test_train_function():
    """Test funcion de entrenamiento"""
    model = MiniTransformer(vocab_size=100, dim=64, heads=4)
    tokens = list(range(10, 90))  # 80 tokens, within vocab range

    # Entrenar pocos pasos
    train(model, tokens, lr=0.001, steps=5, batch_size=2, seq_len=10, warmup_steps=2)

    # Verificar que el modelo esta en modo train
    assert model.training == True

def test_train_updates_weights():
    """Test que train actualiza pesos"""
    model = MiniTransformer(vocab_size=100, dim=64, heads=4)
    tokens = list(range(10, 90))  # Within vocab range

    # Guardar peso inicial
    initial_weight = model.token_emb.weight.data.clone()

    # Entrenar
    train(model, tokens, lr=0.01, steps=10, batch_size=2, seq_len=10)

    # Verificar que cambió
    assert not torch.equal(initial_weight, model.token_emb.weight.data)
