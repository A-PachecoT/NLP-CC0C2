#!/usr/bin/env python3
"""Tests para eval"""
import pytest
import torch
import tempfile
import tarfile
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from eval import load_model, load_tokens, calculate_perplexity
from train import save_model
from transformer import MiniTransformer

def test_save_and_load_model():
    """Test guardado y carga de modelo"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Crear modelo
        model = MiniTransformer(vocab_size=100, dim=64, heads=4)

        # Guardar
        tar_path = Path(tmpdir) / 'model.tar.gz'
        save_model(model, str(tar_path))

        assert tar_path.exists()

        # Cargar
        loaded_model = load_model(str(tar_path))
        assert loaded_model.vocab_size == 100
        assert loaded_model.dim == 64

def test_load_tokens_eval():
    """Test carga de tokens para eval"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens_path = Path(tmpdir) / 'tokens.jsonl'
        with open(tokens_path, 'w') as f:
            json.dump({'tokens': [1, 2, 3, 4, 5], 'vocab_size': 100}, f)

        tokens = load_tokens(str(tokens_path))
        assert tokens == [1, 2, 3, 4, 5]

def test_calculate_perplexity():
    """Test calculo de perplexity"""
    model = MiniTransformer(vocab_size=100, dim=64, heads=4)
    model.eval()

    tokens = list(range(10, 90))  # 80 tokens

    perplexity, loss = calculate_perplexity(model, tokens, seq_len=16)

    assert isinstance(perplexity, float)
    assert isinstance(loss, float)
    assert perplexity > 0
    assert loss > 0

def test_perplexity_consistent():
    """Test que perplexity es consistente"""
    model = MiniTransformer(vocab_size=100, dim=64, heads=4)
    model.eval()

    tokens = list(range(10, 90))

    ppl1, _ = calculate_perplexity(model, tokens, seq_len=16)
    ppl2, _ = calculate_perplexity(model, tokens, seq_len=16)

    assert ppl1 == ppl2  # Determinista en eval mode
