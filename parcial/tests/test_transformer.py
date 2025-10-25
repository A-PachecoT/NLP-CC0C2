#!/usr/bin/env python3
"""Tests para transformer"""
import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from transformer import MiniTransformer, PositionalEncoding

def test_positional_encoding_sinusoidal():
    """Test encoding posicional sinusoidal"""
    pos_enc = PositionalEncoding(dim=64, max_len=100, pos_type='sinusoidal')
    x = torch.randn(2, 10, 64)
    out = pos_enc(x)
    assert out.shape == (2, 10, 64)

def test_positional_encoding_rope():
    """Test RoPE"""
    pos_enc = PositionalEncoding(dim=64, max_len=100, pos_type='rope')
    x = torch.randn(2, 10, 64)
    out = pos_enc(x)
    assert out.shape == (2, 10, 64)

def test_transformer_creation():
    """Test creacion de transformer"""
    model = MiniTransformer(vocab_size=100, dim=64, heads=4, max_len=128)
    assert model.vocab_size == 100
    assert model.dim == 64

def test_transformer_forward():
    """Test forward pass"""
    model = MiniTransformer(vocab_size=100, dim=64, heads=4)
    x = torch.randint(0, 100, (2, 10))  # (batch, seq)

    logits = model(x)
    assert logits.shape == (2, 10, 100)  # (batch, seq, vocab)

def test_transformer_with_kv_cache():
    """Test forward con KV-cache"""
    model = MiniTransformer(vocab_size=100, dim=64, heads=4)
    x = torch.randint(0, 100, (1, 5))

    kv_cache = {}
    logits = model(x, kv_cache=kv_cache)
    assert logits.shape == (1, 5, 100)
    assert len(kv_cache) > 0

def test_transformer_eval_mode():
    """Test modo evaluacion"""
    model = MiniTransformer(vocab_size=100, dim=64, heads=4)
    model.eval()

    x = torch.randint(0, 100, (1, 10))
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 10, 100)

def test_transformer_different_dims():
    """Test diferentes dimensiones"""
    for dim in [32, 64, 128]:
        model = MiniTransformer(vocab_size=100, dim=dim, heads=4)
        x = torch.randint(0, 100, (1, 5))
        logits = model(x)
        assert logits.shape == (1, 5, 100)
