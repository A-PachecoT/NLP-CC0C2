#!/usr/bin/env python3
"""Tests para attention"""
import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from attention import MultiHeadAttention

def test_attention_creation():
    """Test creacion de atencion"""
    attn = MultiHeadAttention(dim=64, heads=4)
    assert attn.dim == 64
    assert attn.heads == 4
    assert attn.head_dim == 16

def test_attention_forward():
    """Test forward pass"""
    attn = MultiHeadAttention(dim=64, heads=4)
    x = torch.randn(2, 10, 64)  # (batch, seq, dim)

    out = attn(x)
    assert out.shape == (2, 10, 64)

def test_attention_causal_mask():
    """Test mascara causal"""
    attn = MultiHeadAttention(dim=64, heads=4)
    x = torch.randn(1, 5, 64)

    mask = torch.triu(torch.ones(5, 5), diagonal=1).bool()
    out = attn(x, mask=mask)
    assert out.shape == (1, 5, 64)

def test_attention_kv_cache():
    """Test KV-cache"""
    attn = MultiHeadAttention(dim=64, heads=4)
    x1 = torch.randn(1, 3, 64)

    # Primera pasada sin cache
    kv_cache = {}
    out1 = attn(x1, kv_cache=kv_cache)
    assert 'k' in kv_cache
    assert 'v' in kv_cache
    assert kv_cache['k'].shape[1] == 3  # seq_len

    # Segunda pasada con cache (solo 1 token nuevo)
    x2 = torch.randn(1, 1, 64)
    out2 = attn(x2, kv_cache=kv_cache)
    assert kv_cache['k'].shape[1] == 4  # 3 + 1
    assert out2.shape == (1, 1, 64)

def test_attention_different_heads():
    """Test diferentes numeros de heads"""
    for heads in [1, 2, 4, 8]:
        attn = MultiHeadAttention(dim=64, heads=heads)
        x = torch.randn(1, 5, 64)
        out = attn(x)
        assert out.shape == (1, 5, 64)
