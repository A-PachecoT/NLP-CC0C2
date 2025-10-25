#!/usr/bin/env python3
"""Tests para bench"""
import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bench import benchmark_generation
from transformer import MiniTransformer

@pytest.fixture
def model():
    """Modelo simple para tests"""
    return MiniTransformer(vocab_size=100, dim=64, heads=4)

def test_benchmark_generation(model):
    """Test benchmark basico"""
    model.eval()

    times_cache, times_no_cache = benchmark_generation(
        model, max_length=10, device='cpu', reps=2, warmup=1
    )

    assert len(times_cache) == 2
    assert len(times_no_cache) == 2
    assert all(t > 0 for t in times_cache)
    assert all(t > 0 for t in times_no_cache)

def test_benchmark_returns_times(model):
    """Test que retorna tiempos validos"""
    model.eval()

    times_cache, times_no_cache = benchmark_generation(
        model, max_length=5, device='cpu', reps=3
    )

    # Verificar estructura
    assert isinstance(times_cache, list)
    assert isinstance(times_no_cache, list)
    assert len(times_cache) == 3
    assert len(times_no_cache) == 3
