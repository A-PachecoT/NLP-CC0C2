#!/usr/bin/env python3
"""Tests para generate"""
import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generate import generate_with_cache, generate_without_cache, benchmark_generation as benchmark_gen
from transformer import MiniTransformer

@pytest.fixture
def model():
    """Modelo simple para tests"""
    return MiniTransformer(vocab_size=100, dim=64, heads=4)

def test_generate_with_cache(model):
    """Test generacion con cache"""
    model.eval()
    input_ids = torch.tensor([[1]], dtype=torch.long)

    output = generate_with_cache(model, input_ids, max_length=10)
    assert output.shape[0] == 1
    assert output.shape[1] <= 10

def test_generate_without_cache(model):
    """Test generacion sin cache"""
    model.eval()
    input_ids = torch.tensor([[1]], dtype=torch.long)

    output = generate_without_cache(model, input_ids, max_length=10)
    assert output.shape[0] == 1
    assert output.shape[1] <= 10

def test_generate_same_output(model):
    """Test que ambos metodos generan igual (con misma seed)"""
    model.eval()
    input_ids = torch.tensor([[1]], dtype=torch.long)

    torch.manual_seed(42)
    out1 = generate_with_cache(model, input_ids, max_length=5)

    torch.manual_seed(42)
    out2 = generate_without_cache(model, input_ids, max_length=5)

    assert torch.equal(out1, out2)

def test_generate_different_lengths(model):
    """Test diferentes longitudes"""
    model.eval()
    input_ids = torch.tensor([[1]], dtype=torch.long)

    for length in [5, 10, 20]:
        output = generate_with_cache(model, input_ids, max_length=length)
        assert output.shape[1] <= length

def test_generate_batch(model):
    """Test generacion con batch"""
    model.eval()
    input_ids = torch.tensor([[1], [2]], dtype=torch.long)  # batch=2

    output = generate_with_cache(model, input_ids, max_length=10)
    assert output.shape[0] == 2

def test_generate_device_cpu(model):
    """Test generacion en CPU"""
    model.eval()
    input_ids = torch.tensor([[1]], dtype=torch.long)

    output = generate_with_cache(model, input_ids, max_length=5, device='cpu')
    assert output.device.type == 'cpu'

def test_generate_both_methods_work(model):
    """Test que ambos metodos completan sin error"""
    model.eval()
    input_ids = torch.tensor([[1]], dtype=torch.long)

    # Con cache
    out1 = generate_with_cache(model, input_ids, max_length=10)
    assert out1.shape[1] <= 10

    # Sin cache
    out2 = generate_without_cache(model, input_ids, max_length=10)
    assert out2.shape[1] <= 10

def test_benchmark_from_generate(model):
    """Test benchmark_generation de generate.py"""
    model.eval()
    input_ids = torch.tensor([[1]], dtype=torch.long)

    results = benchmark_gen(model, input_ids, max_length=5, device='cpu', reps=2)

    assert 'with_cache' in results
    assert 'without_cache' in results
    assert results['with_cache']['mean'] > 0
    assert results['with_cache']['std'] >= 0
    assert results['without_cache']['mean'] > 0
    assert results['without_cache']['std'] >= 0
