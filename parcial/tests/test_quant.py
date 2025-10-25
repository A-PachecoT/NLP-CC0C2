#!/usr/bin/env python3
"""Tests para cuantizacion"""
import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quant import (
    quantize_tensor_int8, dequantize_tensor_int8,
    quantize_tensor_int4, dequantize_tensor_int4,
    quantize_model, get_model_size, QuantizedLinear
)
from transformer import MiniTransformer

def test_quantize_int8():
    """Test cuantizacion Int8"""
    tensor = torch.randn(10, 10)
    q_tensor, scale = quantize_tensor_int8(tensor)

    assert q_tensor.dtype == torch.int8
    assert q_tensor.min() >= -128
    assert q_tensor.max() <= 127

def test_dequantize_int8():
    """Test descuantizacion Int8"""
    tensor = torch.randn(10, 10)
    q_tensor, scale = quantize_tensor_int8(tensor)
    dq_tensor = dequantize_tensor_int8(q_tensor, scale)

    assert dq_tensor.dtype == torch.float32
    # Error pequeño por cuantizacion
    assert torch.allclose(tensor, dq_tensor, atol=0.1)

def test_quantize_int4():
    """Test cuantizacion Int4"""
    tensor = torch.randn(10, 10)
    q_tensor, scale = quantize_tensor_int4(tensor)

    assert q_tensor.dtype == torch.int8  # Almacenado como int8
    assert q_tensor.min() >= -8
    assert q_tensor.max() <= 7

def test_dequantize_int4():
    """Test descuantizacion Int4"""
    tensor = torch.randn(10, 10)
    q_tensor, scale = quantize_tensor_int4(tensor)
    dq_tensor = dequantize_tensor_int4(q_tensor, scale)

    assert dq_tensor.dtype == torch.float32
    # Mayor error por menos bits
    assert torch.allclose(tensor, dq_tensor, atol=0.2)

def test_quantized_linear():
    """Test capa Linear cuantizada"""
    weight = torch.randn(20, 10)
    bias = torch.randn(20)
    q_weight, scale = quantize_tensor_int8(weight)

    q_linear = QuantizedLinear(q_weight, bias, scale, bits=8)

    x = torch.randn(5, 10)
    out = q_linear(x)
    assert out.shape == (5, 20)

def test_quantize_model():
    """Test cuantizacion de modelo completo"""
    model = MiniTransformer(vocab_size=100, dim=64, heads=4)
    model_q = quantize_model(model, bits=8)

    # Verificar que tiene QuantizedLinear
    has_quantized = False
    for module in model_q.modules():
        if isinstance(module, QuantizedLinear):
            has_quantized = True
            break
    assert has_quantized

def test_model_size():
    """Test calculo de tamaño"""
    model_fp32 = MiniTransformer(vocab_size=100, dim=64, heads=4)
    model_int8 = quantize_model(model_fp32, bits=8)

    size_fp32 = get_model_size(model_fp32)
    size_int8 = get_model_size(model_int8)

    # Int8 debe ser mas pequeño
    assert size_int8 < size_fp32

def test_quantized_forward():
    """Test forward de modelo cuantizado"""
    model = MiniTransformer(vocab_size=100, dim=64, heads=4)
    model_q = quantize_model(model, bits=8)
    model_q.eval()

    x = torch.randint(0, 100, (1, 5))
    with torch.no_grad():
        logits = model_q(x)
    assert logits.shape == (1, 5, 100)
