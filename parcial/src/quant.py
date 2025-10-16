#!/usr/bin/env python3
"""Cuantizacion de modelos: Int8 y Int4"""
import torch
import torch.nn as nn
import copy

def quantize_tensor_int8(tensor):
    """Cuantiza tensor FP32 a Int8 (simetrico)"""
    # Calcular scale
    abs_max = torch.max(torch.abs(tensor))
    scale = abs_max / 127.0

    # Cuantizar
    q_tensor = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)

    return q_tensor, scale

def dequantize_tensor_int8(q_tensor, scale):
    """Descuantiza Int8 a FP32"""
    return q_tensor.to(torch.float32) * scale

def quantize_tensor_int4(tensor):
    """Cuantiza tensor FP32 a Int4 (simetrico)"""
    # Calcular scale
    abs_max = torch.max(torch.abs(tensor))
    scale = abs_max / 7.0  # Int4: -8 a 7

    # Cuantizar
    q_tensor = torch.round(tensor / scale).clamp(-8, 7).to(torch.int8)  # Guardado como int8

    return q_tensor, scale

def dequantize_tensor_int4(q_tensor, scale):
    """Descuantiza Int4 a FP32"""
    return q_tensor.to(torch.float32) * scale

class QuantizedLinear(nn.Module):
    """Capa Linear cuantizada"""
    def __init__(self, weight, bias, scale, bits=8):
        super().__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('scale', scale)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        self.bits = bits

    def forward(self, x):
        # Descuantizar weight para forward pass
        if self.bits == 8:
            weight_fp = dequantize_tensor_int8(self.weight, self.scale)
        else:  # 4 bits
            weight_fp = dequantize_tensor_int4(self.weight, self.scale)

        return nn.functional.linear(x, weight_fp, self.bias)

def quantize_model(model, bits=8):
    """Cuantiza todas las capas Linear del modelo"""
    model_q = copy.deepcopy(model)

    # Cuantizar recursivamente
    def quantize_module(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Cuantizar pesos
                weight = child.weight.data
                bias = child.bias.data if child.bias is not None else None

                if bits == 8:
                    q_weight, scale = quantize_tensor_int8(weight)
                else:  # 4 bits
                    q_weight, scale = quantize_tensor_int4(weight)

                # Reemplazar con QuantizedLinear
                q_linear = QuantizedLinear(q_weight, bias, scale, bits=bits)
                setattr(module, name, q_linear)
            else:
                quantize_module(child)

    quantize_module(model_q)
    return model_q

def get_model_size(model):
    """Calcula tamaño del modelo en MB"""
    total_params = 0
    total_bytes = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params

        # Calcular bytes según dtype
        if param.dtype == torch.float32:
            bytes_per_param = 4
        elif param.dtype == torch.int8:
            bytes_per_param = 1
        elif param.dtype == torch.float16:
            bytes_per_param = 2
        else:
            bytes_per_param = 4  # Default

        total_bytes += num_params * bytes_per_param

    # Incluir buffers
    for buffer in model.buffers():
        num_params = buffer.numel()
        if buffer.dtype == torch.float32:
            bytes_per_param = 4
        elif buffer.dtype == torch.int8:
            bytes_per_param = 1
        else:
            bytes_per_param = 4

        total_bytes += num_params * bytes_per_param

    size_mb = total_bytes / (1024 * 1024)
    return size_mb

if __name__ == '__main__':
    # Test simple
    import argparse
    from eval import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Modelo .tar.gz')
    parser.add_argument('--bits', type=int, choices=[8, 4], default=8)
    args = parser.parse_args()

    # Cargar modelo
    print(f"Cargando modelo desde {args.model}")
    model = load_model(args.model)

    # Cuantizar
    print(f"Cuantizando a Int{args.bits}")
    model_q = quantize_model(model, bits=args.bits)

    # Comparar tamaños
    size_fp32 = get_model_size(model)
    size_quant = get_model_size(model_q)

    print(f"\nTamaños:")
    print(f"  FP32: {size_fp32:.2f} MB")
    print(f"  Int{args.bits}: {size_quant:.2f} MB")
    print(f"  Reduccion: {size_fp32/size_quant:.2f}x")
