#!/usr/bin/env python3
"""Atencion multi-head con causal masking y KV-cache opcional"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, "dim debe ser divisible por heads"

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        # Proyecciones lineales Q, K, V, O
        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)
        self.fc_o = nn.Linear(dim, dim)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None, kv_cache=None):
        """
        x: (batch, seq_len, dim)
        mask: (seq_len, seq_len) causal mask
        kv_cache: dict con 'k' y 'v' para autoregresivo
        """
        batch, seq_len, _ = x.shape

        # Proyecciones Q, K, V
        Q = self.fc_q(x)  # (batch, seq_len, dim)
        K = self.fc_k(x)
        V = self.fc_v(x)

        # Si hay cache, concatenar K y V previos
        if kv_cache is not None:
            if 'k' in kv_cache:
                K = torch.cat([kv_cache['k'], K], dim=1)
                V = torch.cat([kv_cache['v'], V], dim=1)
            kv_cache['k'] = K
            kv_cache['v'] = V

        # Reshape para multi-head: (batch, heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Aplicar mascara causal
        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        # Concatenar heads: (batch, seq_len, dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)

        # Proyeccion final
        return self.fc_o(out)

def create_causal_mask(seq_len, device='cpu'):
    """Mascara causal: impide atender tokens futuros"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
    return mask
