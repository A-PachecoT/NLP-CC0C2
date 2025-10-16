#!/usr/bin/env python3
"""Mini-Transformer decoder-only con 1 bloque"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention import MultiHeadAttention, create_causal_mask

class PositionalEncoding(nn.Module):
    """Codificacion posicional sinusoidal (Vaswani et al. 2017)"""
    def __init__(self, dim, max_len=512, pos_type='sinusoidal'):
        super().__init__()
        self.pos_type = pos_type

        if pos_type == 'sinusoidal':
            pe = self._sinusoidal(max_len, dim)
        else:  # rope
            pe = self._rope(max_len, dim)

        self.register_buffer('pe', pe)

    def _sinusoidal(self, max_len, dim):
        pos = torch.arange(max_len).unsqueeze(1).float()
        i = torch.arange(dim).unsqueeze(0).float()
        angles = pos / torch.pow(10000, (2 * (i // 2)) / dim)

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        return pe

    def _rope(self, max_len, dim):
        pos = torch.arange(max_len).unsqueeze(1).float()
        i = torch.arange(dim // 2).unsqueeze(0).float()
        theta = pos / torch.pow(10000, (2 * i) / dim)

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.cos(theta)
        pe[:, 1::2] = torch.sin(theta)
        return pe

    def forward(self, x):
        # x: (batch, seq_len, dim)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class FeedForward(nn.Module):
    """FFN de 2 capas con ReLU"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    """1 bloque decoder con attention + FFN"""
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads)
        self.ffn = FeedForward(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None, kv_cache=None):
        # Attention con residual (pre-norm)
        x = x + self.attn(self.ln1(x), mask, kv_cache)
        # FFN con residual (pre-norm)
        x = x + self.ffn(self.ln2(x))
        return x

class MiniTransformer(nn.Module):
    """Decoder-only con 1 bloque"""
    def __init__(self, vocab_size, dim, heads, max_len=512, pos_type='rope'):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.pos_type = pos_type
        self.max_len = max_len

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(dim, max_len, pos_type)

        # 1 bloque decoder
        self.block = TransformerBlock(dim, heads)

        # Output projection
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(self, tokens, kv_cache=None):
        """
        tokens: (batch, seq_len) indices de tokens
        """
        batch, seq_len = tokens.shape

        # Token embeddings + positional encoding
        x = self.token_emb(tokens)  # (batch, seq_len, dim)
        x = self.pos_enc(x)

        # Mascara causal
        device = tokens.device
        mask = create_causal_mask(seq_len, device=device)

        # 1 bloque transformer
        x = self.block(x, mask, kv_cache)

        # Proyeccion a logits
        logits = self.out_proj(x)  # (batch, seq_len, vocab_size)

        return logits
