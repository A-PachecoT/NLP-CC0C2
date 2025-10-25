#!/usr/bin/env python3
"""Tests para tokenizer"""
import pytest
import tempfile
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tokenizer import tokenize_corpus

def test_tokenize_corpus():
    """Test tokenizacion completa"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Crear corpus simple
        corpus_path = Path(tmpdir) / 'corpus.txt'
        corpus_path.write_text("hola mundo hola")

        vocab_path = Path(tmpdir) / 'vocab.txt'
        output_path = Path(tmpdir) / 'tokens.jsonl'

        # Tokenizar
        tokenize_corpus(str(corpus_path), str(vocab_path), str(output_path))

        # Verificar vocab
        assert vocab_path.exists()
        vocab_lines = vocab_path.read_text().strip().split('\n')
        assert '<PAD>' in vocab_lines
        assert '<UNK>' in vocab_lines
        assert '<BOS>' in vocab_lines
        assert '<EOS>' in vocab_lines
        assert 'hola' in vocab_lines
        assert 'mundo' in vocab_lines

        # Verificar tokens
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
            assert 'tokens' in data
            assert 'vocab_size' in data
            assert len(data['tokens']) == 3  # "hola mundo hola"

def test_tokenize_special_tokens():
    """Test tokens especiales"""
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = Path(tmpdir) / 'corpus.txt'
        corpus_path.write_text("test")

        vocab_path = Path(tmpdir) / 'vocab.txt'
        output_path = Path(tmpdir) / 'tokens.jsonl'

        tokenize_corpus(str(corpus_path), str(vocab_path), str(output_path))

        vocab_lines = vocab_path.read_text().strip().split('\n')
        # Primeros 4 deben ser especiales
        assert vocab_lines[0] == '<PAD>'
        assert vocab_lines[1] == '<UNK>'
        assert vocab_lines[2] == '<BOS>'
        assert vocab_lines[3] == '<EOS>'

def test_tokenize_vocab_size():
    """Test tamaño de vocabulario"""
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = Path(tmpdir) / 'corpus.txt'
        corpus_path.write_text("a b c a b a")

        vocab_path = Path(tmpdir) / 'vocab.txt'
        output_path = Path(tmpdir) / 'tokens.jsonl'

        tokenize_corpus(str(corpus_path), str(vocab_path), str(output_path))

        with open(output_path) as f:
            data = json.load(f)
            # 4 especiales + 3 unicos (a, b, c)
            assert data['vocab_size'] == 7
