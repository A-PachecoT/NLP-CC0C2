#!/usr/bin/env python3
"""Tokenizer basico por espacios para Mini-Transformer"""
import json
from collections import Counter

def tokenize_corpus(corpus_path, vocab_path, output_path):
    """Tokeniza corpus y genera vocab"""
    # Leer corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenizar por espacios
    tokens = text.split()

    # Crear vocabulario con tokens unicos
    vocab_counter = Counter(tokens)
    vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + [w for w, _ in vocab_counter.most_common()]

    # Mapeo token -> id
    token2id = {t: i for i, t in enumerate(vocab)}

    # Guardar vocabulario
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(f"{token}\n")

    # Convertir tokens a IDs
    token_ids = [token2id.get(t, token2id['<UNK>']) for t in tokens]

    # Guardar tokens en formato jsonl
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'tokens': token_ids, 'vocab_size': len(vocab)}, f)
        f.write('\n')

    print(f"Vocab size: {len(vocab)}")
    print(f"Total tokens: {len(token_ids)}")
    print(f"Unique tokens: {len(vocab_counter)}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tokenizador basico')
    parser.add_argument('input', help='Archivo de corpus')
    parser.add_argument('--output', required=True, help='Archivo de salida (jsonl)')
    parser.add_argument('--vocab', required=True, help='Archivo de vocabulario')
    args = parser.parse_args()

    tokenize_corpus(args.input, args.vocab, args.output)
