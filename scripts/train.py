import tensorflow as tf
import numpy as np
from model import GPT
import tokenizer as tk
import json
import argparse

# Configurable parameters.
use_spacy = True
num_blocks = 12
num_attention_heads = 12
context_size = 512
attention_dim = 768
feed_forward_dim = 3072
activation = 'gelu'
token_embed_dim = 384
dropout = 0.1
epochs = 1
batch_size = 1

if __name__ == '__main__':
    print('Available devices:', tf.config.list_physical_devices())

    parser = argparse.ArgumentParser()
    help_text = 'If true, use Spacy tokenizer instead of character wise tokenization.'
    parser.add_argument('--use_spacy', type=bool, default=use_spacy, help=help_text)
    help_text = 'Number of transformer blocks.'
    parser.add_argument('--num_blocks', type=int, default=num_blocks, help=help_text)
    help_text = 'Number of attention heads in multi-head attention.'
    parser.add_argument('--num_attention_heads', type=int, default=num_attention_heads, help=help_text)
    help_text = 'Number of tokens in each context.'
    parser.add_argument('--context_size', type=int, default=context_size, help=help_text)
    help_text = 'Dimension of attention layers.'
    parser.add_argument('--attention_dim', type=int, default=attention_dim, help=help_text)
    help_text = 'Dimension of feed forward layers.'
    parser.add_argument('--feed_forward_dim', type=int, default=feed_forward_dim, help=help_text)
    help_text = 'Activation function.'
    parser.add_argument('--activation', type=str, default=activation, help=help_text)
    help_text = 'Dimension of token embeddings.'
    parser.add_argument('--token_embed_dim', type=int, default=token_embed_dim, help=help_text)
    help_text = 'Dropout rate.'
    parser.add_argument('--dropout', type=float, default=dropout, help=help_text)
    help_text = 'Number of epochs.'
    parser.add_argument('--epochs', type=int, default=epochs, help=help_text)
    help_text = 'Batch size.'
    parser.add_argument('--batch_size', type=int, default=batch_size, help=help_text)
    args = parser.parse_args()

    with open ('../data/preprocessed/vocab.json', 'r') as f:
        vocab = json.load(f)
    vocab_size = vocab['vocab_size']
    
    with open('../data/tiny_shakespeare.txt', 'r', encoding='utf8') as f:
        text = f.read()

    if args.use_spacy:
        tokenized_text = tk.spacy_tokenize(text)
    else:
        tokenized_text = tk.character_tokenize(text)

    model = GPT(
        num_blocks = num_blocks,
        num_attention_heads = num_attention_heads,
        context_size = context_size,
        attention_dim = attention_dim,
        feed_forward_dim = feed_forward_dim,
        activation = activation,
        token_embed_dim = token_embed_dim,
        dropout = dropout,
        vocab_size = vocab_size
    )

    num_contexts = int(len(tokenized_text) / context_size)
    num_tokens = num_contexts * context_size

    input_tokens = np.array(
        tokenized_text[:num_tokens]
    ).reshape(num_contexts, context_size)
    print("Inputs shape:", input_tokens.shape)

    one_hot_labels = np.array(
        tk.tokens_to_labels(tokenized_text, vocab_size)[:num_tokens]
    ).reshape(num_contexts, context_size, vocab_size)
    print("Labels shape:", one_hot_labels.shape)

    model.compile(
        optimizer = 'adam', 
        loss = 'categorical_crossentropy', 
        metrics=['accuracy']
    )

    model.fit(
        x = input_tokens, 
        y = one_hot_labels, 
        epochs = epochs, 
        batch_size = batch_size,
        verbose = 1
    )

    model.network.save('../data/output/gpt')