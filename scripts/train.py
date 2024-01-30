import tensorflow as tf
import numpy as np
from model import GPT
import tokenizer as tk
import json
import argparse
import os

def main():
    print('Available devices:', tf.config.list_physical_devices())

    parser = argparse.ArgumentParser()
    help_text = 'Number of transformer blocks.'
    parser.add_argument('--num_blocks', type=int, default=12, help=help_text)
    help_text = 'Number of attention heads in multi-head attention.'
    parser.add_argument('--num_attention_heads', type=int, default=12, help=help_text)
    help_text = 'Number of tokens in each context.'
    parser.add_argument('--context_size', type=int, default=512, help=help_text)
    help_text = 'Dimension of attention layers.'
    parser.add_argument('--attention_dim', type=int, default=768, help=help_text)
    help_text = 'Dimension of feed forward layers.'
    parser.add_argument('--feed_forward_dim', type=int, default=3072, help=help_text)
    help_text = 'Activation function.'
    parser.add_argument('--activation', type=str, default='gelu', help=help_text)
    help_text = 'Dimension of token embeddings.'
    parser.add_argument('--token_embed_dim', type=int, default=384, help=help_text)
    help_text = 'Dropout rate.'
    parser.add_argument('--dropout', type=float, default=0.1, help=help_text)
    help_text = 'Number of epochs.'
    parser.add_argument('--epochs', type=int, default=100, help=help_text)
    help_text = 'Batch size.'
    parser.add_argument('--batch_size', type=int, default=64, help=help_text)
    args = parser.parse_args()

    with open('data/preprocessed/vocab.json', 'r') as f:
        vocab = json.load(f)
    vocab_size = vocab['vocab_size']
    
    with open('data/preprocessed/preprocessed_data.json', 'r') as f:
        tokenized_text = json.load(f)['tokenized_text']
    
    model = GPT(
        num_blocks = args.num_blocks,
        num_attention_heads = args.num_attention_heads,
        context_size = args.context_size,
        attention_dim = args.attention_dim,
        feed_forward_dim = args.feed_forward_dim,
        activation = args.activation,
        token_embed_dim = args.token_embed_dim,
        dropout = args.dropout,
        vocab_size = vocab_size
    )

    num_contexts = int(len(tokenized_text) / args.context_size)
    num_tokens = num_contexts * args.context_size
    print('Context size:', args.context_size)
    print('Num contexts:', num_contexts)
    print('Num tokens:', num_tokens)
    
    print('Loading tokens... ', end='', flush=True)
    input_tokens = np.array(
        tokenized_text[:num_tokens]
    ).reshape(num_contexts, args.context_size)
    print('DONE')
    print('Inputs shape:', input_tokens.shape)
    
    print('Converting tokens to one-hot labels... ', end='', flush=True)
    one_hot_labels = np.array(
        tk.tokens_to_labels(tokenized_text, vocab_size)[:num_tokens]
    ).reshape(num_contexts, args.context_size, vocab_size)
    print('DONE')
    print('Labels shape:', one_hot_labels.shape)

    model.compile(
        optimizer = 'adam', 
        loss = 'categorical_crossentropy', 
        metrics=['accuracy']
    )

    model.fit(
        x = input_tokens, 
        y = one_hot_labels, 
        epochs = args.epochs, 
        batch_size = args.batch_size,
        verbose = 1
    )
    
    output_dir = 'data/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.network.save(os.path.join(output_dir, 'gpt'))

if __name__ == '__main__:
    main()
