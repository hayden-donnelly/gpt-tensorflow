import tensorflow as tf
import numpy as np
from model import GPT
import tokenizer as tk
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

# Overrides parameters from command line.
exec(open('configurator.py').read())

if __name__ == '__main__':
    print(tf.config.list_physical_devices())

    with open('../data/tiny_shakespeare.txt', 'r', encoding='utf8') as f:
        text = f.read()

    if use_spacy:
        tokenized_text, vocab_size, encoding_map, decoding_map = tk.spacy_tokenize(text)
    else:
        tokenized_text, vocab_size, encoding_map, decoding_map = tk.character_tokenize(text)

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