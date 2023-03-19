import tensorflow as tf
import keras
from keras import layers
import numpy as np
import spacy

class gpt(keras.Model):
    def __init__(self, num_blocks, token_dim, num_attention_heads, attention_dim, feed_forward_dim, context_size, activation):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_dim = token_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dim = attention_dim
        self.feed_forward_dim = feed_forward_dim
        self.context_size = context_size
        self.activation = activation
        self.attention_mask = None

    def call(self, inputs):
        token_embed = layers.Embedding(
            input_dim = self.token_dim, 
            output_dim = self.token_dim
        )(inputs)

        positional_embed = layers.Embedding(
            input_dim = self.context_size, 
            output_dim = self.token_dim
        )(np.arange(self.context_size))

        x = layers.Add()([token_embed, positional_embed])
        
        for _ in range(self.num_blocks):
            att = layers.MultiHeadAttention(
                num_heads = self.num_attention_heads,
                key_dim = self.attention_dim, 
                value_dim = self.attention_dim,
                output_shape = (self.token_dim)
            )(x, x, attention_mask = self.attention_mask)

            x = layers.Add()([x, att])
            x = layers.LayerNormalization()(x)

            ff = layers.Dense(
                units = self.feed_forward_dim, 
                activation = self.activation
            )(x)
            
            ff = layers.Dense(
                units = self.token_dim, 
                activation = self.activation
            )(ff)
            
            x = layers.Add()([x, ff])
            x = layers.LayerNormalization()(x)

        return x

def tokenize(text):
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 2000000
    doc = nlp(text)

    token_ids = []
    id_map = {}
    for token in doc:
        if token.text not in id_map:
            id_map[token.text] = len(id_map)
        token_id = id_map[token.text]
        token_ids.append(token_id)

    return token_ids, len(id_map)

def main():
    with open('tiny_shakespeare.txt', 'r', encoding='utf8') as f:
        text = f.read()

    tokenized_text, token_dim = tokenize(text)

    print(tokenized_text[:10])

    model = gpt(
        num_blocks = 12,
        num_attention_heads = 12,
        context_size = 512,
        attention_dim = 768,
        feed_forward_dim = 3072,
        activation = 'gelu',
        token_dim = token_dim
    )

if __name__ == '__main__':
    main()