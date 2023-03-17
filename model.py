import tensorflow as tf
import keras
from keras import layers

class gpt(keras.Models):
    def __init__(self, num_blocks, vocab_size, num_attention_heads):
        super().__init__()
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads

    def train_step(self, data):
        pass

    def call(self, inputs):
        x = layers.Embedding(input_dim=self.vocab_size, output_dim=self.vocab_size)(inputs)
        
        for _ in range(self.num_blocks):
            atttention_out = layers.MultiHeadAttention(
                num_heads=self.num_attention_heads,
                key_dim=self.vocab_size, 
                value_dim=self.vocab_size
            )(x, x, attention_mask=None)

            x = layers.Add()([x, atttention_out])
            x = layers.LayerNormalization()(x)

            feed_forward_out = layers.Dense(self.vocab_size, activation='relu')(x)
            x = layers.Add()([x, feed_forward_out])
            x = layers.LayerNormalization()(x)

        return x
