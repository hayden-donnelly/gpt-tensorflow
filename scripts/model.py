import tensorflow as tf
import keras
from keras import layers
import numpy as np

class GPT(keras.Model):
    def __init__(
        self, 
        num_blocks, 
        token_dim, 
        num_attention_heads, 
        attention_dim, 
        feed_forward_dim, 
        context_size, 
        activation
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_dim = token_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dim = attention_dim
        self.feed_forward_dim = feed_forward_dim
        self.context_size = context_size
        self.activation = activation
        self.attention_mask = np.tril(np.ones((context_size, context_size)), 0)
        self.network = self.create_network()

    def train_step(self, data):
        with tf.GradientTape() as tape:
            inputs = data[0]
            labels = data[1]
            probs = self.network(inputs, training = True)

            loss = self.compiled_loss(labels, probs)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {m.name: m.result() for m in self.metrics}

    def create_network(self):
        inputs = layers.Input((self.context_size))
        token_embed = layers.Embedding(
            input_dim = self.token_dim, 
            output_dim = self.token_dim
        )(inputs)

        positional_embed = layers.Embedding(
            input_dim = self.context_size, 
            output_dim = self.token_dim
        )(np.arange(self.context_size))

        x = token_embed + positional_embed

        for _ in range(self.num_blocks):
            att = layers.MultiHeadAttention(
                num_heads = self.num_attention_heads,
                key_dim = self.attention_dim, 
                value_dim = self.attention_dim,
            )(x, x, attention_mask = self.attention_mask)
            
            x = x + att
            x = layers.LayerNormalization()(x)

            ff = layers.Dense(
                units = self.feed_forward_dim, 
                activation = self.activation
            )(x)
            
            ff = layers.Dense(
                units = self.token_dim, 
                activation = self.activation
            )(ff)
            
            x = x + ff
            x = layers.LayerNormalization()(x)
        
        x = layers.Dense(
            units = self.token_dim, 
            activation = 'softmax'
        )(x)

        return keras.Model(inputs = inputs, outputs = x)
        