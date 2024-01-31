import tensorflow as tf
import tokenizer as tk
import numpy as np
import keras
import json
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    help_text = 'The context size that the model was trained with.'
    parser.add_argument('--context_size', type=int, default=512, help=help_text)
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--prompt', type=str, required=True)
    args = parser.parse_args()

    with open('data/preprocessed/vocab.json', 'r') as f:
        vocab = json.load(f)
        decoding_map = vocab['decoding_map']
        encoding_map = vocab['encoding_map']
    vocab_size = len(encoding_map)

    model_path = 'data/output/gpt'
    assert os.path.exists(model_path), (
        'Could not find model. Try training a model with scripts/train.py before sampling.'
    )
    gpt = tf.keras.models.load_model(model_path)
    tokens = tk.tokenize_from_map(args.prompt, encoding_map)

    current_token_index = len(tokens)
    tokens_tensor = tf.reshape(tf.convert_to_tensor(tokens, dtype = tf.int64), (1, current_token_index))
    padding = tf.zeros((1, args.context_size - current_token_index), dtype = tf.int64)
    tokens_tensor = tf.concat((tokens_tensor, padding), axis = -1)
    tokens = tokens_tensor.numpy()

    for i in range(args.max_len):
        probs = gpt(tokens_tensor)[0, current_token_index, :].numpy()
        pred_token = np.random.choice(np.arange(0, vocab_size), p = probs)
        tokens[0, current_token_index] = pred_token
        tokens_tensor = tf.convert_to_tensor(tokens)
        decoded_token = tk.tokens_to_string([pred_token], decoding_map)
        if i == 0:
            print(args.prompt + decoded_token, end='', flush=True)
        else:
            print(decoded_token, end='', flush=True)
