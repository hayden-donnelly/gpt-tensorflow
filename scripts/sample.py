import tensorflow as tf
import tokenizer as tk
import keras
import json

if __name__ == "__main__":
    with open('../data/preprocessed/vocab.json', 'r') as f:
        vocab = json.load(f)
        decoding_map = vocab['decoding_map']
        encoding_map = vocab['encoding_map']
    
    gpt = keras.models.load_model('../data/output/gpt.h5')
    #text = "Hello"

    #tokenized_input = tk.tokenize_from_map(text, encoding_map)
    #pred = gpt(tokenized_input)
    #token = tk.labels_to_token(pred)[0]
    #text += tk.tokens_to_string(token, decoding_map)
    
    #print(tokenized_input)