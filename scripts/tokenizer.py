import tensorflow as tf
import spacy
import json

# Configurable parameters.
use_spacy = True
text_path = '../data/tiny_shakespeare.txt'

# Overrides parameters from command line.
exec(open('configurator.py').read())

# Spacy tokenization.
def spacy_tokenize(text):
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 2000000
    doc = nlp(text)

    token_ids = []
    encoding_map = {}
    decoding_map = {}
    for token in doc:
        if token.text not in encoding_map:
            encoding_map[token.text] = len(encoding_map)
            decoding_map[len(decoding_map)] = token.text
        token_id = encoding_map[token.text]
        token_ids.append(token_id)

    return token_ids, len(encoding_map), encoding_map, decoding_map

# Character-wise tokenization.
def character_tokenize(text):
    characters = set(text)
    encoding_map = {}
    decoding_map = {}
    for character in characters:
        encoding_map[character] = len(encoding_map)
        decoding_map[len(decoding_map)] = character

    token_ids = []
    for character in text:
        token_ids.append(encoding_map[character])

    return token_ids, len(encoding_map), encoding_map, decoding_map

def tokens_to_labels(tokens, vocab_size):
    labels = []
    for i in range(len(tokens)):
        labels.append(tf.one_hot(indices = [i], depth = vocab_size).numpy().tolist())

    return labels

def labels_to_tokens(labels):
    tokens = []
    for i in range(len(labels)):
        tokens.append(tf.argmax(labels[i]), axis = 1)
    return tokens

def tokens_to_string(tokens, vocab):
    string = ""
    for i in range(len(tokens)):
        string += vocab[tokens[i]]
    return string

if __name__ == "__main__":
    with open(text_path, 'r', encoding='utf8') as f:
        text = f.read()

    if use_spacy:
        tokenized_text, vocab_size, encoding_map, decoding_map = spacy_tokenize(text)
    else:
        tokenized_text, vocab_size, encoding_map, decoding_map = character_tokenize(text)

    vocab = {
        'vocab_size': vocab_size, 
        'encoding_map': encoding_map, 
        'decoding_map': decoding_map
    }

    with open('../data/preprocessed/vocab.json', 'w') as vocab_file:
        vocab_file.write(json.dumps(vocab))

    preprocessed_data = {
        'tokenized_text': tokenized_text,
        'labels': tokens_to_labels(tokenized_text, vocab_size)
    }

    with open('../data/preprocessed/preprocessed_data.json', 'w') as preprocessed_file:
        preprocessed_file.write(json.dumps(preprocessed_data))
