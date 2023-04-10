import tensorflow as tf
import spacy

# Spacy tokenization.
def spacy_tokenize(text):
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

# Character-wise tokenization.
def character_tokenize(text):
    characters = set(text)
    character_map = {}
    for character in characters:
        character_map[character] = len(character_map)

    token_ids = []
    for character in text:
        token_ids.append(character_map[character])

    return token_ids, len(character_map)

def tokens_to_labels(tokens, vocab_size):
    labels = []
    for i in range(len(tokens)):
        labels.append(tf.one_hot(indices = [i], depth = vocab_size))

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