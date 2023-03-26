import tensorflow as tf
import numpy as np
from model import GPT
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

def get_labels(tokens, token_dim):
    labels = []
    for i in range(len(tokens)):
        labels.append(tf.one_hot(indices = [i], depth = token_dim))

    return labels

if __name__ == '__main__':
    print(tf.config.list_physical_devices())

    with open('../data/tiny_shakespeare.txt', 'r', encoding='utf8') as f:
        text = f.read()

    #tokenized_text, token_dim = spacy_tokenize(text)
    tokenized_text, token_dim = character_tokenize(text)

    model = GPT(
        num_blocks = 12,
        num_attention_heads = 12,
        context_size = 512,
        attention_dim = 768,
        feed_forward_dim = 3072,
        activation = 'gelu',
        token_dim = token_dim
    )

    num_contexts = int(len(tokenized_text) / model.context_size)
    num_tokens = num_contexts * model.context_size

    input_tokens = np.array(
        tokenized_text[:num_tokens]
    ).reshape(num_contexts, model.context_size)
    print("input tokens shape:", input_tokens.shape)

    one_hot_labels = np.array(
        get_labels(tokenized_text, token_dim)[:num_tokens]
    ).reshape(num_contexts, 512, token_dim)
    print("labels shape:", one_hot_labels.shape)

    model.compile(
        optimizer = 'adam', 
        loss = 'categorical_crossentropy', 
        metrics=['accuracy']
    )

    model.fit(
        x = input_tokens, 
        y = one_hot_labels, 
        epochs = 1, 
        batch_size = 1
    )