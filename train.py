"""Train separate LSTM next-word prediction models for English and Tamil datasets.

Generates n-gram sequence inputs and trains an Embedding + LSTM + Softmax architecture.
Saves model files (.keras), tokenizers (.json), and max sequence lengths (.json).
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train_language(language):
    dataset_path = os.path.join(BASE_DIR, "dataset", f"{language}.txt")
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist.")
        return

    with open(dataset_path, encoding="utf-8") as f:
        lines = [line.strip().lower() for line in f if line.strip()]

    if len(lines) < 3:
        raise ValueError(f"Add more sentences to {dataset_path}")

    # Standard filter stripping punctuation only (NOT stripping letters 't' or 'n'!)
    punctuation_filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    tokenizer = Tokenizer(oov_token="<oov>", filters=punctuation_filters)
    tokenizer.fit_on_texts(lines)
    
    total_words = len(tokenizer.word_index) + 1
    print(f"[{language.upper()}] Vocabulary size: {total_words} unique words")

    sequences = []
    for line in lines:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(2, len(token_list) + 1):
            sequences.append(token_list[:i])

    if not sequences:
        raise ValueError(f"Dataset for {language} needs longer sentences.")

    max_len = max(len(s) for s in sequences)
    padded = pad_sequences(sequences, maxlen=max_len, padding="pre")

    X, y = padded[:, :-1], padded[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    embedding_dim = 128
    lstm_units = 128

    model = Sequential([
        Embedding(total_words, embedding_dim, input_length=max_len - 1),
        LSTM(lstm_units, return_sequences=False),
        Dropout(0.2),
        Dense(total_words, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print(f"Training {language} model over {len(sequences)} sequence pairs for 60 epochs...")
    
    model.fit(X, y, epochs=60, batch_size=16, verbose=1)

    model_dir = os.path.join(BASE_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)

    model.save(os.path.join(model_dir, f"{language}_model.keras"))
    
    with open(os.path.join(model_dir, f"{language}_tokenizer.json"), "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    with open(os.path.join(model_dir, f"{language}_maxlen.json"), "w", encoding="utf-8") as f:
        json.dump({"max_sequence_len": max_len}, f)

    print(f"Successfully trained and saved {language} model!")

if __name__ == "__main__":
    for lang in ("english", "tamil"):
        train_language(lang)
