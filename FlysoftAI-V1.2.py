import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Baca dataset dari file
file_path = "dataset/Conversation.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    corpus = file.readlines()

# Tokenisasi
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = to_categorical(y, num_classes=total_words)

# Pembuatan Model
model = Sequential()
model.add(Embedding(total_words, 100))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.3))  # Tambahkan lapisan dropout
model.add(LSTM(128))
model.add(Dense(total_words, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1, batch_size=5)

# Fungsi untuk menyimpan model ke file
def save_model(model, tokenizer, model_path="saved_model", tokenizer_path="tokenizer.pickle"):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Simpan model
    model.save(os.path.join(model_path, "chatbot_model.h5"))

    # Simpan tokenizer
    with open(os.path.join(model_path, tokenizer_path), 'wb') as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

# Fungsi untuk menggabungkan data baru ke dalam dataset
def merge_new_data(corpus, new_data):
    return corpus + new_data

# Fungsi untuk melatih model dengan data baru
def train_model_with_new_data(model, tokenizer, new_data, epochs=10):
    # Tokenisasi data baru
    tokenizer.fit_on_texts(new_data)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in new_data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_length = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

    X, y = input_sequences[:,:-1], input_sequences[:,-1]
    y = to_categorical(y, num_classes=total_words)

    # Latih model dengan data baru
    model.fit(X, y, epochs=epochs, verbose=1, batch_size=5)

# Fungsi untuk menghasilkan respons dari model
def generate_response(input_text, max_length=50):
    generated_text = input_text
    for _ in range(max_length):
        input_seq = tokenizer.texts_to_sequences([input_text])[0]
        input_seq = pad_sequences([input_seq], maxlen=max_sequence_length-1, padding='pre')
        predicted_probs = model.predict(input_seq)[0]
        predicted_word_index = np.argmax(predicted_probs)
        predicted_word = tokenizer.index_word[predicted_word_index]

        input_text += " " + predicted_word
        generated_text += " " + predicted_word
        # Hentikan loop jika kata terakhir adalah tanda baca
        if predicted_word in [".","!","\n"]:
            break

    return generated_text

# Contoh penggunaan
while True:
    # Uji coba
    input_text = input("User >> ")
    response = generate_response(input_text)
    print(f"Input: {input_text}")
    print(f"Output: {response}")

    # Simpan model dan tokenizer jika diperlukan
    save_model(model, tokenizer)

    # Ambil respons model dan tambahkan data baru ke dalam dataset
    new_data = [response]  # Gantilah ini dengan data respons yang sebenarnya
    corpus = merge_new_data(corpus, new_data)
    train_model_with_new_data(model, tokenizer, new_data)
