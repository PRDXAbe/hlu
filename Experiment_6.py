import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

(x, y), (xt, yt) = imdb.load_data(num_words=10000)
x = pad_sequences(x, maxlen=200)
xt = pad_sequences(xt, maxlen=200)
model = Sequential([
    Input(shape=(200,)), Embedding(10000, 128),
    LSTM(64), Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x, y, epochs=3, batch_size=64)
print('Test loss, accuracy:', model.evaluate(xt, yt))

words = imdb.get_word_index()
sentence = 'This movie was super'
ids = [min(words.get(w.lower(), -1) + 3, 9999)
       if w.lower() in words else 2 for w in sentence.split()]
a = pad_sequences([ids], maxlen=200)
p = model.predict(a, verbose=0)[0, 0]
print(sentence, '->', 'Positive' if p >= 0.5 else 'Negative')
