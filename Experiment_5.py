import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense

text = Path('DataSet/Exp5/shakespeare.txt').read_text(encoding='utf-8')
chars = sorted(set(text))
toi = {c: i for i, c in enumerate(chars)}
ids = np.array([toi[c] for c in text])
n = 100
x = np.array([ids[i:i+n] for i in range(len(ids)-n)])
y = np.array([ids[i+1:i+n+1] for i in range(len(ids)-n)])
data = tf.data.Dataset.from_tensor_slices((x, y))
data = data.shuffle(10000).batch(64, drop_remainder=True)
model = Sequential([
    Input(shape=(None,)), Embedding(len(chars), 128),
    SimpleRNN(256, return_sequences=True),
    Dense(len(chars))
])
model.compile(optimizer='adam', loss=tf.keras.losses.
              SparseCategoricalCrossentropy(from_logits=True))
model.fit(data, epochs=1)

def generate(seed, length=200):
    for _ in range(length):
        a = np.array([[toi[c] for c in seed[-n:]]])
        p = model.predict(a, verbose=0)[0, -1].argmax()
        seed += chars[p]
    return seed

print(generate('Shall '))
