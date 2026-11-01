import gzip
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

folder = Path('DataSet/Exp3')
def load(name):
    with gzip.open(folder / (name + '.gz'), 'rb') as f:
        return idx2numpy.convert_from_string(f.read())

x = load('emnist-balanced-train-images-idx3-ubyte')
y = load('emnist-balanced-train-labels-idx1-ubyte')
xt = load('emnist-balanced-test-images-idx3-ubyte')
yt = load('emnist-balanced-test-labels-idx1-ubyte')
x = x.reshape(-1, 28, 28, 1) / 255.0
xt = xt.reshape(-1, 28, 28, 1) / 255.0

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, 3, activation='relu'), MaxPooling2D(2),
    Conv2D(64, 3, activation='relu'), MaxPooling2D(2),
    Flatten(), Dense(128, activation='relu'),
    Dropout(0.5), Dense(47, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
h = model.fit(x, y, epochs=4, batch_size=64,
              validation_split=0.2)
print('Test loss, accuracy:', model.evaluate(xt, yt))
for i, key in enumerate(['accuracy', 'loss']):
    plt.subplot(1, 2, i + 1)
    plt.plot(h.history[key], label='Train')
    plt.plot(h.history['val_' + key], label='Validation')
    plt.legend()
plt.show()

mapping = {}
with open(folder / 'emnist-balanced-mapping.txt') as f:
    for line in f:
        a, b = line.split()
        mapping[int(a)] = chr(int(b))
i = 25
p = model.predict(xt[i:i + 1], verbose=0).argmax()
print('Actual:', mapping[int(yt[i])], 'Predicted:', mapping[p])
plt.imshow(xt[i].squeeze(), cmap='gray')
plt.show()
