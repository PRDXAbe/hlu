import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Input

(x, y), (xt, yt) = tf.keras.datasets.mnist.load_data()
x, xt = x / 255.0, xt / 255.0
model = Sequential([
    Input(shape=(28, 28)), Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
h = model.fit(x, y, epochs=10, batch_size=32,
              validation_split=0.2)
print('Test loss, accuracy:', model.evaluate(xt, yt, verbose=0))
for i, key in enumerate(['loss', 'accuracy']):
    plt.subplot(1, 2, i + 1)
    plt.plot(h.history[key], label='Train')
    plt.plot(h.history['val_' + key], label='Validation')
    plt.title(key.title())
    plt.legend()
plt.show()
