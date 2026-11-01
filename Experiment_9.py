import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers as L

(x, y), (xt, yt) = tf.keras.datasets.fashion_mnist.load_data()
x = x[..., None].astype('float32')
xt = xt[..., None].astype('float32') / 255
real_data = (x - 127.5) / 127.5
x = x / 255

g = tf.keras.Sequential([
    L.Input(shape=(100,)), L.Dense(7*7*256, use_bias=False),
    L.BatchNormalization(), L.LeakyReLU(), L.Reshape((7, 7, 256)),
    L.Conv2DTranspose(128, 4, 2, padding='same', use_bias=False),
    L.BatchNormalization(), L.LeakyReLU(),
    L.Conv2DTranspose(64, 4, 2, padding='same', use_bias=False),
    L.BatchNormalization(), L.LeakyReLU(),
    L.Conv2D(1, 3, padding='same', activation='tanh')
])
d = tf.keras.Sequential([
    L.Input(shape=(28, 28, 1)),
    L.Conv2D(64, 4, 2, padding='same'), L.LeakyReLU(0.2),
    L.Dropout(0.3), L.Conv2D(128, 4, 2, padding='same'),
    L.LeakyReLU(0.2), L.Dropout(0.3),
    L.Flatten(), L.Dense(1, activation='sigmoid')
])
opt = lambda: tf.keras.optimizers.Adam(0.0002, 0.5)
d.compile(optimizer=opt(), loss='binary_crossentropy')
d.trainable = False
gan = tf.keras.Sequential([g, d])
gan.compile(optimizer=opt(), loss='binary_crossentropy')

for epoch in range(50):
    for i in range(0, len(real_data), 128):
        real = real_data[i:i+128]
        n = len(real)
        fake = g.predict(np.random.randn(n, 100), verbose=0)
        d.trainable = True
        d.train_on_batch(real, np.ones((n, 1)) * 0.9)
        d.train_on_batch(fake, np.zeros((n, 1)))
        d.trainable = False
        gan.train_on_batch(np.random.randn(n, 100), np.ones((n, 1)))

pics = (g.predict(np.random.randn(16, 100), verbose=0) + 1) / 2
for i, pic in enumerate(pics):
    plt.subplot(4, 4, i+1)
    plt.imshow(pic.squeeze(), cmap='gray')
    plt.axis('off')
plt.show()

fake = (g.predict(np.random.randn(5000, 100), verbose=0) + 1) / 2
def classifier():
    m = tf.keras.Sequential([
        L.Input(shape=(28, 28, 1)),
        L.Conv2D(32, 3, activation='relu'), L.MaxPooling2D(),
        L.Conv2D(64, 3, activation='relu'), L.MaxPooling2D(),
        L.Flatten(), L.Dense(64, activation='relu'),
        L.Dense(10, activation='softmax')
    ])
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m

m1 = classifier()
m1.fit(x, y, epochs=3, batch_size=128)
a1 = m1.evaluate(xt, yt, verbose=0)[1]
xx = np.concatenate([x, fake])
yy = np.concatenate([y, np.random.choice(y, len(fake))])
order = np.random.permutation(len(xx))
m2 = classifier()
m2.fit(xx[order], yy[order], epochs=3, batch_size=128)
a2 = m2.evaluate(xt, yt, verbose=0)[1]
print('Original accuracy:', a1, 'Augmented accuracy:', a2)
plt.bar(['Original', 'GAN Augmented'], [a1*100, a2*100])
plt.ylabel('Accuracy (%)')
plt.show()
