import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

faces = fetch_lfw_people(min_faces_per_person=70, resize=0.5)
x = faces.images[..., None] / 255.0
y = faces.target
x, xt, y, yt = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)
model = Sequential([
    Input(shape=x.shape[1:]),
    Conv2D(32, 3, activation='relu'), MaxPooling2D(2),
    Conv2D(64, 3, activation='relu'), MaxPooling2D(2),
    Flatten(), Dense(128, activation='relu'),
    Dense(len(faces.target_names), activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2)
print('Test loss, accuracy:', model.evaluate(xt, yt))
p = model.predict(xt[:1], verbose=0).argmax()
print('Actual:', faces.target_names[yt[0]])
print('Predicted:', faces.target_names[p])
plt.imshow(xt[0].squeeze(), cmap='gray')
plt.show()
