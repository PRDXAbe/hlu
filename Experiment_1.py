import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img = cv2.imread('DataSet/Exp1/flower.jpeg', 0)
equ = cv2.equalizeHist(img)
h1 = cv2.calcHist([img], [0], None, [256], [0, 256])
h2 = cv2.calcHist([equ], [0], None, [256], [0, 256])

for i, pic in enumerate([img, equ]):
    plt.subplot(2, 2, i + 1)
    plt.imshow(pic, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, i + 3)
    plt.plot([h1, h2][i])
plt.show()

blur = cv2.GaussianBlur(img, (5, 5), 0)
edges = cv2.Canny(blur, 100, 200)
plt.imshow(edges, cmap='gray')
plt.show()

pic = cv2.cvtColor(cv2.imread('DataSet/Exp1/flower.jpeg'),
                   cv2.COLOR_BGR2RGB)
arr = np.expand_dims(pic, 0)
aug = ImageDataGenerator(rotation_range=40, zoom_range=0.2,
                         horizontal_flip=True).flow(arr, batch_size=1)
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(pic if i == 0 else next(aug)[0].astype('uint8'))
    plt.axis('off')
plt.show()

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
k = np.ones((5, 5), np.uint8)
er = cv2.erode(binary, k, iterations=1)
di = cv2.dilate(binary, k, iterations=1)
for i, pic in enumerate([img, binary, er, di]):
    plt.subplot(2, 2, i + 1)
    plt.imshow(pic, cmap='gray')
    plt.axis('off')
plt.show()
