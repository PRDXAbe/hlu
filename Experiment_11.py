import string
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, Add

base = Path('DataSet/Exp11')
imgs = base / 'Flickr8k_Dataset' / 'Flicker8k_Dataset'
txt = base / 'Flickr8k_text'
capfile = txt / 'Flickr8k.token.txt'
if not capfile.exists():
    capfile = txt / 'Flickr8k.token'

def split(file):
    return [s.strip() for s in (txt / file).read_text().splitlines()
            if (imgs / s.strip()).is_file()]

train = split('Flickr_8k.trainImages.txt')
valid = split('Flickr_8k.devImages.txt')
test = split('Flickr_8k.testImages.txt')
caps = {}
for line in capfile.read_text(encoding='utf-8').splitlines():
    if '\t' not in line:
        continue
    name, sentence = line.split('\t', 1)
    name = name.split('#')[0]
    sentence = sentence.lower().translate(
        str.maketrans('', '', string.punctuation))
    caps.setdefault(name, []).append('startseq ' + sentence + ' endseq')

cnn = InceptionV3(weights='imagenet')
cnn = Model(cnn.input, cnn.layers[-2].output)
features = {}
for name in dict.fromkeys(train + valid + test):
    im = Image.open(imgs / name).convert('RGB').resize((299, 299))
    im = np.array(im, dtype='float32')[None]
    features[name] = cnn.predict(preprocess_input(im), verbose=0)[0]

sentences = [s for name in train for s in caps[name]]
tok = Tokenizer()
tok.fit_on_texts(sentences)
vocab = len(tok.word_index) + 1
maxlen = max(len(s.split()) for s in sentences)

def sequences(names):
    a, b, y = [], [], []
    for name in names:
        for sentence in caps[name]:
            ids = tok.texts_to_sequences([sentence])[0]
            for j in range(1, len(ids)):
                a.append(features[name])
                b.append(pad_sequences([ids[:j]], maxlen=maxlen)[0])
                y.append(ids[j])
    return np.array(a), np.array(b), np.array(y)

x1, x2, y = sequences(train)
v1, v2, vy = sequences(valid)
a = Input(shape=(2048,))
b = Dropout(0.3)(a)
b = Dense(256, activation='relu')(b)
c = Input(shape=(maxlen,))
d = Embedding(vocab, 256, mask_zero=True)(c)
d = Dropout(0.3)(d)
d = LSTM(256)(d)
z = Add()([b, d])
z = Dense(256, activation='relu')(z)
out = Dense(vocab, activation='softmax')(z)
model = Model([a, c], out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
h = model.fit([x1, x2], y, epochs=5, batch_size=256,
              validation_data=([v1, v2], vy))
model.save('caption_model.keras')
plt.plot(h.history['loss'], label='Train')
plt.plot(h.history['val_loss'], label='Validation')
plt.legend()
plt.show()

def caption(name):
    result = 'startseq'
    for _ in range(maxlen):
        seq = tok.texts_to_sequences([result])
        seq = pad_sequences(seq, maxlen=maxlen)
        p = model.predict([features[name][None], seq], verbose=0)
        word = tok.index_word.get(int(p.argmax()))
        if not word or word == 'endseq':
            break
        result += ' ' + word
    return result.replace('startseq', '').strip()

name = test[0]
plt.imshow(Image.open(imgs / name))
plt.title(caption(name))
plt.axis('off')
plt.show()
