import nltk
import numpy as np
from nltk.corpus import treebank
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('treebank', quiet=True)
pairs = [list(zip(*s)) for s in treebank.tagged_sents()]
words = [list(w) for w, t in pairs]
tags = [list(t) for w, t in pairs]
x, xt, y, yt = train_test_split(words, tags, test_size=0.2,
                               random_state=42)
wt = Tokenizer(oov_token='<UNK>')
tt = Tokenizer(lower=False, filters='')
wt.fit_on_texts(x)
tt.fit_on_texts(y)
maxlen = max(map(len, x))
def prep(seq, tok):
    ids = tok.texts_to_sequences(seq)
    return pad_sequences(ids, maxlen=maxlen, padding='post')

x1, x2 = prep(x, wt), prep(xt, wt)
y1, y2 = prep(y, tt), prep(yt, tt)
a = Input(shape=(maxlen,))
b = Embedding(len(wt.word_index)+1, 128, mask_zero=True)(a)
b, h, c = LSTM(128, return_sequences=True,
               return_state=True)(b)
b = LSTM(128, return_sequences=True)(b, initial_state=[h, c])
out = Dense(len(tt.word_index)+1, activation='softmax')(b)
model = Model(a, out)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x1, y1, epochs=5, batch_size=32, validation_split=0.1,
          sample_weight=(y1 != 0).astype('float32'))
print('Test loss, accuracy:', model.evaluate(x2, y2,
      sample_weight=(y2 != 0).astype('float32')))

i = min(782, len(xt)-1)
p = model.predict(x2[i:i+1], verbose=0)[0].argmax(axis=-1)
rev = {v: k for k, v in tt.word_index.items()}
print('Words:', xt[i])
print('Actual:', yt[i])
print('Predicted:', [rev.get(int(n), '?') for n in p[:len(xt[i])]])
