import numpy as np
from pathlib import Path
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

lines = Path('DataSet/Exp8/eng_tamil_pairs.txt').read_text(
    encoding='utf-8').splitlines()
pairs = list(dict.fromkeys(tuple(s.split('\t', 1)) for s in lines
                           if '\t' in s))
eng = [a for a, b in pairs]
tam = ['<start> ' + b + ' <end>' for a, b in pairs]
et, tt = Tokenizer(filters=''), Tokenizer(filters='')
et.fit_on_texts(eng)
tt.fit_on_texts(tam)
e = et.texts_to_sequences(eng)
t = tt.texts_to_sequences(tam)
me, mt = max(map(len, e)), max(map(len, t))
e = pad_sequences(e, maxlen=me, padding='post')
t = pad_sequences(t, maxlen=mt, padding='post')
a = Input(shape=(me,))
b = Embedding(len(et.word_index)+1, 64)(a)
_, h, c = LSTM(128, return_state=True)(b)
d = Input(shape=(mt-1,))
f = Embedding(len(tt.word_index)+1, 64)(d)
f = LSTM(128, return_sequences=True)(f, initial_state=[h, c])
out = Dense(len(tt.word_index)+1, activation='softmax')(f)
model = Model([a, d], out)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit([e, t[:, :-1]], t[:, 1:], epochs=100, batch_size=32,
          sample_weight=(t[:, 1:] != 0).astype('float32'))

rev = {v: k for k, v in tt.word_index.items()}
start = tt.word_index['<start>']
end = tt.word_index['<end>']
def translate(sentence):
    seq = et.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=me, padding='post')
    target = [start]
    for _ in range(mt-1):
        a = pad_sequences([target], maxlen=mt-1, padding='post')
        p = model.predict([seq, a], verbose=0)
        w = int(p[0, len(target)-1].argmax())
        if w in (0, end):
            break
        target.append(w)
    return ' '.join(rev.get(i, '') for i in target[1:])

for s in ['Get down.', 'Good morning.', 'Thank you.', 'hospital']:
    print(s, '->', translate(s))
