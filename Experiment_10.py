import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, layers as L

path = Path('DataSet/Exp10/sales.csv')
df = pd.read_csv(path)
cols = ['customer_id', 'product_id', 'product_name',
        'product_category', 'quantity', 'order_status']
df = df.dropna(subset=cols).copy()
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
df['order_status'] = df['order_status'].str.strip().str.lower()
df = df[(df.order_status == 'completed') & (df.quantity > 0)]
items = df.drop_duplicates('product_id').set_index('product_id')
ids = df[['customer_id', 'product_id']].drop_duplicates().copy()
users = {v: i for i, v in enumerate(ids.customer_id.unique())}
prods = {v: i for i, v in enumerate(ids.product_id.unique())}
rev = {v: k for k, v in prods.items()}
ids['u'] = ids.customer_id.map(users)
ids['p'] = ids.product_id.map(prods)
ids['label'] = 1.0
bought = ids.groupby('u')['p'].apply(set).to_dict()
rng = np.random.default_rng(42)
neg = []
for u, seen in bought.items():
    rest = list(set(range(len(prods))) - seen)
    for p in rng.choice(rest, min(4, len(rest)), replace=False):
        neg.append((u, p, 0.0))
data = pd.concat([
    ids[['u', 'p', 'label']],
    pd.DataFrame(neg, columns=['u', 'p', 'label'])
], ignore_index=True)
train, test = train_test_split(data, test_size=0.2,
                               random_state=42, stratify=data.label)
def inputs(frame):
    return [frame.u.to_numpy(), frame.p.to_numpy()]

ui = L.Input(shape=(1,), dtype='int32')
pi = L.Input(shape=(1,), dtype='int32')
u = L.Flatten()(L.Embedding(len(users), 32)(ui))
p = L.Flatten()(L.Embedding(len(prods), 32)(pi))
z = L.Concatenate()([u, p])
z = L.Dense(64, activation='relu')(z)
z = L.Dropout(0.2)(z)
z = L.Dense(32, activation='relu')(z)
out = L.Dense(1, activation='sigmoid')(z)
model = Model([ui, pi], out)
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(name='auc')])
model.fit(inputs(train), train.label.to_numpy(), epochs=30,
          batch_size=64, validation_data=(inputs(test), test.label.to_numpy()))

def recommend(customer, top=5):
    u = users[customer]
    choices = list(set(range(len(prods))) - bought.get(u, set()))
    if not choices:
        return pd.DataFrame()
    a = np.full(len(choices), u, dtype='int32')
    b = np.array(choices, dtype='int32')
    scores = model.predict([a, b], verbose=0).ravel()
    rows = []
    for j in np.argsort(scores)[::-1][:top]:
        pid = rev[choices[j]]
        rows.append((pid, items.loc[pid, 'product_name'],
                     items.loc[pid, 'product_category'], float(scores[j])))
    return pd.DataFrame(rows, columns=[
        'product_id', 'product_name', 'category', 'score'])

customer = next(iter(users))
print(recommend(customer))
