from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sentences = [
    "natural language processing is interesting",
    "machine learning is a part of artificial intelligence",
    "deep learning improves machine learning",
    "natural language processing uses machine learning",
    "artificial intelligence includes machine learning",
    "cats and dogs are pets",
    "dog is a faithful animal",
    "cat is a cute animal"
]

corpus = [sentence.split() for sentence in sentences]

cbow = Word2Vec(
    corpus, vector_size=50, window=3, min_count=1, sg=0
)
skipgram = Word2Vec(
    corpus, vector_size=50, window=3, min_count=1, sg=1
)

print("CBOW:", cbow.wv.most_similar("machine"))
print("Skip-Gram:", skipgram.wv.most_similar("machine"))

words = skipgram.wv.index_to_key
points = PCA(n_components=2).fit_transform(skipgram.wv[words])

plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1])

for word, point in zip(words, points):
    plt.annotate(word, point)

plt.title("Skip-Gram Word Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
