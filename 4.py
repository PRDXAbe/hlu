import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

resources = [
    "punkt_tab",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker_tab",
    "words"
]

for resource in resources:
    nltk.download(resource, quiet=True)

text = "Barack Obama was born in Hawaii and worked at Microsoft."
tags = nltk.pos_tag(nltk.word_tokenize(text))
print(nltk.ne_chunk(tags))

documents = [
    "Natural Language Processing is interesting",
    "Machine Learning is a part of Artificial Intelligence",
    "Natural Language Processing uses Machine Learning"
]

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(documents)

print("Features:", vectorizer.get_feature_names_out())
print("TF-IDF:\n", matrix.toarray())
