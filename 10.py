import nltk
from collections import defaultdict, Counter

nltk.download("punkt_tab", quiet=True)

corpus = """
Natural Language Processing is an important field of Artificial Intelligence.
Natural Language Processing enables computers to understand human language.
Artificial Intelligence and Machine Learning are transforming technology.
Machine Learning is a subset of Artificial Intelligence.
"""

tokens = nltk.word_tokenize(corpus.lower())
successors = defaultdict(list)

for first, second in nltk.bigrams(tokens):
    successors[first].append(second)

word = input("Enter a word: ").strip().lower()

if word in successors:
    next_words = successors[word]
    print("Possible words:", next_words)
    print("Prediction:", Counter(next_words).most_common(1)[0][0])
else:
    print("Word not found in corpus.")
