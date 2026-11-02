import nltk
from nltk.util import ngrams

nltk.download("punkt_tab", quiet=True)

text = "this is my cat this is my dog"
tokens = nltk.word_tokenize(text.lower())

for n in range(1, 4):
    print(f"{n}-grams:", list(ngrams(tokens, n)))
