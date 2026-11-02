import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

for resource in ["punkt_tab", "stopwords", "wordnet"]:
    nltk.download(resource, quiet=True)

text = (
    "Natural Language Processing is a branch of Artificial Intelligence. "
    "It enables computers to understand human language "
    "and process text efficiently."
)

words = nltk.word_tokenize(text)
stop_words = set(stopwords.words("english"))
filtered = [w for w in words if w.lower() not in stop_words]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("Sentences:", nltk.sent_tokenize(text))
print("Words:", words)
print("Filtered:", filtered)
print("Stems:", [stemmer.stem(w) for w in filtered])
print("Lemmas:", [lemmatizer.lemmatize(w) for w in filtered])
