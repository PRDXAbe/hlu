from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

messages = [
    "Congratulations! You have won a free lottery ticket",
    "Claim your cash prize now",
    "Limited time offer, click here",
    "Meeting scheduled at 10 AM tomorrow",
    "Please submit your assignment today",
    "Let's have lunch together",
    "You won a free vacation package",
    "Your bank account has been credited"
]
labels = ["Spam", "Spam", "Spam", "Ham", "Ham", "Ham", "Spam", "Ham"]

X_train, X_test, y_train, y_test = train_test_split(
    messages, labels, test_size=0.25, random_state=42,
    stratify=labels
)

model = make_pipeline(
    TfidfVectorizer(stop_words="english"),
    MultinomialNB()
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(classification_report(y_test, predictions, zero_division=0))

message = input("Enter a message: ")
print("Prediction:", model.predict([message])[0])
