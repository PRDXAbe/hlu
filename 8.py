import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

df = pd.read_csv(input("CSV path: "))

if not {"text", "label"}.issubset(df.columns):
    raise ValueError("CSV must contain text and label columns.")

df = df.dropna(subset=["text", "label"])

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

model = make_pipeline(
    TfidfVectorizer(stop_words="english", max_df=0.7),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(classification_report(y_test, predictions, zero_division=0))

while True:
    news = input("Enter news or 'exit': ").strip()
    if news.lower() == "exit":
        break
    if news:
        print("Prediction:", model.predict([news])[0])
