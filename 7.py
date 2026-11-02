from textblob import TextBlob

text = input("Enter text: ")
polarity, subjectivity = TextBlob(text).sentiment

if polarity > 0:
    sentiment = "POSITIVE"
elif polarity < 0:
    sentiment = "NEGATIVE"
else:
    sentiment = "NEUTRAL"

print("Polarity:", polarity)
print("Subjectivity:", subjectivity)
print("Sentiment:", sentiment)
print("Subjective" if subjectivity > 0.5 else "Relatively objective")
