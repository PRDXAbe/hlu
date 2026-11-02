from nltk.chat.util import Chat, reflections

pairs = [
    [
        r"(hi|hello|hey)\b.*",
        ["Hello! How can I help you?"]
    ],
    [
        r"what is your name\s*\??",
        ["My name is NLP Chatbot."]
    ],
    [
        r"how are you\s*\??",
        ["I am fine. Thank you!"]
    ],
    [
        r"what can you do\s*\??",
        ["I can answer simple questions and help you learn NLP."]
    ],
    [
        r".*\byour creator\b.*",
        ["I was created using Python and NLTK."]
    ],
    [
        r".*\bmachine learning\b.*",
        ["Machine Learning is a branch of Artificial Intelligence."]
    ],
    [
        r".*\bnlp\b.*",
        ["NLP stands for Natural Language Processing."]
    ],
    [
        r"(bye|goodbye)\b.*",
        ["Goodbye! Have a nice day."]
    ],
    [
        r".*",
        ["Could you please rephrase your question?"]
    ]
]

chatbot = Chat(pairs, reflections)
print("Type 'bye' or 'goodbye' to exit.")

while True:
    text = input("> ").strip()
    print(chatbot.respond(text))
    if text.lower() in {"bye", "goodbye"}:
        break
