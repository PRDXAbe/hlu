
class TextPreprocessor:
    def process(self, text):
        return text.lower().strip()

class IntentDetector:
    def detect(self, text):
        if "refund" in text:
            return "Refund Request"
        if "payment" in text:
            return "Billing Issue"
        if "login" in text:
            return "Technical Support"
        return "General Inquiry"

class SentimentAnalyzer:
    def analyze(self, text):
        words = ["bad", "angry", "issue", "problem", "not working"]
        return "Negative" if any(w in text for w in words) else "Neutral"

class TicketRouter:
    def route(self, intent):
        routes = {
            "Refund Request": "Finance Team",
            "Billing Issue": "Accounts Team",
            "Technical Support": "IT Support Team"
        }
        return routes.get(intent, "Customer Service Team")

class CustomerSupportFacade:
    def __init__(self):
        self.pre = TextPreprocessor()
        self.det = IntentDetector()
        self.sent = SentimentAnalyzer()
        self.route = TicketRouter()

    def handle(self, query):
        text = self.pre.process(query)
        intent = self.det.detect(text)
        print("Intent:", intent)
        print("Sentiment:", self.sent.analyze(text))
        print("Department:", self.route.route(intent))

CustomerSupportFacade().handle(input("Enter Query: "))
