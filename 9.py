
class Observer:
    def update(self, score):
        pass

class AdminAlert(Observer):
    def update(self, score):
        print("Admin Alert: Anomaly Score =", score)

class FirewallAlert(Observer):
    def update(self, score):
        print("Firewall: Blocking Traffic")

class LoggingService(Observer):
    def update(self, score):
        print("Logging: Incident Recorded")

class NetworkMonitor:
    def __init__(self):
        self.observers = []
        self.threshold = 70

    def attach(self, observer):
        self.observers.append(observer)

    def detect(self, score):
        if score >= self.threshold:
            for o in self.observers:
                o.update(score)
        else:
            print("Network Traffic Normal")

monitor = NetworkMonitor()

for o in [AdminAlert(), FirewallAlert(), LoggingService()]:
    monitor.attach(o)

monitor.detect(int(input("Enter Anomaly Score: ")))
