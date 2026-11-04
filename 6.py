
import threading

class TelemetryManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def log(self, msg):
        print("Telemetry:", msg)

class WebDriver:
    def start(self):
        print("Web Driver Started")

class DesktopDriver:
    def start(self):
        print("Desktop Driver Started")

class MobileDriver:
    def start(self):
        print("Mobile Driver Started")

class DriverFactory:
    @staticmethod
    def create_driver(name):
        drivers = {
            "Web": WebDriver,
            "Desktop": DesktopDriver,
            "Mobile": MobileDriver
        }
        return drivers[name]()

telemetry = TelemetryManager()
telemetry.log("Application Started")

for name in ["Web", "Desktop", "Mobile"]:
    DriverFactory.create_driver(name).start()

telemetry.log("All Drivers Created")
