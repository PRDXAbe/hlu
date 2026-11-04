
class SmartWarehouseRobot:
    def __init__(self):
        self.name = "Smart Warehouse Sorting Robot"

    def functional(self):
        return [
            "Scan Package", "Read Barcode",
            "Find Destination", "Sort Package",
            "Update Inventory", "Delivery Status"
        ]

    def non_functional(self):
        return {
            "Performance": "100 packages/min",
            "Reliability": "99.9%",
            "Safety": "Emergency Stop",
            "Scalability": "Multiple Robots",
            "Availability": "24x7"
        }

    def failures(self):
        return {
            "Sensor": ("Sensor Failure", "Restart Sensor"),
            "Camera": ("Capture Failure", "Retry Capture"),
            "Barcode": ("Not Detected", "Manual Check"),
            "Sorting": ("Mechanical Jam", "Stop Conveyor"),
            "Database": ("Connection Lost", "Reconnect"),
            "Network": ("Communication Failure", "Retry")
        }

robot = SmartWarehouseRobot()
print(robot.name)
print("Functional:", robot.functional())
print("Non-Functional:", robot.non_functional())
print("Failures and Mitigation:", robot.failures())
