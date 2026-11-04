from abc import ABC, abstractmethod
import random

class Sensor(ABC):
    @abstractmethod
    def read_data(self):
        pass

class TemperatureSensor(Sensor):
    def read_data(self):
        return f"Temperature: {random.randint(25, 40)} °C"

class HumiditySensor(Sensor):
    def read_data(self):
        return f"Humidity: {random.randint(40, 80)} %"

class MotionSensor(Sensor):
    def read_data(self):
        return f"Motion: {random.choice(['Detected', 'Not Detected'])}"

sensors = [TemperatureSensor(), HumiditySensor(), MotionSensor()]

for s in sensors:
    print(s.read_data())
