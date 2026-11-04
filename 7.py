
import json
import csv
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

with open("emp.json", "w") as f:
    json.dump({"employees": [
        {"id": 101, "name": "Alice"},
        {"id": 102, "name": "Bob"}
    ]}, f)

root = ET.Element("employees")
for i, name in [(201, "Charlie"), (202, "David")]:
    e = ET.SubElement(root, "employee")
    ET.SubElement(e, "id").text = str(i)
    ET.SubElement(e, "name").text = name
ET.ElementTree(root).write("emp.xml")

with open("emp.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerows([["id", "name"], [301, "Eve"], [302, "Frank"]])

class DataSource(ABC):
    @abstractmethod
    def get_data(self):
        pass

class JSONData:
    def read(self):
        with open("emp.json") as f:
            return json.load(f)["employees"]

class XMLData:
    def read(self):
        return [
            {"id": e.findtext("id"), "name": e.findtext("name")}
            for e in ET.parse("emp.xml").getroot()
        ]

class CSVData:
    def read(self):
        with open("emp.csv") as f:
            return list(csv.DictReader(f))

class Adapter(DataSource):
    def __init__(self, source):
        self.source = source

    def get_data(self):
        return self.source.read()

class Pipeline:
    def process(self, source):
        data = source.get_data()
        for item in data:
            print(item)
        print("Total:", len(data))

pipe = Pipeline()
for source in [JSONData(), XMLData(), CSVData()]:
    pipe.process(Adapter(source))
