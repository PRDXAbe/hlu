import unittest
from unittest.mock import patch

def get_weather():
    return "Sunny"

def decision():
    weather = get_weather()
    if weather == "Sunny":
        return "Go Outside"
    if weather == "Rainy":
        return "Stay Home"
    return "Unknown Decision"

class TestDecision(unittest.TestCase):
    @patch("__main__.get_weather", return_value="Sunny")
    def test_sunny(self, mock):
        self.assertEqual(decision(), "Go Outside")

    @patch("__main__.get_weather", return_value="Rainy")
    def test_rainy(self, mock):
        self.assertEqual(decision(), "Stay Home")

    @patch("__main__.get_weather", return_value="Cloudy")
    def test_unknown(self, mock):
        self.assertEqual(decision(), "Unknown Decision")

unittest.main(argv=[""], exit=False)
