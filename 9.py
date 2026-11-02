from deep_translator import GoogleTranslator

text = input("Enter text: ")
source = input("Source language code: ")
target = input("Target language code: ")

translator = GoogleTranslator(source=source, target=target)
print("Translation:", translator.translate(text))
