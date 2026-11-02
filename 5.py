from nltk import CFG
from nltk.parse import ChartParser

grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP
Det -> 'the'
N -> 'cat' | 'dog'
V -> 'chased' | 'saw'
""")

parser = ChartParser(grammar)
sentence = "the cat chased the dog"

for tree in parser.parse(sentence.split()):
    print(tree)
