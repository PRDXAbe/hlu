import nltk

for resource in ["punkt_tab", "averaged_perceptron_tagger_eng"]:
    nltk.download(resource, quiet=True)

sentence = "The little cat sat on the red mat"
tags = nltk.pos_tag(nltk.word_tokenize(sentence))

parser = nltk.RegexpParser(r"NP: {<DT>?<JJ>*<NN>}")
tree = parser.parse(tags)

print("POS tags:", tags)
print(tree)

for subtree in tree.subtrees(lambda t: t.label() == "NP"):
    print(" ".join(word for word, tag in subtree.leaves()))
