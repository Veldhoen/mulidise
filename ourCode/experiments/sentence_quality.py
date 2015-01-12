#!/usr/bin/python
from gensim.models.doc2vec import Doc2Vec, LabeledLineSentence, LabeledSentence
from itertools import izip, islice
import string

def preprocess(s):
    return s.rstrip().lower().translate(string.maketrans("",""), string.punctuation)

class LiteralLineSentence(object):
    def __init__(self, filename, size):
        self.filename = filename
        self.size = size

    def __iter__(self):
        for uid, line in islice(enumerate(open(self.filename)), self.size):
            line = preprocess(line)
            yield LabeledSentence(words=line.split(), labels=[line])

f = '/Users/benno/Documents/ Misc/data/de-en/europarl-v7.de-en.en'
n = 50000
sentences = LiteralLineSentence(f, n)
print '%s sentences' % n

model = Doc2Vec(alpha=0.025, min_alpha=0.025, size=256)
model.build_vocab(sentences)
print '%s words in vocab' % (len(model.vocab) - n)

print 'epochs'
for epoch in range(10):
    model.train(sentences)
    print epoch
    model.alpha -= 0.002  # decrease the learning rate

lines = [
'you will be aware from the press and television that there have been a number of bomb explosions and killings in sri lanka'
,'but madam president my personal request has not been met'
,'we then put it to a vote'
,'thank you very much'
]

for l in lines:
    print '%s:\n' % l, model.most_similar(l, topn=10)