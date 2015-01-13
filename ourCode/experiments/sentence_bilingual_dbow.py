#!/usr/bin/python
from gensim.models.doc2vec import Doc2Vec, LabeledLineSentence, LabeledSentence
from itertools import izip, islice
import string
import sys

def preprocess(s):
    return s.rstrip().lower().translate(string.maketrans("",""), string.punctuation)

class BitextMergedSentence(object):
    def __init__(self, filename, size):
        self.filename = filename
        self.size = size
 
    def __iter__(self):
        f = self.filename
        with open(f + 'en') as ens, open(f + 'de') as des: 
            for i, (en, de) in enumerate(islice(izip(ens, des), self.size)):
                pen = preprocess(en)
                en = ['%s_en'%w for w in pen.split()]
                de = ['%s_de'%w for w in preprocess(de).split()]
                yield LabeledSentence(words=en+de, labels=[pen])

f = sys.argv[1]+'/europarl-v7.de-en.'
n = 50000
sentences = BitextMergedSentence(f, n)
print '%s sentences' % n

model = Doc2Vec(dm=0, alpha=0.025, min_alpha=0.025, size=256)
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