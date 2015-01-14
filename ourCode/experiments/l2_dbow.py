#!/usr/bin/python
from gensim.models.doc2vec import Doc2Vec, LabeledLineSentence, LabeledSentence
from itertools import izip, islice
from random import shuffle
import timeit
import sys
from inspection import preprocess, inspect_words

class BitextWordLabelsSentence(object):
    def __init__(self, filename, size):
        self.filename = filename
        self.size = size
 
    def __iter__(self):
        f = self.filename
        with open(f + 'en') as ens, open(f + 'de') as des: 
            for i, (en, de) in enumerate(islice(izip(ens, des), self.size)):
                en = ['%s_en'%w for w in preprocess(en).split()]
                de = ['%s_de'%w for w in preprocess(de).split()]
                langs = [en, de]
                shuffle(langs)
                l1, l2 = langs
                yield LabeledSentence(words=l1, labels=l2)

print 'Learning to predict all l2 words from every l1 word'
start = timeit.default_timer()

f = sys.argv[1]+'/europarl-v7.de-en.'
n = 50000
sentences = BitextWordLabelsSentence(f, n)
print '%s sentences' % n

model = Doc2Vec(dm=0, alpha=0.025, min_alpha=0.025, size=256)
model.build_vocab(sentences)
print '%s words in vocab' % len(model.vocab)

print 'epochs'
for epoch in range(10):
    model.train(sentences)
    print epoch
    model.alpha -= 0.002  # decrease the learning rate
stop = timeit.default_timer()
print 'Running time %ss' % (stop - start)

inspect_words(model)