#!/usr/bin/python
from gensim.models.doc2vec import Doc2Vec, LabeledLineSentence, LabeledSentence
from itertools import izip, islice
import string
import sys
import timeit
from inspection import preprocess, inspect_words, inspect_sentences

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

print 'Learning sentence embeddings'
start = timeit.default_timer()

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
stop = timeit.default_timer()
print 'Running time %ss' % (stop - start)

inspect_sentences(model)


print 'Learning word embeddings'
# # scale the sentence vectors
# for s in sentences:
#     for l in s.labels:
#         if l in model.vocab:
#             model.syn0[model.vocab[l].index] *= 5

model.sg = 0 # switch over to Distributed Memory
model.train_lbls = False # stop training sentences
model.alpha = 0.025

print 'epochs'
for epoch in range(10):
    model.train(sentences)
    print epoch
    model.alpha -= 0.002  # decrease the learning rate
stop = timeit.default_timer()
print 'Running time %ss' % (stop - start)

inspect_words(model)