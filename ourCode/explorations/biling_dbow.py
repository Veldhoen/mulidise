#!/usr/bin/python
from gensim.models.doc2vec import Doc2Vec, LabeledLineSentence, LabeledSentence
from itertools import izip, islice
import sys
import timeit
from inspection import preprocess, inspect_words, inspect_sentences
from numpy import sqrt, newaxis, float32 as REAL

class BitextTriples(object):
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
                yield pen, en, de

## Train sentences ##
print 'Learning sentence embeddings'
start = timeit.default_timer()

f = sys.argv[1]+'/europarl-v7.de-en.'
n = 500000
sentences = (LabeledSentence(words=en+de, labels=[pen]) for pen, en, de in BitextTriples(f, n))
print '%s sentences' % n

model = Doc2Vec(dm=0, alpha=0.025, min_alpha=0.025, size=256)
model.build_vocab(sentences)
print '%s words in vocab' % (len(model.vocab) - n)

print 'epochs'
for epoch in range(10):
    sentences = (LabeledSentence(words=en+de, labels=[pen]) for pen, en, de in BitextTriples(f, n))
    model.train(sentences)
    print epoch
    model.alpha -= 0.002  # decrease the learning rate
print 'Running time %ss' % (timeit.default_timer() - start)

inspect_sentences(model)

for pen, en, de in BitextTriples(f, n):
    # set sentence vector
    if pen in model:
        # set words
        for w in en+de:
            if w in model:
                model.syn0[model.vocab[w].index] += model[pen]
for w in model.vocab:
    model.syn0[model.vocab[w].index] /= (model.vocab[w].count+1.0)
model.syn0norm = (model.syn0 / sqrt((model.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)

with open(sys.argv[2], 'a') as out:
    for w in model.vocab:
        if '_' in w:
            out.write('%s : %s\n'%(w, ' '.join((str(d) for d in model[w]))))