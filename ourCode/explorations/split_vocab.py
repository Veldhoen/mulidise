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
n = 50000
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

## Build two vocabularies & initialize ##
print 'Building two vocabularies'

sentences_en = (LabeledSentence(words=en, labels=[pen]) for pen, en, de in BitextTriples(f, n))
sentences_de = (LabeledSentence(words=de, labels=[pen]) for pen, en, de in BitextTriples(f, n))

model_en = Doc2Vec(alpha=0.015, min_alpha=0.015, size=256, min_count=0)
model_en.build_vocab(sentences_en)
model_en.train_lbls = False
model_de = Doc2Vec(alpha=0.015, min_alpha=0.015, size=256, min_count=0)
model_de.build_vocab(sentences_de)
model_de.train_lbls = False
print 'Running time %ss' % (timeit.default_timer() - start)

# Initializing
print 'Initializing two word spaces'
    
for pen, en, de in BitextTriples(f, n):
    # set sentence vector
    if pen in model:
        model_en.syn0[model_en.vocab[pen].index] = model[pen]
        model_de.syn0[model_de.vocab[pen].index] = model[pen]
        # set words
        for w in en:
            model_en.syn0[model_en.vocab[w].index] += model[pen]
        for w in de:
            model_de.syn0[model_de.vocab[w].index] += model[pen]
for w in model_en.vocab:
    model_en.syn0[model_en.vocab[w].index] /= (model_en.vocab[w].count+1.0)
    if w in model:
        model.syn0[model.vocab[w].index] = model_en.syn0[model_en.vocab[w].index]
for w in model_de.vocab:
    model_de.syn0[model_de.vocab[w].index] /= (model_de.vocab[w].count+1.0)
    if w in model:
        model.syn0[model.vocab[w].index] = model_de.syn0[model_de.vocab[w].index]

model.syn0norm = (model.syn0 / sqrt((model.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)
model_en.syn0norm = (model_en.syn0 / sqrt((model_en.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)
model_de.syn0norm = (model_de.syn0 / sqrt((model_de.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)


print 'Running time %ss' % (timeit.default_timer() - start)

inspect_words(model)
inspect_words(model_en)
inspect_words(model_de)


# Train two vocabularies
print 'epochs'
for epoch in range(10):
    sentences_en = (LabeledSentence(words=en, labels=[pen]) for pen, en, de in BitextTriples(f, n))
    sentences_de = (LabeledSentence(words=de, labels=[pen]) for pen, en, de in BitextTriples(f, n))
    model_en.train(sentences_en)
    model_de.train(sentences_de)
    print epoch
    model_en.alpha -= 0.001  # decrease the learning rate
    model_de.alpha -= 0.001  # decrease the learning rate
print 'Running time %ss' % (timeit.default_timer() - start)

# combine
print 'Combining models'
for w in model_en.vocab:
    if w in model:
        model.syn0[model.vocab[w].index] = model_en.syn0[model_en.vocab[w].index]
for w in model_de.vocab:
    if w in model:
        model.syn0[model.vocab[w].index] = model_de.syn0[model_de.vocab[w].index]
model.syn0norm = (model.syn0 / sqrt((model.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)

inspect_words(model)


