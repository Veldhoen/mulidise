#!/usr/bin/python
from gensim.models.doc2vec import Doc2Vec, LabeledLineSentence, LabeledSentence
from itertools import izip, islice
from random import shuffle
import string
import csv

def preprocess(s):
	return s.rstrip().lower().translate(string.maketrans("",""), string.punctuation)

class BitextDoubleSentence(object):
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
                for l in langs:
                    yield LabeledSentence(words=l, labels=['SENT_%s'%i])

print 'Simply training German and English words with the bitext sentence id as label'

f = '/Users/benno/Documents/ Misc/data/de-en/europarl-v7.de-en.'
n = 50000
sentences = BitextDoubleSentence(f, n)
print '%s sentences' % n

model = Doc2Vec(alpha=0.025, min_alpha=0.025)
model.build_vocab(sentences)
print '%s words in vocab' % (len(model.vocab) - n)

print 'epochs'
for epoch in range(10):
    model.train(sentences)
    print epoch
    model.alpha -= 0.002  # decrease the learning rate

ws = ['mrs_en', 'april_en', 'objection_en', 'debate_en', 'answer_en']
ws += ['frau_de', 'april_de', 'einwand_de', 'debatte_de', 'antwort_de']
for w in ws:
    print '%s:\n' % w, model.most_similar(w, topn=10)


simtot = 0.0
for row in csv.DictReader(open('dictionary.csv')):
    sim = model.similarity(*['%s_%s'%(a,b) for (b,a) in row.items()])
    print row, sim
    simtot += sim
print 'total:', simtot