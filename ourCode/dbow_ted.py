#!/usr/bin/python
from gensim.models.doc2vec import Doc2Vec, LabeledLineSentence, LabeledSentence
from itertools import izip, chain, islice
import sys
import timeit
import os
import numpy as np

def file_filter(root, filt):
	for (dirpath, dirnames, filenames) in os.walk(root):
		filt = os.path.join(root, filt)
		if filt in dirpath:
			for f in filenames:
				if f.endswith('.ted'):
					yield f, os.path.join(dirpath, f)

print 'Training from TED corpus (%s), saving to %s' % (sys.argv[1], sys.argv[2])
start = timeit.default_timer()

class BitextMergedSentence(object):
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2
 
    def __iter__(self):
        with open(self.l1) as ens, open(self.l2) as des: 
            for i, (en, de) in enumerate(izip(ens, des)):
                pen = en
                en = en.split()
                de = de.split()
                yield LabeledSentence(words=en+de, labels=[pen])

corpus_en = {name:f for name, f in file_filter(sys.argv[1], 'en-de')}
corpus_de = {name:f for name, f in file_filter(sys.argv[1], 'de-en')}
corpus = (BitextMergedSentence(corpus_en[n], corpus_de[n]) for n in corpus_en)
sentences = islice(chain.from_iterable(corpus), None)

size=256
model = Doc2Vec(dm=0, alpha=0.025, min_alpha=0.025, size=size)
model.build_vocab(sentences)
print '%s words in vocab' % len(model.vocab)

print 'epochs'
for epoch in range(10):
    model.train(sentences)
    print epoch
    model.alpha -= 0.002  # decrease the learning rate
stop = timeit.default_timer()
print 'Running time %ss' % (stop - start)


for name, f in corpus_en.iteritems():
    dataset, topic, posneg = f.split('/')[-4:-1]
    with open(f) as lines:
        vec = np.zeros(size)
        for line in lines:
            vec += model[line]
    out = os.path.join(sys.argv[2], '%s/%s' % (dataset, topic))
    vecstr = ' '.join(['%s:%s' % (i, d) for i,d in enumerate(vec)])
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))
    with open(out, 'a') as o:
        o.write('%s %s\n' % (1+int(posneg=='negative'), vecstr))