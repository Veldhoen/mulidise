#!/usr/bin/python
from gensim.models.word2vec import *
import sys
import os
from itertools import chain, islice

def file_filter(root, filt):
	for (dirpath, dirnames, filenames) in os.walk(root):
		filt = os.path.join(root, filt)
		if filt in dirpath:
			for f in filenames:
				if f.endswith('.ted'):
					yield os.path.join(dirpath, f)

print 'Training from TED corpus (%s), saving to %s' % (sys.argv[1], sys.argv[2])

corpus = (LineSentence(f) for f in file_filter(sys.argv[1], ''))
sentences = chain.from_iterable(corpus)
sentences = list(sentences)

print len(sentences), 'sentences'

model = Word2Vec(sentences, size=300, window=5, min_count=5)

for e in range(20):
	model.train(sentences)

print len(model.vocab), 'words'

model.save_word2vec_format(sys.argv[2], fvocab=None, binary=False)

for w in ['queen_en', 'but_en', 'maybe_en', 'i_en', 'some_en']:
	print w, model.most_similar(w)