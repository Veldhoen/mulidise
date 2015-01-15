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

corpus = (LineSentence(f) for f in file_filter(sys.argv[1], 'en-'))
sentences = islice(chain.from_iterable(corpus), None)

model = Word2Vec(sentences, size=40, window=5, min_count=5, workers=8)

model.train(sentences)

model.save_word2vec_format(sys.argv[2], fvocab=None, binary=False)