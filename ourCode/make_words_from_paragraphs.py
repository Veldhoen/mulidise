#!/usr/bin/python
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import sys
import timeit
from parallel_bow import ParallelMergedLabeledLineSentence
from parawords import parawords

"""
	usage: <corpus-root> <n-langs> <m-langs> <export-file>

	open corpus (n langs), init vocab (n langs)
	train softmax & sentences on corpus (n langs)
	open corpus (m langs), init vocab (m langs)
	make parawords (m langs from n langs)
	export parawords (m langs)
"""

corpus_root, n_langs, m_langs, export_file, corpus_size = sys.argv[1:]
n_langs, m_langs = n_langs.split(','), m_langs.split(',')

corpus_size = int(corpus_size)
size=256
model_n = Doc2Vec(dm=0, alpha=0.025, min_alpha=0.025, size=size)
corpus_n = ParallelMergedLabeledLineSentence(corpus_root, n_langs, corpus_size)
model_n.build_vocab(corpus_n)
print '%s words in vocab_n' % (len(model_n.vocab) - corpus_size)

print 'epochs on corpus:'
start = timeit.default_timer()
for epoch in range(10):
    model_n.train(corpus_n)
    print epoch,
    model_n.alpha -= 0.002  # decrease the learning rate
print '(%ss)' % (timeit.default_timer() - start)

model_m = Doc2Vec(dm=0, alpha=0.025, min_alpha=0.025, size=size)
corpus_m = ParallelMergedLabeledLineSentence(corpus_root, m_langs, corpus_size)
model_m.build_vocab(corpus_m)
print '%s words in vocab_m' % (len(model_m.vocab) - corpus_size)


parawords(model_n, model_m, corpus_m, export_file)