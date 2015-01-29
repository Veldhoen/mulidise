#!/usr/bin/python
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import sys
import timeit
from parallel_bow import ParallelMergedLabeledLineSentence
from ted_paragraph_docs import save_ted_docs, TedLabeledLineSentence
from itertools import chain

"""
	usage: <corpus-root> <langs> <ted-dir> <export-dir> <corpus-size>

	open corpus (1 or 2 langs) & ted, init vocab
	train softmax & sentences on corpus
	train sentences on ted
	export ted documents 
"""
corpus_root, langs, ted_dir, export_dir, corpus_size = sys.argv[1:]
langs = langs.split(',')

corpus_size = int(corpus_size)
size=256
model = Doc2Vec(dm=0, alpha=0.025, min_alpha=0.025, size=size)
corpus = ParallelMergedLabeledLineSentence(corpus_root, langs, corpus_size)
if len(langs)==1:
    ted = TedLabeledLineSentence(ted_dir)
else:
    ted = BiTedLabeledLineSentence(ted_dir)
model.build_vocab(chain(ted, corpus))
print '%s words & sents in vocab' % len(model.vocab)

print 'epochs on corpus:'
start = timeit.default_timer()
for epoch in range(10):
    model.train(corpus)
    print epoch,
    model.alpha -= 0.002  # decrease the learning rate
print '(%ss)' % (timeit.default_timer() - start)

print 'Stop training softmax!'
model.train_words = False

print 'epochs on ted:'
model.alpha=0.025
start = timeit.default_timer()
for epoch in range(10):
    model.train(ted)
    print epoch,
    model.alpha -= 0.002  # decrease the learning rate
print '(%ss)' % (timeit.default_timer() - start)

save_ted_docs(model, ted_dir, export_dir)