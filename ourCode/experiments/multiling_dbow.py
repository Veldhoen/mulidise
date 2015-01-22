#!/usr/bin/python
from gensim.models.doc2vec import Doc2Vec, LabeledLineSentence, LabeledSentence
from itertools import izip, islice, izip_longest
import sys, os, glob
import timeit
from inspection import preprocess, inspect_words, inspect_sentences
from contextlib import nested

def mappend(iter, ap):
    for i in iter:
        yield ap, i

class MultiText(object):
    def __init__(self, filenames, size):
        self.filenames = filenames
        self.size = size
 
    def __iter__(self):
        with nested(*(open(f) for f in self.filenames)) as texts:
            pe = {os.path.splitext(t.name)[1][1:]:t for t in texts }
            pe = [mappend(t,e) for e,t in pe.items()]
            for i, line in enumerate(islice(izip(*pe), self.size)):
                ws = ['%s_%s'%(w,s) for s,l in line for w in preprocess(l).split()]
                lbl = preprocess(dict(line)['en'])
                yield LabeledSentence(words=ws, labels=[lbl])


files =  glob.glob(os.path.expanduser(sys.argv[1]))
print 'Using files:', '\n\t'.join(['']+files)

n=50000
sentences = MultiText(files, n)

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