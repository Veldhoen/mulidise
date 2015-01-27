#!/usr/bin/python
from gensim.models.doc2vec import Doc2Vec, LabeledLineSentence, LabeledSentence
from itertools import izip, islice, izip_longest
import sys, os, glob
import timeit
from inspection import preprocess, inspect_words, inspect_sentences
from contextlib import nested
from numpy import sqrt, newaxis, float32 as REAL

def mappend(iter, ap):
    for i in iter:
        yield ap, i

class MultiText(object):
    def __init__(self, filenames, size, filt=None):
        self.filenames = filenames
        self.size = size
        self.filter = filt
 
    def __iter__(self):
        filt = self.filter
        with nested(*(open(f) for f in self.filenames)) as texts:
            pe = {os.path.splitext(t.name)[1][1:]:t for t in texts }
            pe = [mappend(t,e) for e,t in pe.items()]
            for i, line in enumerate(islice(izip(*pe), self.size)):
                sl = ((s,l) for s,l in line if (not filt) or (filt==s))
                ws = ['%s_%s'%(w,s) for s,l in sl for w in preprocess(l).split()]
                lbl = preprocess(dict(line)['en'])
                yield LabeledSentence(words=ws, labels=[lbl])


files =  glob.glob(os.path.expanduser(sys.argv[1]))
print 'Using files:', '\n\t'.join(['']+files)

print 'Saving vectors to', sys.argv[2]

n=50000
suf = sys.argv[3]
print 'Training dbow language', suf
sentences = MultiText(files, n, filt=suf)
print '%s sentences' % n, 'like', next(islice(sentences, 1))

model = Doc2Vec(dm=0, alpha=0.025, min_alpha=0.025, size=256)
model.build_vocab(sentences)
print '%s words  in vocab' % (len(model.vocab)-n)

print 'epochs'
for epoch in range(10):
    model.train(sentences)
    print epoch
    model.alpha -= 0.002  # decrease the learning rate
stop = timeit.default_timer()

inspect_sentences(model)

model2 = Doc2Vec(dm=0, alpha=0.025, min_alpha=0.025, size=256)
model2.build_vocab(MultiText(files, n))

for ls in MultiText(files, n):
    # set sentence vector
    pen = ls.labels[0]
    if pen in model:
        # set words
        for w in ls.words:
            if w in model2:
                model2.syn0[model2.vocab[w].index] += model[pen]
for w in model2.vocab:
    model2.syn0[model2.vocab[w].index] /= (model2.vocab[w].count+1.0)
model2.syn0norm = (model2.syn0 / sqrt((model2.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)

with open(sys.argv[2], 'a') as out:
    for w in model2.vocab:
        if '_' in w:
            out.write('%s : %s\n'%(w, ' '.join((str(d) for d in model2[w]))))