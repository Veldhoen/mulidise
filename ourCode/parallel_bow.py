#!/usr/bin/python
from gensim.models.doc2vec import Doc2Vec, LabeledLineSentence, LabeledSentence
from itertools import izip, islice, izip_longest
import sys, os
from inspection import preprocess
from contextlib import nested
from numpy import sqrt, newaxis, float32 as REAL

class ParallelMergedLabeledLineSentence(object):
    """
        all the files called `file_root.lang` are opened,
        one by one together, their lines are preprocessed,
        those lines are stuck together with suffixes on the words,
        and returned with the first one as a label
    """
    def __init__(self, file_root, langs, size=None):
        self.files = ['%s.%s'%(file_root,l) for l in langs]
        self.size = size
        self.l1 = langs[0]
 
    def __iter__(self):
        def mappend(iter, ap):
            for i in iter:
                yield ap, i
        with nested(*(open(f) for f in self.files)) as texts:
            # dict of {lang: text} for open files
            pe = {os.path.splitext(t.name)[1][1:]:t for t in texts }
            # transposed
            pe = [mappend(t,e) for e,t in pe.items()]
            for i, line in enumerate(islice(izip(*pe), self.size)):
                ws = ['%s_%s'%(w,s) for s,l in line for w in preprocess(l).split()]
                lbl = preprocess(dict(line)[self.l1])
                yield LabeledSentence(words=ws, labels=[lbl])