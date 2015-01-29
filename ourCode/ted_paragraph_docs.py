#!/usr/bin/python
from gensim.models.doc2vec import Doc2Vec, LabeledLineSentence, LabeledSentence
from itertools import izip, chain, islice
import os
import numpy as np

def ted_file_filter(root):
    for (dirpath, dirnames, filenames) in os.walk(root):
        for name in filenames:
            if name.endswith('.ted'):
                yield name, os.path.join(dirpath, name)


def save_ted_docs(model, ted_dir, out_dir):
    """
        `model` has sentences in `<l1>`
        `ted_dir` ends with `/<l1>-<l2>`
    """
    size = model.layer1_size
    for name, f in ted_file_filter(ted_dir):
        skipped_sentences = 0
        dataset, topic, posneg = f.split('/')[-4:-1]
        with open(f) as lines:
            vec = np.zeros(size)
            for line in lines:
                if line in model:
                    vec += model[line]
                else:
                    skipped_sentences += 1
        out = os.path.join(out_dir, '%s/%s' % (dataset, topic))
        vecstr = ' '.join(['%s:%s' % (i, d) for i,d in enumerate(vec)])
        if not os.path.exists(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
        with open(out, 'a') as o:
            o.write('%s %s\n' % (1+int(posneg=='negative'), vecstr))
        print 'skipped %s sentences in %s' % (skipped_sentences, f)

class TedLabeledLineSentence(object):
    def __init__(self, ted_dir):
        # make a dict of unique documents
        self.ted_docs = {name:f for name, f in ted_file_filter(ted_dir)}
 
    def __iter__(self):
        for doc in self.ted_docs.values():
            with open(doc) as lines:
                for l in lines:
                    # ted corpus has lang suffixes
                    yield LabeledSentence(words=l.split(), labels=[l])

class BiTedLabeledLineSentence(object):
    def __init__(self, ted_dir):
        path_parts = ted_dir.rstrip('/').split('/')
        print path_parts
        l1,l2 = path_parts[-1].split('-')
        ted_dir2 = os.path.join('/'.join(path_parts[:-1]), '%s-%s'%(l2,l1))
        print ted_dir2
        # make two dicts of unique documents
        self.ted_l1 = {}
        self.ted_l2 = {}
        for name, f in ted_file_filter(ted_dir):
            self.ted_l1[name] = f
            self.ted_l2[name] = os.path.join(ted_dir2, f.replace(ted_dir, ''))
 
    def __iter__(self):
        for name in self.ted_l1:
            with open(self.ted_l1[name]) as l1s, open(self.ted_l2[name]) as l2s:
                for l1, l2 in izip(l1s, l2s):
                    # ted corpus has lang suffixes
                    ws = l1.split() + l2.split()
                    yield LabeledSentence(words=ws, labels=[l1])