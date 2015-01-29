#!/usr/bin/python
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from numpy import sqrt, newaxis, float32 as REAL

def parawords(model1, model2, labeled_sentences, out_file, filt='_'):
    """
        `model1` can be trained on a subset of `model2` vocabulary
    """
    for ls in labeled_sentences:
        # get alignment vector (in our case, alignments are sentences)
        for alignment in ls.labels:
            if alignment in model1:
                # set words
                for w in ls.words:
                    if w in model2:
                        model2.syn0[model2.vocab[w].index] += model1[alignment]
    for w in model2.vocab:
        model2.syn0[model2.vocab[w].index] /= (model2.vocab[w].count+1.0)
    model2.syn0norm = (model2.syn0 / sqrt((model2.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)

    with open(out_file, 'w') as out:
        for w in model2.vocab:
            if filt in w: # filter on word/label character
                out.write('%s : %s\n'%(w, ' '.join((str(d) for d in model2[w]))))