from gensim.models.word2vec import Word2Vec, Vocab
from numpy import zeros, float32 as REAL
import sys

def load_eval_format(fname, fvocab=None):
    """ Adapted from Word2Vec.load_word2vec_format """
    counts = None
    if fvocab is not None:
        counts = {}
        with open(fvocab) as fin:
            for line in fin:
                word, count = unicode(line.decode('utf-8')).strip().split()[:2]
                counts[word] = int(count)
    with open(fname) as fin:
        vocab_size = sum(1 for line in fin)
    with open(fname) as fin:
        layer1_size = len(fin.readline().split()) - 2
    with open(fname) as fin:
        result = Word2Vec(size=layer1_size)
        result.syn0 = zeros((vocab_size, layer1_size), dtype=REAL)
        for line_no, line in enumerate(fin):
            parts = unicode(line.decode('utf-8')).split()
            if len(parts) != layer1_size + 2:
                raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
            word, weights = parts[0], map(REAL, parts[2:])
            if counts is None:
                result.vocab[word] = Vocab(index=line_no, count=vocab_size - line_no)
            elif word in counts:
                result.vocab[word] = Vocab(index=line_no, count=counts[word])
            else:
                result.vocab[word] = Vocab(index=line_no, count=None)
            result.index2word.append(word)
            result.syn0[line_no] = weights
    result.init_sims(True)
    return result

# ops we need:
# - most similar other language
# - analogy without suffixes
# ?

model = load_eval_format(sys.argv[1])