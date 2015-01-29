from gensim.models.word2vec import Word2Vec, Vocab
from numpy import zeros, float32 as REAL
import sys
import logging

def load_eval_format(fname, fvocab=None, suffix_filter=None):
    """ Adapted from Word2Vec.load_word2vec_format """
    counts = None
    if fvocab is not None:
        counts = {}
        with open(fvocab) as fin:
            for line in fin:
                word, count = unicode(line.decode('utf-8')).strip().split()[:2]
                counts[word] = int(count)
    with open(fname) as fin:
        if suffix_filter:
            vocab_size = sum(1 for line in fin if line.split(' ', 1)[0].endswith(suffix_filter))
        else:
            vocab_size = sum(1 for line in fin)
    with open(fname) as fin:
        layer1_size = len(fin.readline().split()) - 2
    with open(fname) as fin:
        result = Word2Vec(size=layer1_size)
        result.syn0 = zeros((vocab_size, layer1_size), dtype=REAL)
        if suffix_filter:
            fin = (line for line in fin if line.split(' ', 1)[0].endswith(suffix_filter))
        for line_no, line in enumerate(fin):
            parts = unicode(line.decode('utf-8')).split()
            if len(parts) != layer1_size + 2:
                raise ValueError("invalid vector on line %s (is this really the text format?): \n%s" % (line_no, line))
            word, weights = parts[0], map(REAL, parts[2:])
            if suffix_filter:
                word = word[:-len(suffix_filter)]
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

suf = sys.argv[2] if len(sys.argv)>1 else None
model = load_eval_format(sys.argv[1], suffix_filter=suf)

# logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
# fh = logging.FileHandler(sys.argv[1]+'.analogy')
# logging.getLogger("gensim.models.word2vec").addHandler(fh)
# model.accuracy('/Users/benno/Documents/wordvecproj/questions-words.txt')