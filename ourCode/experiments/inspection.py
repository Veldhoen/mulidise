import csv
import string

def preprocess(s):
    return s.rstrip().lower().translate(string.maketrans("",""), string.punctuation)

def inspect_sentences(model):
    lines = [
    'you will be aware from the press and television that there have been a number of bomb explosions and killings in sri lanka'
    ,'but madam president my personal request has not been met'
    ,'we then put it to a vote'
    ,'thank you very much'
    ]
    for l in lines:
        if l in model:
            print '%s:\n' % l, model.most_similar(l, topn=10)

def inspect_words(model):
    en = ['mrs_en', 'april_en', 'objection_en', 'debate_en', 'answer_en']
    de = ['frau_de', 'april_de', 'einwand_de', 'debatte_de', 'antwort_de']
    for l, suf in [(en, '_de'), (de, '_en')]:
        for w in l:
            if w in model:
                print '%s:' % w
                print model.most_similar(w, topn=10)
                print most_similar_suffixed(model, suf, w, topn=10)
    try:
        simtot = 0.0
        for row in csv.DictReader(open('dictionary.csv')):
            sim = model.similarity(*['%s_%s'%(a,b) for (b,a) in row.items()])
            print row, sim
            simtot += sim
        print 'total:', simtot
    except:
        print '(dictionary failed)'

def most_similar_suffixed(model, suffix, positive=[], negative=[], topn=10):
    sim = model.most_similar(positive, negative, topn=10000000)
    return [(w,d) for w,d in sim if w.endswith(suffix)][:topn]