import csv

def inspect_sentences(model):
    lines = [
    'you will be aware from the press and television that there have been a number of bomb explosions and killings in sri lanka'
    ,'but madam president my personal request has not been met'
    ,'we then put it to a vote'
    ,'thank you very much'
    ]
    for l in lines:
        print '%s:\n' % l, model.most_similar(l, topn=10)

def inspect_words(model):
    en = ['mrs_en', 'april_en', 'objection_en', 'debate_en', 'answer_en']
    de = ['frau_de', 'april_de', 'einwand_de', 'debatte_de', 'antwort_de']
    for l, suf in [(en, '_de'), (de, '_en')]:
        for w in l:
            print '%s:' % w
            print model.most_similar(w, topn=10)
            print most_similar_suffixed(model, suf, w, topn=10)

    simtot = 0.0
    for row in csv.DictReader(open('dictionary.csv')):
        sim = model.similarity(*['%s_%s'%(a,b) for (b,a) in row.items()])
        print row, sim
        simtot += sim
    print 'total:', simtot

def most_similar_suffixed(model, suffix, positive=[], negative=[], topn=10):
    sim = model.most_similar(positive, negative, topn=10000000)
    return [(w,d) for w,d in sim if w.endswith(suffix)][:topn]