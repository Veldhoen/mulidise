from __future__ import division
from numpy import array, dot

def analogyScore(a1,a2,b1,b2):
    return cos(a1,a2)*cos(b2,b1)/(cos(b2,a1)+0.001)

def cos(a,b):

    return dot(a,b)/(len(a)*len(b))


def computeScore(embeddings, analogies):

    vocabulary = embeddings.keys()
#    print vocabulary
    correct = 0
    incorrect = 0
    unsolvable = 0

    for (a1,a2,b1,b2) in analogies:
#        print a1,a2,b1,b2
        knownWords = 0
        if a1+'_en' in vocabulary: knownWords+=1
        if a2+'_en' in vocabulary: knownWords+=1
        if b1+'_en' in vocabulary: knownWords+=1
        if b2+'_en' in vocabulary: knownWords+=1
 #       print knownWords
        if knownWords == 4:
            vecA1 = embeddings[a1+'_en']
            vecA2 = embeddings[a2+'_en']
            vecB1 = embeddings[b1+'_en']
            maxScore = -1
            bestWord = ""
            for word in vocabulary:
                if word not in [a1,a2,b1]:
                    vecB2 = embeddings[word]
                    score = analogyScore(vecA1,vecA2,vecB1,vecB2)
                    if score > maxScore:
                        maxScore = score
                        bestWord = word.split('_')[:-1]
            if bestWord == b2: correct += 1
            else: incorrect += 1
        else:
       #     print a1, a2, b1, b2
            unsolvable +=1
    solvable = correct + incorrect
    if solvable >0: totalScore = correct/(correct+incorrect)
    else: totalScore = 0
    print 'Solved', solvable, 'analogy questions.'
    print 'Score:', totalScore, '(', correct, 'out of', incorrect,')'

def loadAnalogies(fromFile):
    analogies = set()
    with open(fromFile, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                analogies.add(tuple(parts))
    return analogies

def initializeEmbeddings(fromFile):
    print 'initializing embeddings...'
    embeddings = dict()
    with open(fromFile, 'r') as f:
         print 'opened', fromFile
         for line in f:
#             print line
             parts = line.strip().split(' : ')
	     if len(parts) < 2: True
#                print line
	     else:

                emb = array([float(val) for val in parts[1].split()])
	        if len(emb)>0:
                   embeddings[parts[0].strip()] = emb
	        else:
	    	   print 'no embedding for:',parts[0].strip()
    print 'done. Obtained', len(embeddings), 'embeddings.'
    return embeddings


def main():
    analogies = loadAnalogies('../../data/analogytask/word-test.v1.txt')
    embeddings = initializeEmbeddings('../../data/embeddingsKlementievEtAl/concatenated.emb')
#    for word, emb in embeddings.iteritems():
#        print word
    computeScore(embeddings,analogies)

if __name__ == "__main__":
   main()


