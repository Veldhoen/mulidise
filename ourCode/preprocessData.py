#!/usr/bin/python

import sys, getopt, os
import shutil

def walkDataDir(inDir, outDir):
    kinds = ['train','test']
    labels = ['positive', 'negative']

    try: shutil.rmtree(outDir, ignore_errors=True)
    except: print 'problem removing tree'


#    try:
    for kind in kinds:
        print kind
        kindDir = os.path.join(inDir,kind)
        topics = os.listdir(kindDir)
        for topic in topics:
            print topic
            for i in range(len(labels)):
                labelDir = os.path.join(kindDir,topic,labels[i])
                for fileName in os.listdir(labelDir):
                    emb = encodeText(os.path.join(labelDir,fileName))
                    outFile = os.path.join(outDir,kind,topic)+'.emb'


                    outputEmbeddings(emb, i+1,outFile)

#        print topicDirs
#    except:
 #       print 'help!'




def encodeText(fromFile):
    summedEmbeddings = [0]*len(embeddings.values()[1])
    norm = 0

    with open(fromFile, 'r') as f:
         for line in f:
             for token in line.split():
                 token = token.lower()
                 #DELETE THE NEXT LINE!
                 token = token.split('_')[0]
                 if token in embeddings:
                     weight = idf.setdefault(token, 1)
                     # if there is no idf value, use 1
                     weighedEmbedding = [value*weight for value in embeddings[token]]
                     summedEmbeddings=[sum(x) for x in zip(summedEmbeddings,weighedEmbedding )]
                     norm += weight
                 else:
#                     print 'no entry for', token
                     break
    if norm == 0: norm = 1
    documentEmbedding = [value/norm for value in summedEmbeddings]
    return documentEmbedding

def initializeEmbeddings(fromFile):
    print 'initializing embeddings...'
    global embeddings
    embeddings = dict()
    with open(fromFile, 'r') as f:
         for line in f:
             parts = line.strip().split(':')
             embeddings[parts[0].strip()] = [float(val) for val in parts[1].split()]
    print 'done.'

def initializeIDFS(fromFile):
    print 'initializing idf values...'

    global idf
    with open(fromFile, 'r') as f:
         for line in f:
             parts = line.strip().split()
             idf[parts[0]]=float(parts[2])
    print 'done.'

def outputEmbeddings(emb, label, outputFile):
    d = os.path.dirname(outputFile)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(outputFile, 'a') as f:
         f.write(str(label))
         for i in range(len(emb)):
             f.write(' '+str(i+1)+':'+str(emb[i]))
         f.write('\n')



def main(argv):

    errorMessage="preprocessData.py -d [data directory] -e [word embeddings] -o [output directory] -i [idfs]"

    try:
      opts, args = getopt.getopt(argv,"hd:e:o:i",["dataDir=","embeddings=""outDir=","idfs="])

    except getopt.GetoptError:
      print errorMessage
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print errorMessage
         sys.exit()
      elif opt in ("-d", "--dataDir"):
         dataDir = arg
      elif opt in ("-e", "--embeddings"):
         embeddingsFile = arg
      elif opt in ("-o", "--outDir"):
         outDir = arg
      elif opt in ("-i", "--idfs"):
         idfFile = arg
    try: dataDir, embeddingsFile, outDir
    except:
        print errorMessage
        sys.exit()

    initializeEmbeddings(embeddingsFile)


    global idf
    idf = dict()
    try: initializeIDFS(idfFile)
    except: idf

    walkDataDir(dataDir,outDir)
#    documentEmbedding = encodeText(textFile)
#    outputEmbeddings(documentEmbedding, label, outputFile)


if __name__ == "__main__":
   main(sys.argv[1:])