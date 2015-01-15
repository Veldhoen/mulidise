#!/usr/bin/python

import sys, getopt, os
import shutil
import numpy
from numpy import array
from multiprocessing import Pool


def walkDataDir(dirs):
    inDir = dirs[0]
    outDir = dirs[1]
    print '\twalking',inDir,'...'
    kinds = ['train','test']
    labels = ['positive', 'negative']

    try: shutil.rmtree(outDir, ignore_errors=True)
    except: print 'problem removing tree'

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
                    if len(emb)<1:
                       print 'watsgeburt', fileName
                       sys.exit()
                    outFile = os.path.join(outDir,kind+'.'+topic+'.emb')
                    outputEmbeddings(emb, i+1,outFile)
    print '\tdone.'



def encodeText(fromFile):
    summedEmbeddings = numpy.zeros(len(embeddings.values()[1]))
    norm = 0

    with open(fromFile, 'r') as f:
         for line in f:
             for token in line.split():
                 token = token.lower()
                 #DELETE THE NEXT LINE!
                 token = ''.join(token.split('_')[:-1])
                 if token in embeddings:
                     weight = idf.setdefault(token, 1)
                     # if there is no idf value, use 1
                     summedEmbeddings+=weight*embeddings[token]
                     norm += weight
                 else:
#                     print 'no entry for', token
                     break
    if norm == 0: norm = 1.0
    documentEmbedding = summedEmbeddings/norm
    return documentEmbedding

def walkLanguages(inDir,outDir):
    p = Pool()
    lans = ['en','it','de','es','fr','nl','pb','pl','ro']
    ins = [os.path.join(inDir,lan+'-'+lans[0]) for lan in lans[1:]]
    ins.insert(0,os.path.join(inDir,lans[0]+'-'+lans[1]))
    outs=[os.path.join(outDir,lan) for lan in lans]


    p.map(walkDataDir,zip(ins,outs))

    # English/ pivot language:
#    walkDataDir(os.path.join(inDir,lans[0]+'-'+lans[1]),os.path.join(outDir,lans[0]))
    # Other languages:
#    for lan in lans[1:]:
#        walkDataDir(os.path.join(inDir,lan+'-'+lans[0]),os.path.join(outDir,lan))


def initializeEmbeddings(fromFile):
    print '\tinitializing embeddings...'
    global embeddings
    embeddings = dict()
    with open(fromFile, 'r') as f:
         for line in f:
             parts = line.strip().split(':')
             embeddings[parts[0].strip()] = array([float(val) for val in parts[1].split()])
    print '\tdone.'

def initializeIDFS(fromFile):
    print '\tinitializing idf values...'

    global idf
    with open(fromFile, 'r') as f:
         for line in f:
             parts = line.strip().split()
             idf[parts[0]]=float(parts[2])
    print '\tdone.'

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

    walkLanguages(dataDir,outDir)

if __name__ == "__main__":
   main(sys.argv[1:])