#!/usr/bin/python

from preprocessTools import *
import sys, getopt, os
import shutil
import numpy
from numpy import array

def walkDataDir(inDir, outDir):
    print 'walking',inDir,'...'
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
                    emb = encodeText(os.path.join(labelDir,fileName), embeddings, idfs)
                    if len(emb)<1:
                       print 'watsgeburt', fileName
                       sys.exit()
                    outFile = os.path.join(outDir,kind+'.'+topic+'.emb')
                    outputEmbeddings(emb, i+1,outFile)
    print 'Done.'

def walkLanguages(inDir,outDir):
    lans = ['en','it','de']
#    lans = ['en','it','de','es','fr','nl','pb','pl','ro']
    # English/ pivot language:
    walkDataDir(os.path.join(inDir,lans[0]+'-'+lans[1]),os.path.join(outDir,lans[0]))
    # Other languages:
    for lan in lans[1:]:
        walkDataDir(os.path.join(inDir,lan+'-'+lans[0]),os.path.join(outDir,lan))

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

    global embeddings
    embeddings = initializeEmbeddings(embeddingsFile)

    global idfs
    idfs = initializeIDFS(idfFile)
    walkLanguages(dataDir,outDir)

if __name__ == "__main__":
   main(sys.argv[1:])
