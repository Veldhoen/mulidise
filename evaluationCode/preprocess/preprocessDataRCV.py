#!/usr/bin/python

import sys, getopt, os
import shutil
import numpy
from numpy import array

def walkDataDir(inDir, outDir):
    try: shutil.rmtree(outDir, ignore_errors=True)
    except: print 'problem removing tree'

    print 'walking',inDir,'...'
    kinds = ['train','test']
    labels = ['C', 'E', 'G', 'M']

    for kind in kinds:
        print kind
        kindDir = os.path.join(inDir,kind)
        dataSets = os.listdir(kindDir)
        for folder in dataSets:
	    tag = '_'+folder[:2].lower()
	    folderDir = os.path.join(kindDir,folder)
	    if os.path.isdir(folderDir):
             print folder
             for i in range(len(labels)):
                labelDir = os.path.join(kindDir,folder,labels[i])
	 	try:
                 for fileName in os.listdir(labelDir):
                    emb = encodeText(os.path.join(labelDir,fileName), embeddings, idfs)
                    if len(emb)<1:
                       print 'watsgeburt', fileName
                       sys.exit()
                    outFile = os.path.join(outDir,kind+'.'+ folder+'.emb')
                    outputEmbeddings(emb, i+1,outFile)
		except: print 'Could not obtain files for', fileName
    print 'Done.'
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

if __name__ == "__main__":
   main(sys.argv[1:])
