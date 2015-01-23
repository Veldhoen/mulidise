#!/usr/bin/python

from preprocessTools import *
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
                for fileName in os.listdir(labelDir):
                    emb = encodeText(os.path.join(labelDir,fileName), embeddings, idfs,tag)
                    print fileName, 'Document embedding:',emb
                    if len(emb)<1:
                       print 'watsgeburt', fileName
                       sys.exit()
                    outFile = os.path.join(outDir,kind+'.'+ folder+'.emb')
                    outputEmbeddings(emb, i+1,outFile)
    print 'Done.'
    
def main(argv):
    global embeddings
    embeddings= initializeEmbeddings(embeddingsFile)

    global idfs
    print 'call function initializeIDFS'
    idfs = initializeIDFS(idfFile)

    walkDataDir(dataDir,outDir)

if __name__ == "__main__":
   main(sys.argv[1:])
