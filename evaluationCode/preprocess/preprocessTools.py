#!/usr/bin/python

import sys, getopt, os
import shutil
import numpy
from numpy import array

def encodeText(fromFile, embeddings, idfs, tag=""):
    summedEmbeddings = numpy.zeros(len(embeddings.values()[1]))
    norm = 0

    with open(fromFile, 'r') as f:
         for line in f:
             for token in line.split():
                 token = token.lower()+tag
                 if token in embeddings:
                     weight = idfs.setdefault(token, 1) # if there is no idf value, use 1
                     summedEmbeddings+=weight*embeddings[token]
                     norm += weight
                 else:
                     True
#                     print 'no entry for', token
    if norm == 0: norm = 1.0
    documentEmbedding = summedEmbeddings/norm
    return documentEmbedding


def initializeEmbeddings(fromFile):
    print 'initializing embeddings...'
    embeddings = dict()
    with open(fromFile, 'r') as f:
         for line in f:
             parts = line.strip().split(' : ')
	     if len(parts) < 2:
                print line
	     else:
                emb = array([float(val) for val in parts[1].split()])
	        if len(emb)>0:
                   embeddings[parts[0].strip()] = emb
	        else:
	    	   print '\tno embedding for:',parts[0].strip()
    print 'done. Retrieved', len(embeddings), 'embeddings.'
    return embeddings

def initializeIDFS(fromFile):
    idfs = dict()
    if fromFile == "":
       print 'no idf values are used.'
    else:
       print 'initializing idf values...'
       with open(fromFile, 'r') as f:
         for line in f:
             parts = line.strip().split()
             idfs[parts[0]]=float(parts[2])
       print 'done.'

    return idfs

def outputEmbeddings(emb, label, outputFile):
    d = os.path.dirname(outputFile)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(outputFile, 'a') as f:
         f.write(str(label))
         for i in range(len(emb)):
             f.write(' '+str(i+1)+':'+str(emb[i]))
         f.write('\n')
         
def readArgs(argv):
    errorMessage="preprocessData.py -d [data directory] -e [word embeddings] -o [output directory] -i [idfs]"
    try:
      opts, args = getopt.getopt(argv,"hd:e:o:i",["dataDir=","embeddings=""outDir=","idfs="])
    except getopt.GetoptError:
      print errorMessage
      sys.exit(2)

    idfFile = ""
    print opts

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
         print 'idfFile=', idfFile, '(',arg,')'
    try: dataDir, embeddingsFile, outDir, idfFile
    except:
        print errorMessage
        sys.exit()
    return dataDir,embeddingsFile,outDir,idfFile
