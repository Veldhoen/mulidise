#!/usr/bin/python

#import sys, getopt, os
#import shutil
import numpy
from numpy import array

def encodeText(fromFile, embeddings, idfs):
    summedEmbeddings = numpy.zeros(len(embeddings.values()[1]))
    norm = 0

    with open(fromFile, 'r') as f:
         for line in f:
             for token in line.split():
                 token = token.lower()
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

def initializeIDFS(fromFile=None):
    idfs = dict()
    if fromFile:
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