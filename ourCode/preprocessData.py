#!/usr/bin/python

import sys, getopt

def encodeText(fromFile):
    summedEmbeddings = [0]*len(embeddings.keys()[0])
    norm = 0

    with open(fromFile, 'r') as f:
         for line in f:
             for token in line.split():
                 token = token.lower()
                 if token in embeddings:
                     weight = idf.setdefault(token, 1)
                     # if there is no idf value, use 1
                     summedEmbeddings+= [value*weight for value in embeddings[token]]
                     norm += weight
                 else:
                     print 'no entry for', token
                     break
    documentEmbedding = [value/norm for value in summedEmbeddings]

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

def main(argv):

    errorMessage="preprocessData.py -t [textfile] -e [word embeddings] -o [output file] -i [idfs]"

    try:
      opts, args = getopt.getopt(argv,"ht:e:o:i:",["textfile=","embeddings=""outputfile=","idfs="])

    except getopt.GetoptError:
      print errorMessage
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print errorMessage
         sys.exit()
      elif opt in ("-t", "--textfile"):
         textFile = arg
      elif opt in ("-e", "--embeddings"):
         embeddingsFile = arg
      elif opt in ("-o", "--outputfile"):
         outputFile = arg
      elif opt in ("-i", "--idfs"):
         idfFile = arg
    try: textFile, embeddingsFile, outputFile
    except:
        print errorMessage
        sys.exit()

    initializeEmbeddings(embeddingsFile)


    global idf
    idf = dict()
    try: initializeIDFS(idfFile)
    except: idf

    encodeText(textFile)

#    for key, value in idf.iteritems():
#        print 'word:', key, 'idf:', value

#    for key, value in embeddings.iteritems():
#        print 'word:', key, 'embedding:', value

#    print 'Files:', textFile,embeddings,output,idf


if __name__ == "__main__":
   main(sys.argv[1:])