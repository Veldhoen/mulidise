#!/usr/bin/python

import sys, getopt

def encodetext(text,embeddings,idfs):
    tokens = text.split()
    summedEmbeddings = [0]*len(embeddings.getkeys[0])
    for token in tokens:
        weight = idfs.setdefault(token, default=1)
        summedEmbeddings+= [value*weight for value in embeddings[token]]
        numerator += weight

def initializeEmbeddings(fromFile):
    global embeddings
    embeddings = dict()
    with open(fromFile, r) as f:
         for line in f:
             word, vector = line.split(':')
             embeddings[word] = [num(val) for val in vector.split()]

def initializeIDFS(fromFile):
    print 'initializing idf values...'
    global idf
    with open(fromFile, 'r') as f:
         print 'opened',fromFile, len(f), 'lines.'
         for line in f:
             word, dunno = line.split(':')
             print word, dunno
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

    global idf
    idf = dict()
    try: initializeIDFS(idfFile)
    except: idf
    
    for key, value in idf.iteritems():
        print 'key:', key
        print 'value:', value


#    print 'Files:', textFile,embeddings,output,idf


if __name__ == "__main__":
   main(sys.argv[1:])