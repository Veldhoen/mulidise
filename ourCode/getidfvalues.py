import sys, getopt,os
from collections import defaultdict
import math

def getIDF(inDir,outFile):
    print inDir
    N=0
    docFreq = defaultdict(int)
    termFreq = defaultdict(int)
    for root, dirs, files in os.walk(inDir, topdown=False):
        N += len(files)
        for fileName in files:
#            try:
                with open(os.path.join(root, fileName),'r') as f:
                    terms = set()
                    for line in f:
                        for token in line.split():
                            terms.add(token)
                            termFreq[token]+=1
 #           except:
#                print 'mislukt:',fileName
#                break
                for token in terms:
                  docFreq[token]+=1

    with open(outFile, 'w') as f:
        for word, freq in docFreq.iteritems():
            f.write(word+'\t'+str(termFreq[word])+'\t'+str(math.log(N/freq))+'\n')


def walkLanguages(inDir,outDir):
    lans = ['en','it','de','es','fr','nl','pb','pl','ro']
    # English/ pivot language:
    getIDF(os.path.join(inDir,lans[0]+'-'+lans[1]),os.path.join(outDir,lans[0])+'.idf')
    # Other languages:
    for lan in lans[1:]:
        getIDF(os.path.join(inDir,lan+'-'+lans[0]),os.path.join(outDir,lan+'.idf'))

def main(argv):

    errorMessage="getidfvalues.py -d [data directory] -o [output directory]"

    try:
      opts, args = getopt.getopt(argv,"hd:o:",["dataDir=","outDir="])

    except getopt.GetoptError:
      print errorMessage
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print errorMessage
         sys.exit()
      elif opt in ("-d", "--dataDir"):
         dataDir = arg
      elif opt in ("-o", "--outDir"):
         outDir = arg
    try: dataDir, outDir
    except:
        print errorMessage
        sys.exit()
    walkLanguages(dataDir,outDir)



if __name__ == "__main__":
   main(sys.argv[1:])