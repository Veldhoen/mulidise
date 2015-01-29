In order to evaluate a set of word embeddings:

1. Make sure the word embeddings are in a single file <emb>
   in the followingformat:
   <word>_<languagetag> : <v0> <v1> ... <vn>
   Note that the language tag for Poruguese must be '_pb'
2. Create a directory <eval> for the evaluation
3. Run classifierTED and/or classifierRCV with <eval> as a
   first and <emb> as a second argument, e.g.:
   bash classifierTED.sh evaluation/bilingual_dbow/ ourEmbeddins/bilingual_dbow_vecs.txt

   The classifier runs:
   1. preprocessing
      obtain document embeddings given the word embeddings.
   2. train classifiers on training data
      for all language-topic combinations
   3. test classifiers on test data
      for all language-language-topc combinations
   4. Summarize results in a readable format
4. The classification results will end up in <eval>/results

The script classifierRCVgivenDocs.sh was created
in order to test readymade document representations.
It was used for the evaluation of dbow sentences.


NB: The evaluation scripts rely on:
- the distribution of Klementiev and Titov provided to us by
  Ivan Titov, which is to be found on 'deze':
  /home/bkruit/mulidise/documentRepresentations/
  and from which we use both data and src code.
- TED data from http://www.clg.ox.ac.uk/tedcorpus
  which is on 'deze':
  /home/sveldhoen/mulidise/ted-cldc/
- TF-IDF values for the TED corpus
  computed with the script 'getidfvalues.py'
  /home/sveldhoen/mulidise/idfsTED/