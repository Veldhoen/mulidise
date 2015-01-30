name=$1
root=/home/bkruit/mulidise
embeddings=$root/ourEmbeddingsNew
evaluations=$root/evaluationsNew

mkdir evaluations/$name
nohup bash classifierTED.sh \
      $evaluations/$name     \
      $embeddings/$name.txt \
      > evaluations/$name/TED.nohup.out
nohup bash classifierRCV.sh \
      $evaluations/$name     \
      $embeddingsNew/$name.txt \
      > evaluations/$name/RCV.nohup.out