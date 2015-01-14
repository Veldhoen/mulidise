benno=/home/bkruit/mulidise
sara=/home/sveldhoen/mulidise

experiment=$1
embeddings=$benno/document-representations/data/embeddings/original-de-en.en

mkdir $experiment/docEmbeddings
mkdir $experiment/models
mkdir $experiment/results


tedDocs=$sara/ted-cldc
tedIDFs=$sara/idfsTED
classifiers=$benno/document-representations/bin
preprocess=$benno/mulidise/ourCode/preprocessData.py

echo preprocess data...
python -u $preprocess\
       -d $tedDocs \
       -e $embeddings \
       -o $experiment/docEmbeddings \
       -i $tedIDFs
echo done.

languages=(en de es fr it nl pb pl ro)
topics=(art arts biology business creativity culture design economics education entertainment global health politics science technology)

for lan in ${languages[@]}; do
  echo $lan
  for topic in ${topics[@]}; do
    echo $topic
    # train, i.e. create model:
    java  -ea -Xmx2000m -cp \
      $classifiers ApLearn  \
      --train-set  $experiment/docEmbeddings/$lan/train.$topic.emb \
      --model-name $experiment/models/$lan.$topic.model \
      --epoch-num 10
  done
done


for lan1 in ${languages[@]}; do
  otherLans=`echo ${languages[@]}| sed "s/\b$lan\b//g"`
  for lan2 in ${otherLans[@]}; do
    for topic in ${topics[@]}; do
      #test classifier of lan1 on lan2
      java  -ea -Xmx2000m -cp \
        $classifiers ApClassify
        --test-set $experiment/docEmbeddings/$lan2/test.$topic.emb \
        --model-name $experiment/models/$lan.$topic.model \
        > $experiment/results/$lan1-$lan2.$topic.result
    done
  done
done
