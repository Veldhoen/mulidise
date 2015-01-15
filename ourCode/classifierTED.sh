benno=/home/bkruit/mulidise
sara=/home/sveldhoen/mulidise

experiment=$1
embeddings=$2
#embeddings=$benno/document-representations/data/embeddings/original-de-en.en

mkdir -p $experiment/docEmbeddings
mkdir $experiment/models
mkdir $experiment/results


tedDocs=$sara/ted-cldc
tedIDFs=$sara/idfsTED
classifiers=$benno/document-representations/bin
preprocess=$benno/mulidise/ourCode/preprocessData.py

date
echo preprocess data...
python -u $preprocess\
       -d $tedDocs \
       -e $embeddings \
       -o $experiment/docEmbeddings \
       -i $tedIDFs
echo done.


date
echo training models...
languages=(en de es fr it nl pb pl ro)
topics=(art arts biology business creativity culture design economics education entertainment global health politics science technology)

for lan in ${languages[@]}; do
  for topic in ${topics[@]}; do
    echo -e '\t' $lan-$topic
    # train, i.e. create model:
    java  -ea -Xmx2000m -cp \
      $classifiers ApLearn  \
      --train-set  $experiment/docEmbeddings/$lan/train.$topic.emb \
      --model-name $experiment/models/$lan.$topic.model \
      --epoch-num 10 &
  done
done
echo done.

wait

date
echo testing models...

for lan1 in ${languages[@]}; do
#  otherLans=`echo ${languages[@]}| sed "s/\b$lan\b//g"`
# for lan2 in ${otherLans[@]}; do
  for lan2 in ${languages[@]}; do
    for topic in ${topics[@]}; do
      echo -e '\t' $lan1-$lan2: $topic
      #test classifier of lan1 on lan2
      java  -ea -Xmx2000m -cp \
        $classifiers ApClassify \
        --test-set $experiment/docEmbeddings/$lan2/test.$topic.emb \
        --model-name $experiment/models/$lan.$topic.model \
        > $experiment/results/$lan1-$lan2.$topic.result &
    done
  done
done
echo done.

wait

date
FILES=$experiment/results/*
OUT=$experiment/results/allResults.txt
echo 'src\ttar\ttopic\taccracy'>$OUT
for f in $FILES
do
  echo -n $f | sed 's/.*\/\///' | sed 's/\./\t/g'| sed 's/-/\t/' | sed 's/result//' >> $OUT
  cat $f | grep -Po '\d+.\d+' >> $OUT
done
