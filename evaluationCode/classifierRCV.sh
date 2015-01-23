benno=/home/bkruit/mulidise
sara=/home/sveldhoen/mulidise

experiment=$1
embeddings=$2
#embeddings=$benno/document-representations/data/embeddings/original-de-en.en

languages=(en de)

mkdir -p $experiment/docEmbeddings/RCV
mkdir -p $experiment/models/RCV
mkdir -p $experiment/results/RCV


classifiers=$benno/document-representations/bin
preprocess=$benno/mulidise/evaluationCode/preprocess/preprocessDataRCV.py

date
echo preprocess data...
python -u $preprocess \
       -d $benno/document-representations/data/rcv-from-binodNoValid \
       -e $embeddings \
       -o $experiment/docEmbeddings/RCV \
       -i $benno/document-representations/data/idfs/concatenated.idf
echo done.



wait
date
echo training models...

sizes=(100 200 500 1000 5000 10000)
for lan in ${languages[@]}; do
  for size in ${sizes[@]}; do
    lanUP=$(echo $lan | tr '[:lower:]' '[:upper:]')
    echo -e '\t' $lan-$size
    # train, i.e. create model:
    java  -ea -Xmx2000m -cp \
      $classifiers ApLearn  \
      --train-set  $experiment/docEmbeddings/RCV/train.$lanUP$size.emb \
      --model-name $experiment/models/RCV/$lan.$size.model \
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
  lanUP=$(echo $lan1 | tr '[:lower:]' '[:upper:]')

  for lan2 in ${languages[@]}; do
  for size in ${sizes[@]}; do
      echo -e '\t' $lan1-$lan2: $size
      #test classifier of lan1 on lan2
      java  -ea -Xmx2000m -cp \
        $classifiers ApClassify \
        --test-set $experiment/docEmbeddings/RCV/test.$lan2.emb \
        --model-name $experiment/models/RCV/$lan1.$size.model \
        > $experiment/results/RCV/$lan1-$lan2.$size.result &
    done
  done
done
echo done.

wait

date
FILES=$experiment/results/RCV/*
OUT=$experiment/results/allResultsRCV.txt
echo -e "train\ttest\ttrainSize\taccuracy">$OUT
for f in $FILES
do
  echo -n $f | sed 's/.*\///' | sed 's/\./\t/g'| sed 's/-/\t/' | sed 's/result//g' >> $OUT
  numbers=`cat $f | grep -Po '\d+\.\d+'`
  echo $numbers | sed 's/ /\t/g' >> $OUT
done
