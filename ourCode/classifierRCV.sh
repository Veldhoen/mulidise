benno=/home/bkruit/mulidise
sara=/home/sveldhoen/mulidise

experiment=$1
embeddings=$2
#embeddings=$benno/document-representations/data/embeddings/original-de-en.en

languages=(en de)

mkdir -p $experiment/docEmbeddings
mkdir $experiment/models
mkdir $experiment/results


RCVDocs=$benno/document-representations/data/rcv-from-binod
RCVIDFs=$benno/document-representations/data/idfs
classifiers=$benno/document-representations/bin
preprocess=$benno/mulidise/ourCode/preprocessDataReuters.py

date
echo preprocess data...
python -u $preprocess\
       -d $RCVDocs \
       -e $embeddings \
       -o $experiment/docEmbeddings \
       -i $RCVIDFs
echo done.


date
echo training models...

sizes=(100 200 500 1000 5000 10000)
for lan in ${languages[@]}; do
  for size in ${sizes[@]}; do
    lanUP=$(echo $lang | tr '[:lower:]' '[:upper:]')
    echo -e '\t' $lan-$size
    # train, i.e. create model:
    java  -ea -Xmx2000m -cp \
      $classifiers ApLearn  \
      --train-set  $experiment/docEmbeddings/train.$lanUP$size.emb \
      --model-name $experiment/models/$lanUP$size.model \
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
  for size in ${sizes[@]}; do
      echo -e '\t' $lan1-$lan2: $size
      #test classifier of lan1 on lan2
      java  -ea -Xmx2000m -cp \
        $classifiers ApClassify \
        --test-set $experiment/docEmbeddings/test.$lan.emb \
        --model-name $experiment/models/$lanUP$size.model \
        > $experiment/results/$lan1-$lan2.$size.result &
    done
  done
done
echo done.

wait

date
FILES=$experiment/results/*
OUT=$experiment/results/allResults.txt
echo -e "src\ttar\ttopic\taccuracy">$OUT
for f in $FILES
do
  echo -n $f | sed 's/.*\/\///' | sed 's/\./\t/g'| sed 's/-/\t/' | sed 's/result//' >> $OUT
  cat $f | grep -Po '\d+.\d+' >> $OUT
done
