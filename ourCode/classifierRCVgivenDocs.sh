benno=/home/bkruit/mulidise
sara=/home/sveldhoen/mulidise

experiment=$1
docEmbeddings=$2

languages=(en de)

mkdir -p $experiment/models/RCV
mkdir -p $experiment/results/RCV


classifiers=$benno/document-representations/bin

wait
date
echo training models...

sizes=(100 200 500 1000 5000 10000)
for lan in ${languages[@]}; do
  for lan2 in ${languages[@]};do
    for size in ${sizes[@]}; do
      lanUP=$(echo $lan | tr '[:lower:]' '[:upper:]')
      echo -e '\t' $lan-$size
      # train, i.e. create model:
      java  -ea -Xmx2000m -cp \
        $classifiers ApLearn  \
        --train-set  $docEmbeddings/train.$lan2-$lanUP$size.$lan \
        --model-name $experiment/models/RCV/$lan.$size.model \
        --epoch-num 10 &
    done
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
        --test-set $docEmbeddings/test.$lan1-$lan2.$lan1 \
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
  echo -n $f | sed 's/.*\/\///' | sed 's/\./\t/g'| sed 's/-/\t/' | sed 's/result//g' >> $OUT
  numbers=`cat $f | grep -Po '\d+\.\d+'`
  echo $numbers | sed 's/ /\t/g' >> $OUT
done
