benno=/home/bkruit/mulidise
sara=/home/sveldhoen/mulidise

experiment=$1

#languages=(en de es fr it nl pb pl ro)
languages=(en)
topics=(art arts biology business creativity culture design economics education entertainment global health politics science technology)

mkdir $experiment/models
mkdir $experiment/results

classifiers=$benno/document-representations/bin

date
echo training models...

for lan in ${languages[@]}; do
  for topic in ${topics[@]}; do
    echo -e '\t' $lan-$topic
    # train, i.e. create model:
    java  -ea -Xmx2000m -cp \
      $classifiers ApLearn  \
      --train-set  $experiment/docEmbeddings/$lan/train/$topic \
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
        $classifiers ApClassifyF1 \
        --test-set $experiment/docEmbeddings/$lan2/test/$topic \
        --model-name $experiment/models/$lan1.$topic.model \
        > $experiment/results/$lan1-$lan2.$topic.result &
    done
  done
done
echo done.

wait

date
FILES=$experiment/results/*
OUT=$experiment/results/allResults.txt
echo -e "train\ttest\ttopic\taccuracy\tprecision\trecall\tF1">$OUT
for f in $FILES
do
  echo -n $f | sed 's/.*\///' | sed 's/\./\t/g'| sed 's/-/\t/' | sed 's/result//g' >> $OUT
  numbers=`cat $f | grep -Po '\d+\.\d+'`
  echo $numbers | sed 's/ /\t/g' >> $OUT
done
