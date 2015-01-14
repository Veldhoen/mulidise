benno=/~/bkruit/mulidise
sara=/~/sveldhoen/mulidise

experiment=$benno/experiment1
embeddings=$benno.document-representations/data/embeddings/original-de-en.en

tedDocs=$sara/ted-cld
tedIDFs=$sara/idfsTED
classifiers=benno/document-representations/bin

# preprocess data
python preprocessData.py \
       -d $tedData
       -e $embeddings
       -o $experiment/docEmbeddings
       -i $tedIDFs

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
    echo $lan1-$lan2 >> $experiment/output/results.txt
    for topic in ${topics[@]}; do
      echo $topic >> $experiment/output/results.txt
      #test classifier of lan1 on lan2
      java  -ea -Xmx2000m -cp \
        $classifiers ApClassify
        --test-set $experiment/docEmbeddings/$lan2/test.$topic.emb \
        --model-name $experiment/models/$lan.$topic.model \
        >> $experiment/output/results.txt
    done
  done
done
