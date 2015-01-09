tedData=sveldhoen/mulidise/ted-cld

embeddings=myEmbeddings


classifiers=myClassifiers



english=en
languages=(de es fr it nl pb pl ro)
kinds=(test train)
topics=(art arts biology business creativity culture design economics education entertainment global health politics science technology)
labels=(negative positive)

for lan in ${languages[@]}; do
    echo $lan
    for topic in ${topics[@]}; do
        echo $topic
        # preprocess: create train and test data
        #
        # train a classifier
    done
done