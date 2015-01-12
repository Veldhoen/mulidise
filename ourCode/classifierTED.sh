tedData=sveldhoen/mulidise/ted-cld

embeddings=myEmbeddings


classifiers=myClassifiers



english=en
languages=(de es fr it nl pb pl ro)
kinds=(test train)
topics=(art arts biology business creativity culture design economics education entertainment global health politics science technology)
labels=(negative positive)


#    lanDirs = dir()
#    lans = ['en','it','de','es','fr','nl','pb','pl','ro']
#    for lan in lans[1:]:
#        lanDirs[lan] = directory+'/'+lan+'-'+lans[0]
#    lanDirs[lans[0]]=directory+'/'+lans[0]+'-'+lans[1]



for lan in ${languages[@]}; do
    echo $lan
    for topic in ${topics[@]}; do
        echo $topic
        # preprocess: create train and test data
        #
        # train a classifier
    done
done