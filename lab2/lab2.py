# Author: Zach Peugh #
#    CSE 5243 Lab2   #
#       2/22/2016     #

################ Runnable Script ##############
import numpy as np
import preprocess2
from sklearn.cluster import *

TOPICS_POSITION = 0
PLACES_POSITION = 1
TITLE_POSITION = 2
BODY_POSITION = 3

BODY_LOWER_CUTOFF = .005
BODY_UPPER_CUTOFF = .10
# REUTERS_DIRECTORY = "/home/0/srini/WWW/674/public/reuters"
REUTERS_DIRECTORY = "reuters/"


full_tuple_list = []
body_word_frequency_dict = dict()
topic_word_frequency_dict = dict()
topics_set = set()
NUM_SUFFIXES = ["00","01","02","03","04","05","06","07","08","09","10","12","13","14","15","16","17","18","19","20","21"]

for i in range(0, len(NUM_SUFFIXES)):
# for i in range(2):

    reuter_array = preprocess2.make_reuter_list_from_file( "{0}/reut2-0{1}.sgm".format(REUTERS_DIRECTORY,NUM_SUFFIXES[i]) )
    split_array = preprocess2.get_entry_array( reuter_array )
    print("Step {0}/{1}".format(i+1,41))
    for article in split_array:
        title = preprocess2.tokenize_and_clean(article[TITLE_POSITION])
        body = preprocess2.tokenize_and_clean(article[BODY_POSITION])
        topics = article[TOPICS_POSITION]
        places = article[PLACES_POSITION]
        full_tuple_list.append( (topics, places, title, body) )
        for topic in topics:
            topics_set.add(topic)
            if topic in topic_word_frequency_dict:
                topic_word_frequency_dict[topic] += 1
            else:
                topic_word_frequency_dict[topic] = 1
        for word in body:
            if word in body_word_frequency_dict:
                body_word_frequency_dict[word] += 1
            else:
                body_word_frequency_dict[word] = 1

sliced_body_dict = preprocess2.throw_out_below_frequency( body_word_frequency_dict, BODY_LOWER_CUTOFF, BODY_UPPER_CUTOFF )

final_vector_dataset = preprocess2.create_feature_vector( list(topics_set), list(sliced_body_dict.keys()), full_tuple_list )

words = final_vector_dataset["words_vectors"]
fives = final_vector_dataset["important_words_vectors"]
keywords = final_vector_dataset["topic_keyword_vectors"]
topics = final_vector_dataset["topics_classes"]
places = final_vector_dataset["places_classes"]


## Remove all samples with no TOPIC label associated with them
indices_to_delete = []
for index, topic in enumerate(topics):
    if (topic == ['']):
        indices_to_delete.append(index)

for count, index in enumerate(indices_to_delete):
    del topics[index - count]
    del fives[index - count]
    del keywords[index - count]
    del places[index - count]
    del words[index - count]



## returns true if the two cluster arrays have at least 1 term in common
def equalClusters(cluster1, cluster2):
    for x in cluster1:
        for z in cluster2:
            if (x == z):
                return True
    return False


def measureEntropy(clusters):
    for cluster in clusters:

        if (equalClusters(cluster)):
            return True



#
# total_documents = len(words)
#
# with open('Outputs/buzzword_vectors.txt','w') as output:
#     output.writelines(str(words[i])+'\n' for i in range(total_documents))
#
# with open('Outputs/important_words_vectors.txt','w') as output:
#     output.writelines(str(fives[i])+'\n' for i in range(total_documents))
#
# with open('Outputs/topic_keyword_vectors.txt','w') as output:
#     output.writelines(str(keywords[i])+'\n' for i in range(total_documents))
#
# with open('Outputs/topics_classes.txt','w') as output:
#     output.writelines(str(topics[i])+'\n' for i in range(total_documents))
#
# with open('Outputs/places_classes.txt','w') as output:
#     output.writelines(str(places[i])+'\n' for i in range(total_documents))


means = len(topics_set) - 30
km_clusterer = KMeans(n_clusters=means)
km_clusterer.fit(words)
km_predicted = km_clusterer.predict(words)

kclusters = []
for cluster in range(means):
    kclusters.append( [topics[index] for index, x in enumerate(km_predicted) if x == cluster] )
