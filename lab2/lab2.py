# Author: Zach Peugh #
#    CSE 5243 Lab2   #
#       2/22/2016     #

################ Runnable Script ##############
import numpy as np
import preprocess2
from sklearn.cluster import *
import itertools
import math
import scipy
from k_means import Kmeans


TOPICS_POSITION = 0
PLACES_POSITION = 1
TITLE_POSITION = 2
BODY_POSITION = 3

BODY_LOWER_CUTOFF = .005
BODY_UPPER_CUTOFF = .10
# REUTERS_DIRECTORY = "/home/0/srini/WWW/674/public/reuters"
REUTERS_DIRECTORY = "../reuters"


full_tuple_list = []
body_word_frequency_dict = dict()
topic_word_frequency_dict = dict()
topics_set = set()
NUM_SUFFIXES = ["00","01","02","03","04","05","06","07","08","09","10","12","13","14","15","16","17","18","19","20","21"]

for i in range(0, len(NUM_SUFFIXES)):
# for i in range(6):

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
both = final_vector_dataset["words_and_topics_vectors"]
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
    del both[index - count]
    del keywords[index - count]
    del places[index - count]
    del words[index - count]



def flattenClusters(clusters):

    flattened_clusters = list(itertools.chain(*clusters))
    frequency_dict = dict()
    final_list = []
    for label in flattened_clusters:
        if (label in frequency_dict):
            frequency_dict[label] += 1
        else:
            frequency_dict[label] = 1

    majority_label = sorted(frequency_dict, key=frequency_dict.get)[-1]
    for cluster in clusters:
        if ( len(cluster) > 1 ):
            freq_list = [frequency_dict[x] for x in cluster]
            final_list.append( cluster[freq_list.index(max(freq_list))] )
        else:
            final_list.append( cluster[0] )


    return (final_list, majority_label)

def measureEntropy(clusters):

    num_labels = len(clusters)

    frequency_dict = dict()
    for label in clusters:
        if (label in frequency_dict):
            frequency_dict[label] += 1
        else:
            frequency_dict[label] = 1

    if (num_labels <= 1):
        return 0

    counts = np.array(list(frequency_dict.values()))
    probabilities = counts / num_labels

    class_count = np.count_nonzero(probabilities)
    if (class_count <= 1):
        return 0

    entropy = 0.0

    # Compute standard entropy.
    for i in probabilities:
        # entropy -= (i * math.log(i, 2) ) / math.log(num_labels, 2)
        entropy -= i * math.log(i, num_labels)

    return entropy

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
entropies = []
weightedEntropies = []
sumEnts = []
sumWeightedEnts = []
all_clusters = []
num_means = 118


#######################JUST BODY WORDS###################
distance_measures = ['euclidean', 'cosine']

for i, metric_name in enumerate(distance_measures):

    km_clusterer = Kmeans(np.array(words), k=num_means, metric=metric_name, maxiter=20)
    km_predicted = km_clusterer.clusters

    kclusters = []
    labels = np.full(118, dtype=object, fill_value='none')
    for cluster in np.unique(km_predicted):
        flattened_cluster, majority_label = flattenClusters( [topics[index] for index, x in enumerate(km_predicted) if x == cluster] )
        kclusters.append(flattened_cluster)
        labels[cluster] = majority_label

    all_clusters.append(kclusters)
    num_clusters = len(words)
    weightedEntropies.append( [measureEntropy(x) * (len(x) / num_clusters) for x in kclusters] )
    entropies.append( [measureEntropy(x) for x in kclusters] )
    sumEnts.append( np.average( [x for x in entropies[i] if(x > 0 or len(kclusters[i]) > 2) ]) )

    predicted_labels = [labels[x] for x in km_predicted]
    correctly_classified = sum([1 for i,x in enumerate(predicted_labels) if x in topics[i] ] ) / len(labels)
    # sumWeightedEnts.append(  )

    print( "%s distance:    entropy: %f    correctly_classified: %f" % (metric_name, sumEnts[i], correctly_classified ) )




###############TOPICS AND BODY##################
# distance_measures = ['euclidean', 'cosine']
#
# for i, metric_name in enumerate(distance_measures):
#
#     km_clusterer = Kmeans(np.array(both), k=num_means, metric=metric_name, maxiter=20)
#     km_predicted = km_clusterer.clusters
#
#     kclusters = []
#     labels = []
#     for cluster in range(num_means):
#         flattened_cluster, majority_label = flattenClusters( [topics[index] for index, x in enumerate(km_predicted) if x == cluster] )
#         kclusters.append(flattened_cluster)
#         labels.append(majority_label)
#
#     all_clusters.append(kclusters)
#     num_clusters = len(both)
#     weightedEntropies.append( [measureEntropy(x) * (len(x) / num_clusters) for x in kclusters] )
#     entropies.append( [measureEntropy(x) for x in kclusters] )
#     sumEnts.append( np.average( [x for x in entropies[i] if(x > 0 or len(kclusters[i]) > 2) ]) )
#
#     predicted_labels = [labels[x] for x in km_predicted]
#     correctly_classified = sum([1 for i,x in enumerate(predicted_labels) if x in topics[i] ] ) / len(labels)
#     # sumWeightedEnts.append(  )
#
#     print( "%s distance:    entropy: %f    correctly_classified: %f" % (metric_name, sumEnts[i], correctly_classified ) )
#
#
#




#
