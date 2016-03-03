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
import time


TOPICS_POSITION = 0
PLACES_POSITION = 1
TITLE_POSITION = 2
BODY_POSITION = 3

BODY_LOWER_CUTOFF = .005
BODY_UPPER_CUTOFF = .10


def preprocessData(reuters_directory="../reuters", num_files=22):

    full_tuple_list = []
    body_word_frequency_dict = dict()
    topic_word_frequency_dict = dict()
    topics_set = set()
    NUM_SUFFIXES = ["00","01","02","03","04","05","06","07","08","09","10","12","13","14","15","16","17","18","19","20","21"]


    for i in range(num_files - 1):

        reuter_array = preprocess2.make_reuter_list_from_file( "{0}/reut2-0{1}.sgm".format(reuters_directory,NUM_SUFFIXES[i]) )
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

    # words = final_vector_dataset["words_vectors"]
    # both = final_vector_dataset["words_and_topics_vectors"]
    # keywords = final_vector_dataset["topic_keyword_vectors"]
    # topics = final_vector_dataset["topics_classes"]
    # places = final_vector_dataset["places_classes"]
    #
    # ## Remove all samples with no TOPIC label associated with them
    # indices_to_delete = []
    # for index, topic in enumerate(topics):
    #     if (topic == [''] or topic == ['none']):
    #         indices_to_delete.append(index)
    #
    # for count, index in enumerate(indices_to_delete):
    #     del topics[index - count]
    #     del both[index - count]
    #     del keywords[index - count]
    #     del places[index - count]
    #     del words[index - count]

    ## Remove all samples with no TOPIC label associated with them
    indices_to_delete = []
    for index, topic in enumerate(final_vector_dataset["topics_classes"]):
        if (topic == [''] or topic == ['none']):
            indices_to_delete.append(index)

    for count, index in enumerate(indices_to_delete):
        del final_vector_dataset["words_vectors"][index - count]
        del final_vector_dataset["words_and_topics_vectors"][index - count]
        del final_vector_dataset["topic_keyword_vectors"][index - count]
        del final_vector_dataset["topics_classes"][index - count]
        del final_vector_dataset["places_classes"][index - count]

    return final_vector_dataset


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


def accuracyScore(predicted_labels, ground_truth_labels):
    return sum([1 for i,x in enumerate(predicted_labels) if x in ground_truth_labels[i] ] ) / len(predicted_labels)


def skew(clusters):
    return np.std( [len(x) for x in clusters])


def writeFeatureVectors(feature_vector_dict):

    total_documents = len(feature_vector_dict['topics_classes'])

    if not os.path.exists('Outputs'):
        os.makedirs('Outputs')

    with open('Outputs/buzzword_vectors.txt','w') as output:
        output.writelines(str(feature_vector_dict['words_vectors'][i])+'\n' for i in range(total_documents))

    with open('Outputs/body_and_title_vectors.txt','w') as output:
        output.writelines(str(feature_vector_dict['words_and_topics_vectors'][i])+'\n' for i in range(total_documents))

    with open('Outputs/topic_keyword_vectors.txt','w') as output:
        output.writelines(str(feature_vector_dict['topic_keyword_vectors'][i])+'\n' for i in range(total_documents))

    with open('Outputs/topics_classes.txt','w') as output:
        output.writelines(str(feature_vector_dict['topics_classes'][i])+'\n' for i in range(total_documents))

    with open('Outputs/places_classes.txt','w') as output:
        output.writelines(str(feature_vector_dict['places_classes'][i])+'\n' for i in range(total_documents))

def kmeans_cluster(feature_vector, ground_truth_labels, num_means=118, metric='euclidean'):

    results = dict()
    t0 = time.clock()
    km_clusterer = Kmeans(np.array(feature_vector), k=num_means, metric=metric, maxiter=20, verbose=0)
    km_predicted = km_clusterer.clusters
    results['time'] = time.clock() - t0

    kclusters = []
    labels = np.full(num_means, dtype=object, fill_value='none')
    unique_clusters = np.unique(km_predicted)
    for cluster in unique_clusters:
        flattened_cluster, majority_label = flattenClusters( [ground_truth_labels[index] for index, x in enumerate(km_predicted) if x == cluster] )
        kclusters.append(flattened_cluster)
        labels[cluster] = majority_label

    results['clusters'] = kclusters
    results['skew'] = skew(kclusters)
    results['entropies'] = [measureEntropy(x) for x in kclusters]
    results['entropy'] = np.average( [x for i, x in enumerate(results['entropies']) if len(kclusters[i]) > 1 ] )
    results['predicted_labels'] = [labels[x] for x in km_predicted]
    results['accuracy'] = accuracyScore(results['predicted_labels'], ground_truth_labels)

    print("\n############## K-Means Clustering (%s distance) Results ##############" % (metric))
    print("Number of Means: %i\nTime to fit: %.2f seconds (%.2f minutes)." % (len(unique_clusters), results['time'] , results['time'] / 60.0) )
    print("Average cluster entropy: %.3f\nSkew: %.3f\nCorrectly classified: %.3f" % (results['entropy'], results['skew'], results['accuracy']) )

    return results

number_of_leafs = 118

############# Hierarchical ################
def hierarchical_cluster(feature_vector, ground_truth_labels, number_of_leafs=118, linkage='ward', metric='euclidean'):

    # The dictionary of results to return
    results = dict()
    # Using sklearns Hierarchical clustering method.
    if (linkage == 'ward'):
        h_clusterer = AgglomerativeClustering(n_clusters= number_of_leafs, linkage=linkage)
    else:
        h_clusterer = AgglomerativeClustering(n_clusters= number_of_leafs, linkage=linkage, affinity=metric)
    # run the Hierarchical clustering algorithm and time it.
    t0 = time.clock()
    h_predicted = h_clusterer.fit_predict(np.array(feature_vector))
    results['time'] = time.clock() - t0

    clusters = np.unique(h_predicted)

    kclusters = []
    labels = np.full(number_of_leafs, dtype=object, fill_value='none')
    for cluster in clusters:
        topics_cluster = [ground_truth_labels[index] for index, x in enumerate(h_predicted) if (cluster != -1 and x == cluster)]
        if topics_cluster:
            flattened_cluster, majority_label = flattenClusters( topics_cluster )
            kclusters.append(flattened_cluster)
            labels[cluster] = majority_label

    results['clusters'] = kclusters
    results['skew'] = skew(kclusters)
    results['entropies'] = [measureEntropy(x) for x in kclusters]
    results['entropy'] = np.average( [x for i, x in enumerate(results['entropies']) if len(kclusters[i]) > 1 ])
    results['predicted_labels'] = [labels[x] for x in h_predicted]
    results['accuracy'] = accuracyScore(results['predicted_labels'], ground_truth_labels)

    print("\n############## Hierarchical Clustering (%s linkage) Results ##############" % (linkage) )
    print("Leaf clusters: %i\nTime to fit: %.2f seconds (%.2f minutes)." % (number_of_leafs, results['time'] , results['time'] / 60.0) )
    print("Average cluster entropy: %.3f\nSkew: %.3f\nCorrectly classified: %.3f" % (results['entropy'], results['skew'],results['accuracy']) )

    return results



#
