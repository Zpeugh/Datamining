# Author: Zach Peugh #
#    CSE 5243 Lab2   #
#       2/22/2016    #
# Utility script to run clustering on preprocessed data.

import numpy as np
import preprocess3
import sklearn
import itertools
import math
import scipy
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import *

TOPICS_POSITION = 0
TITLE_POSITION = 1
BODY_POSITION = 2

BODY_LOWER_CUTOFF = .005
BODY_UPPER_CUTOFF = .10


def preprocessData(reuters_directory="/home/0/srini/WWW/674/public/reuters", num_files=21):

    full_tuple_list = []
    body_word_frequency_dict = dict()
    topic_word_frequency_dict = dict()
    topics_set = set()
    NUM_SUFFIXES = ["00","01","02","03","04","05","06","07","08","09","10","12","13","14","15","16","17","18","19","20","21"]

    for i in range(num_files):

        reuter_array = preprocess3.make_reuter_list_from_file( "{0}/reut2-0{1}.sgm".format(reuters_directory,NUM_SUFFIXES[i]) )
        split_array = preprocess3.get_entry_array( reuter_array )
        print("Step {0}/{1}".format(i+1,41))
        for article in split_array:
            title = preprocess3.tokenize_and_clean(article[TITLE_POSITION])
            body = preprocess3.tokenize_and_clean(article[BODY_POSITION])
            topics = article[TOPICS_POSITION]
            full_tuple_list.append( (topics, title, body) )
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

    sliced_body_dict = preprocess3.throw_out_below_frequency( body_word_frequency_dict, BODY_LOWER_CUTOFF, BODY_UPPER_CUTOFF )
    final_vector_dataset = preprocess3.create_feature_vector( list(topics_set), list(sliced_body_dict.keys()), full_tuple_list )

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
        del final_vector_dataset["topics_ints"][index - count]


    return final_vector_dataset


def split_vectors(feature_vector, training_split):
    split = math.floor( training_split * len(feature_vector) )
    if feature_vector.ndim == 1:
        return (feature_vector[0:split], feature_vector[split:-1])
    else:
        return (feature_vector[0:split,:], feature_vector[split:-1,:])

# returns the predicted labels based on the training labels, and indices of predicted
def get_predicted_labels(labels, predicted):
    return [labels[x] for x in predicted]


def show_confusion_matrix(true, predicted):
    matrix = confusion_matrix(true, predicted)

    plt.matshow(matrix / float(matrix.max()))
    plt.colorbar()
    plt.show()
    return True


# training_split is the number between 0 and 1 that is the percent of data you wish
# to use to train the data.
def knn_classify(feature_vector, ground_truth_labels, training_split):

    results = dict()
    training_data, predicting_data = split_vectors(np.array(feature_vector), training_split)
    training_labels, actual_labels = split_vectors(np.array(ground_truth_labels), training_split)
    nn_class = NN(n_neighbors=1)

    off_t0 = time.clock()
    nn_class.fit(training_data)
    results['offline_time'] = (time.clock() - off_t0) / float(len(training_data))


    on_t0 = time.clock()
    predicted = nn_class.kneighbors(predicting_data, return_distance=False)
    results['online_time'] = (time.clock() - on_t0) / float(len(predicting_data))

    predicted_labels = get_predicted_labels(ground_truth_labels, predicted)
    #results["accuracy"] = accuracy_score(predicted_labels, actual_labels)
    results["accuracy"] = sum([1 for i, x in enumerate(predicted_labels) if x == actual_labels[i]]) / float(len(actual_labels))
    results["precision"] = precision_score(actual_labels, predicted_labels, pos_label=None,average='weighted')

    print("\n\n#########  %i/%i Split KNN Classifer Results  #########" % (100*training_split, round(100*(1-training_split))))
    print("Offline training time: %.7f seconds" % (results["offline_time"]))
    print("Online training time: %.7f seconds" % (results["online_time"]))
    print("Prediction accuracy: %.2f%%" % (100 * results["accuracy"]))
    print("Prediction precision: %.2f%%" % (100 * results["precision"]))

    return results

def naive_bayes_classify(feature_vector, ground_truth_labels, training_split):

    results = dict()
    training_data, predicting_data = split_vectors(np.array(feature_vector), training_split)
    training_labels, actual_labels = split_vectors(np.array(ground_truth_labels), training_split)
    nb_class = GaussianNB()

    off_t0 = time.clock()
    nb_class.fit(training_data, training_labels)
    results['offline_time'] = (time.clock() - off_t0) / float(len(training_data))

    on_t0 = time.clock()
    predicted_labels = nb_class.predict(predicting_data)
    results['online_time'] = (time.clock() - on_t0) / float(len(predicting_data))

    results["accuracy"] = sum([1 for i, x in enumerate(predicted_labels) if x == actual_labels[i]]) / float(len(actual_labels))
    results["precision"] = precision_score(actual_labels, predicted_labels, pos_label=None,average='weighted')

    print("\n\n#########  %i/%i Split Naive Bayesian Classifer Results  #########" % (100*training_split, round(100*(1-training_split))))
    print("Offline training time: %.7f seconds" % (results["offline_time"]))
    print("Online training time: %.7f seconds" % (results["online_time"]))
    print("Prediction accuracy: %.2f%%" % (100 * results["accuracy"]))
    print("Prediction precision: %.2f%%" % (100 * results["precision"]))

    return results

def dtree_classify(feature_vector, ground_truth_labels, training_split):

    results = dict()
    training_data, predicting_data = split_vectors(np.array(feature_vector), training_split)
    training_labels, actual_labels = split_vectors(np.array(ground_truth_labels), training_split)
    dt_class = DTC()

    off_t0 = time.clock()
    dt_class.fit(training_data, training_labels)
    results['offline_time'] = (time.clock() - off_t0) / float(len(training_data))

    on_t0 = time.clock()
    predicted_labels = dt_class.predict(predicting_data)
    results['online_time'] = (time.clock() - on_t0) / float(len(predicting_data))

    results["accuracy"] = sum([1 for i, x in enumerate(predicted_labels) if x == actual_labels[i]]) / float(len(actual_labels))
    results["precision"] = precision_score(actual_labels, predicted_labels, pos_label=None,average='weighted')

    print("\n\n#########  %i/%i Split Decision Tree Classifer Results  #########" % (100*training_split, round(100*(1-training_split))))
    print("Offline training time: %.7f seconds" % (results["offline_time"]))
    print("Online training time: %.7f seconds" % (results["online_time"]))
    print("Prediction accuracy: %.2f%%" % (100 * results["accuracy"]))
    print("Prediction precision: %.2f%%" % (100 * results["precision"]))


    return results
