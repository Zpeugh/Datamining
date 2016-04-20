# Author: Zach Peugh #
#    CSE 5243 Lab2   #
#       2/22/2016    #
# Utility script to run clustering on preprocessed data.

import numpy as np
import preprocess5
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
BODY_POSITION = 1

BODY_LOWER_CUTOFF = .005
BODY_UPPER_CUTOFF = .10


def preprocess_data(reuters_directory="/home/0/srini/WWW/674/public/reuters", num_files=21):

    full_tuple_list = []
    body_word_frequency_dict = dict()
    topics_set = set()
    NUM_SUFFIXES = ["00","01","02","03","04","05","06","07","08","09","10","12","13","14","15","16","17","18","19","20","21"]

    for i in range(num_files):

        reuter_array = preprocess5.make_reuter_list_from_file( "{0}/reut2-0{1}.sgm".format(reuters_directory,NUM_SUFFIXES[i]) )
        split_array = preprocess5.get_entry_array( reuter_array )
        print("Step {0}/{1}".format(i+1,41))
        for article in split_array:
            body = preprocess5.tokenize_and_clean(article[BODY_POSITION])
            topics = article[TOPICS_POSITION]
            full_tuple_list.append( (topics, body) )
            for topic in topics:
                if topic:
                    topics_set.add(topic)
            for word in body:
                if word in body_word_frequency_dict:
                    body_word_frequency_dict[word] += 1
                else:
                    body_word_frequency_dict[word] = 1

    sliced_body_dict = preprocess5.throw_out_below_frequency( body_word_frequency_dict, BODY_LOWER_CUTOFF, BODY_UPPER_CUTOFF )
    final_vector_dataset = preprocess5.create_feature_vector( list(topics_set), list(sliced_body_dict.keys()), full_tuple_list )

    topics = final_vector_dataset["topics_classes"]
    keywords = final_vector_dataset["words_vectors"]
    ## Remove all samples with no TOPIC labels associated with them
    indices_to_delete = []
    for i, topic in enumerate(topics):
        if (topic == [''] or keywords[i] == ['none']):
            indices_to_delete.append(i)

    for count, index in enumerate(indices_to_delete):
        del topics[index - count]
        del keywords[index - count]

    return ( topics, keywords )


def make_transaction_string(words, topics):
    if len(topics) > 1:
        return " ".join(words) + " " + "[" + ",".join(topics) + "]"
    elif len(topics) == 0:
        return " ".join(words) + " " + topics[0]

def create_transactions_array(class_vector, keyword_vector):

    transaction_array = []

    for i, classes in enumerate(class_vector):
        transaction_array.append(make_transaction_string(keyword_vector[i], classes))

    return transaction_array

def write_transactions(t_array, filename):
    with open(filename,'w') as output:
        output.writelines( t+'\n' for t in t_array if t)


def write_appearances(class_vector, filename):
    lines = ['antecedent\n']
    topics_set = set()

    for topics in class_vector:
        topics_set.add("[" + ",".join(topics) + "]")

    lines += [topic + ' consequent\n' for topic in topics_set]
    with open(filename,'w') as output:
        output.writelines( lines )


def split_vectors(feature_vector, training_split):
    split = math.floor( training_split * len(feature_vector) )
    if feature_vector.ndim == 1:
        return (feature_vector[0:split], feature_vector[split:-1])
    else:
        return (feature_vector[0:split,:], feature_vector[split:-1,:])


#
# print("\n\n#########  %i/%i Split Decision Tree Classifer Results  #########" % (100*training_split, round(100*(1-training_split))))
# print("Offline training time: %.7f seconds" % (results["offline_time"]))
# print("Online training time: %.7f seconds" % (results["online_time"]))
# print("Prediction accuracy: %.2f%%" % (100 * results["accuracy"]))
# print("Prediction precision: %.2f%%" % (100 * results["precision"]))
#
