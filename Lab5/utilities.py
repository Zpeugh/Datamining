# Author: Zach Peugh #
#    CSE 5243 Lab2   #
#       2/22/2016    #
# Utility script to run clustering on preprocessed data.

import numpy as np
import preprocess5
import itertools
import math
import time
import matplotlib.pyplot as plt
import subprocess
from operator import itemgetter

TOPICS_POSITION = 0
BODY_POSITION = 1

BODY_LOWER_CUTOFF = .005
BODY_UPPER_CUTOFF = .10

IN_FILE = 'transactions.txt'
OUT_FILE = 'C:\\Users\\Zach\\documents\\github\\datamining\\lab5\\rules.txt'
CONSTRAINT_FILE = 'appearances.txt'

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

    return ( topics, keywords, topics_set )


def make_transaction_string(keyword, words):
    return " ".join(words) + " " + keyword


def create_transactions_array(class_vector, keyword_vector):

    transaction_array = []

    for i, classes in enumerate(class_vector):
        if len(classes) == 1:
            transaction_array.append(make_transaction_string(classes[0], keyword_vector[i]))
        elif len(class_vector) > 1:
            for cl in classes:
                transaction_array.append(make_transaction_string(cl, keyword_vector[i]))

    return transaction_array


def write_transactions(t_array, filename):
    with open(filename,'w') as output:
        output.writelines( t+'\n' for t in t_array)


def write_appearances(topics, filename):
    lines = ['antecedent\n']
    lines += [topic + ' consequent\n' for topic in topics if topic]
    with open(filename,'w') as output:
        output.writelines( line for line in lines )


def split_vectors(feature_vector, training_split):
    split = int( math.floor( training_split * len(feature_vector) ) )
    return (feature_vector[0:split], feature_vector[split:-1])

# Takes a line from the rules output file and returns
# [class, {word1, word2, word2, ... wordn}, support, confidence]
#
def split_line(line):
    rule = []
    line = line.split('<-', 1)
    rule.append(line[0].strip())                        #class
    line = line[1].split('(', 1)
    rule.append( set( (line[0].strip()).split(" ")) )   #words set
    nums = line[1].split(', ')
    rule.append( float(nums[0]) )                       #support
    rule.append( float(nums[1][:-2]) )                  #confidence
    return rule

#  Returns the array of all association rules.  Each entry in the array looks like
#  [class, [word1, word2, word2], support, confidence]
def read_rules_from_file():
    rules = []
    with open( OUT_FILE ) as rule_file:
        for line in rule_file:
            rules.append( split_line(line) )
    return rules

# Takes an array of rules and orders them based on confidence, then support
def order_rules(rules_array):
    return sorted(rules_array, key=itemgetter(3, 2), reverse=True)

# Given the feature vectors, create all of the association rules and return the
# array of them for use in classification.
def train_AR_classifier(topics, words, topics_set, support, confidence):

    t_array = create_transactions_array(topics, words)
    write_transactions(t_array, IN_FILE)
    write_appearances(topics_set, CONSTRAINT_FILE)
    fo = open(OUT_FILE, 'w')
    params = ['apriori', '-tr', '-c'+str(confidence), '-s'+str(support), '-R'+CONSTRAINT_FILE, IN_FILE, OUT_FILE]
    subprocess.call(params)
    fo.close()

    return order_rules(read_rules_from_file())

def jaccard_sim( s1, s2 ):
    if (len(s1) > 0):
        return len(s1.intersection(s2)) / float( len(s1.union(s2)) )
    else:
        return 0

#takes lists of predicted and actual topics of the same length and
#indices.  Then returns the average jaccard similarity of each set
def accuracy(pred, actual):
    accuracies = 0
    for i, p in enumerate(pred):
        if not p.isdisjoint(actual[i]):
            accuracies += jaccard_sim(p, actual[i])
    return accuracies / float( len(actual) )


# Takes a testing set of words and returns the association rule classifiers most confident guess
# for each data point
def predict_classes(words_sets, rules, k):

    default_rule = rules[0][0]
    predicted_classes = []
    for i, word_set in enumerate(words_sets):
        predicted_classes.append( set() )
        current_k = 0
        for rule in rules:
            if rule[1].issubset(word_set) and current_k < k:
                predicted_classes[i].add( rule[0] )
                current_k +=1
        if current_k == 0:
            predicted_classes[i].add( default_rule )

    return predicted_classes

def run_ar_classifier(topics, words, topics_set, support, confidence, t_split, k):

    results = dict()

    training_words, testing_words = split_vectors(words, t_split)
    training_topics, testing_topics = split_vectors(topics, t_split)

    t0 = time.time()
    rules = train_AR_classifier(training_topics, training_words, topics_set, support, confidence)
    results["offline_time"] = (time.time() - t0) / float(len(training_topics))

    t0 = time.time()
    predicted_topics = predict_classes(testing_words, rules, k)
    results["online_time"] = (time.time() - t0) / float(len(testing_topics))

    results["accuracy"] = accuracy(predicted_topics, testing_topics)

    print("\n\n#########  %i/%i Split AR Classifer Results  #########" % (100*t_split, round(100*(1-t_split))))
    print("Offline training time: %.7f seconds" % (results["offline_time"]))
    print("Online training time: %.7f seconds" % (results["online_time"]))
    print("Prediction accuracy: %.2f%%" % (100 * results["accuracy"]))

    return results
#
