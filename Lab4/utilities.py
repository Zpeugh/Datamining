#!/usr/bin/env python

# Utility script for minHashing on preprocessed data.

import preprocess4
import time
import random
import matplotlib.pyplot as plt
import itertools

PRIME_MOD = 4294967311
MAX_HASH_BIN = 2**32-1

def jaccard_sim( s1, s2 ):
    if (len(s1) > 0):
        return len(s1.intersection(s2)) / float( len(s1.union(s2)) )
    else:
        return 0

def estimated_sim( s1, s2 ):
    count = 0
    for sig in s1:
        count = count + (sig in s2)

    return count / float(len(s1))


#Returns n unique random coefficients between 0-MAX_HASH_BIN
def random_coefficients(n):

    coeff_set = set()
    while len(coeff_set) < n:
        coeff_set.add(random.randint(0,MAX_HASH_BIN))

    return list(coeff_set)


def min_shingle_hash(shingles, a, b):
    minHash = MAX_HASH_BIN
    for shingle in shingles:
        hashCode = (a* shingle + b) % PRIME_MOD
        if hashCode < minHash:
            minHash = hashCode
    return minHash


# return the ordered list of n documents with k signatures each to be compared
def minHash(document_list, k=16):

    coeffs = random_coefficients(k + 1)
    signatures_list = []

    for doc in document_list:
        signatures = []
        for i in range(k):
            a = coeffs[i]
            b = coeffs[i+1]
            signatures.append(min_shingle_hash(doc, a, b))
        signatures_list.append(signatures)

    return signatures_list


def get_jaccard_similarities(document_sets):
    num_docs = len(document_sets)
    jacc_sims = []
    binary_tuples = list(itertools.combinations([i for i in range(num_docs)], 2))

    for i, tup in enumerate(binary_tuples):
            ds1 = document_sets[tup[0]]
            ds2 = document_sets[tup[1]]
            jacc_sims.append( jaccard_sim(ds1, ds2) )

    return jacc_sims


def get_estimated_similarities(document_sets):
    num_docs = len(document_sets)
    est_sims = []
    binary_tuples = list(itertools.combinations([i for i in range(num_docs)], 2))

    for i, tup in enumerate(binary_tuples):
            ds1 = document_sets[tup[0]]
            ds2 = document_sets[tup[1]]
            est_sims.append( estimated_sim(ds1, ds2) )

    return est_sims


def mean_squared_error(estimated, actual):
    return sum( [(x - actual[i])**2 for i, x in enumerate(estimated)] ) / float(len(estimated))






    #
