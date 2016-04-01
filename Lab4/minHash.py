#!/usr/bin/env python
import preprocess4
import utilities
import time

# REUTERS_DIRECTORY = "/home/0/srini/WWW/674/public/reuters"
REUTERS_DIRECTORY = "../reuters"

# Change num_files to a sample size of between 1-21 reuters files
tuple_list = preprocess4.preprocess_data(reuters_directory=REUTERS_DIRECTORY)

shingles_list = []
topics_list = []

for tup in tuple_list:
    shingles_list.append( tup[1] )
    topics_list.append( tup[0] )


def minHash_k( k, actual_sims ):
    print("\n######################### K = %d #########################" % k)
    t_0 = time.time()
    signatures = utilities.minHash(shingles_list,k=k)
    est_sims = utilities.get_estimated_similarities(signatures)
    print("Estimated similarity time: %.4f seconds" % (time.time() - t_0) )
    msqe = utilities.mean_squared_error(est_sims, actual_sims)
    print("Mean squared error: %.6f" % msqe )



############################ K = 16 ############################
print("\n######################### Jaccard Similarity time #########################")
t_0 = time.time()
jacc_sims  = utilities.get_jaccard_similarities(shingles_list)
print("Jaccard similarity time: %.4f seconds" % (time.time() - t_0) )

minHash_k(16, jacc_sims)
minHash_k(32, jacc_sims)
# minHash_k(64, jacc_sims)
# minHash_k(128, jacc_sims)
# minHash_k(256, jacc_sims)
