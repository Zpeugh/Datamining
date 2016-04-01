#!/usr/bin/env python
import preprocess4
import utilities
import time

# REUTERS_DIRECTORY = "/home/0/srini/WWW/674/public/reuters"
REUTERS_DIRECTORY = "../reuters"

# Change num_files to a sample size of between 1-21 reuters files
tuple_list = preprocess4.preprocess_data(reuters_directory=REUTERS_DIRECTORY, num_files=1)

shingles_list = []
topics_list = []

for tup in tuple_list:
    shingles_list.append( tup[1] )
    topics_list.append( tup[0] )


############################ K = 16 ############################
def minHashK(k)
    signatures = utilities.minHash(shingles_list,k=40)
    t_0 = time.time()
    jacc_sims  = utilities.get_jaccard_similarities(shingles_list)
    print("Jaccard similarity time to complete for k=40: %.4f seconds" % (time.time() - t_0) )
    t_0 = time.time()
    est_sims = utilities.get_estimated_similarities(signatures)
    print("Estimated similarity time to complete for k=40: %.4f seconds" % (time.time() - t_0) )
    msqe = utilities.mean_squared_error(est_sims, jacc_sims)
    print("The mean squared error of the estimation for k=40: %.6f" % msqe )
