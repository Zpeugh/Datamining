#!/usr/bin/env python
import preprocess4
import utilities
import time
import matplotlib.pyplot as plt
# import marshal

# REUTERS_DIRECTORY = "/home/0/srini/WWW/674/public/reuters"
REUTERS_DIRECTORY = "../reuters"

# Change num_files to a sample size of between 1-21 reuters files
tuple_list = preprocess4.preprocess_data(num_files=6, reuters_directory=REUTERS_DIRECTORY)

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
    t_elapsed = time.time() - t_0
    print("Estimated similarity time: %.4f seconds" % (t_elapsed) )
    msqe = utilities.mean_squared_error(est_sims, actual_sims)
    print("Mean squared error: %.6f" % msqe )
    return t_elapsed, msqe



print("\n######################### Actual Jaccard Similarity #########################")
t_0 = time.time()
jacc_sims  = utilities.get_jaccard_similarities(shingles_list)
print("Jaccard similarity time: %.4f seconds" % (time.time() - t_0) )


t1, e1 = minHash_k(2, jacc_sims)
t2, e2 = minHash_k(5, jacc_sims)
t3, e3 = minHash_k(10, jacc_sims)
t4, e4 = minHash_k(15, jacc_sims)
t5, e5 = minHash_k(20, jacc_sims)
t6, e6 = minHash_k(30, jacc_sims)
t7, e7 = minHash_k(50, jacc_sims)



ks = [2, 5, 10, 15, 20, 30, 50]
times = [t1, t2, t3, t4, t5, t6, t7]
errors = [e1, e2, e3, e4, e5, e6, e7]

plt.figure(1)
plt.plot(ks, times)
plt.xlabel('K signatures')
plt.ylabel('Time (seconds)')
plt.show()

plt.clf
plt.figure(2)
plt.plot(ks, errors)
plt.xlabel('K signatures')
plt.ylabel('Mean Squared Errors')
plt.show()

# t1, e1 = minHash_k(16, jacc_sims)
# t2, e2 = minHash_k(32, jacc_sims)
# t3, e3 = minHash_k(64, jacc_sims)
# t4, e4 = minHash_k(128, jacc_sims)
# t5, e5 = minHash_k(256, jacc_sims)
#
# ks = [16, 32, 64, 128, 256]
# times = [t1, t2, t3, t4, t5]
# errors = [e1, e2, e3, e4, e5]
#
# plt.figure(1)
# plt.plot(ks, times)
# plt.xlabel('K signatures')
# plt.ylabel('Time (seconds)')
# plt.show()
#
# plt.figure(2)
# plt.plot(ks, errors)
# plt.xlabel('K signatures')
# plt.ylabel('Mean Squared Errors')
# plt.show()
