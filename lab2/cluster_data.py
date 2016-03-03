import utilities


# REUTERS_DIRECTORY = "/home/0/srini/WWW/674/public/reuters"

# Change num_files to a sample size of between 1-21 reuters files
vector_dataset = utilities.preprocessData(num_files=5)

just_body = vector_dataset["words_vectors"]
topic_and_body = vector_dataset["words_and_topics_vectors"]
ground_truth_labels = vector_dataset["topics_classes"]

#metric can be any in scipy.spatial.distance module
# i.e.  'minkowski', 'euclidean', 'dice', 'jaccard', 'cosine', etc.
# num_means should be < 50 for a sample of under 8 files or an error may be thrown
k_results = lab2.kmeans_cluster(topic_and_body, ground_truth_labels, num_means=40, metric='euclidean')
# k_results = lab2.kmeans_cluster(just_body, ground_truth_labels, num_means=40, metric='euclidean')




# linkage can be 'complete','average' 'ward' (only for euclidean)
# metric can be 'cosine', 'manhatten', 'euclidean'
h_results = utilities.hierarchical_cluster(topic_and_body, ground_truth_labels, number_of_leafs=40, linkage='complete', metric='euclidean')
# h_results = utilities.hierarchical_cluster(just_body, ground_truth_labels, number_of_leafs=leafs, linkage='complete', metric='euclidean')
