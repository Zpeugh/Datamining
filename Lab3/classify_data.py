import utilities


# REUTERS_DIRECTORY = "/home/0/srini/WWW/674/public/reuters"
REUTERS_DIRECTORY = "../reuters"

# Change num_files to a sample size of between 1-21 reuters files
vector_dataset = utilities.preprocessData(reuters_directory=REUTERS_DIRECTORY, num_files=21)

just_body = vector_dataset["words_vectors"]
topic_and_body = vector_dataset["words_and_topics_vectors"]
ground_truth_labels = vector_dataset["topics_classes"]
int_labels = vector_dataset["topics_ints"]


utilities.knn_classify(topic_and_body, int_labels, 0.5)
utilities.naive_bayes_classify(topic_and_body, int_labels, 0.5)
utilities.dtree_classify(topic_and_body, int_labels, 0.5)


utilities.knn_classify(topic_and_body, int_labels, 0.66)
utilities.naive_bayes_classify(topic_and_body, int_labels, 0.66)
utilities.dtree_classify(topic_and_body, int_labels, 0.66)


utilities.knn_classify(topic_and_body, int_labels, 0.8)
utilities.naive_bayes_classify(topic_and_body, int_labels, 0.8)
utilities.dtree_classify(topic_and_body, int_labels, 0.8)
