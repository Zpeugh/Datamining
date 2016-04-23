import utilities

NUM_FILES = 5

SUPPORT = '2'
CONFIDENCE = '60'

topics, words, topics_set = utilities.preprocess_data(num_files=NUM_FILES)

results = utilities.run_ar_classifier(topics, words, topics_set, SUPPORT, CONFIDENCE, .666, 3)
