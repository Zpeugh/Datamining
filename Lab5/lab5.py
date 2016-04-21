import utilities

REUTERS_DIRECTORY = "../reuters"
NUM_FILES = 21

SUPPORT = '3'
CONFIDENCE = '40'

topics, words, topics_set = utilities.preprocess_data(reuters_directory=REUTERS_DIRECTORY, num_files=NUM_FILES)

results = utilities.run_ar_classifier(topics, words, topics_set, SUPPORT, CONFIDENCE, .8, 5)
