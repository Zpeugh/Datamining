import utilities

REUTERS_DIRECTORY = "../reuters"
NUM_FILES = 5

SUPPORT = '2'
CONFIDENCE = '60'

topics, words, topics_set = utilities.preprocess_data(reuters_directory=REUTERS_DIRECTORY, num_files=NUM_FILES)

results = utilities.run_ar_classifier(topics, words, topics_set, SUPPORT, CONFIDENCE, .666, 3)
