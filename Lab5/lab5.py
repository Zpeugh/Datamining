import utilities
import subprocess


REUTERS_DIRECTORY = "../reuters"
NUM_FILES = 5
IN_FILE = 'transactions.txt'
OUT_FILE = 'C:\\Users\\Zach\\documents\\github\\datamining\\lab5\\rules.txt'
CONSTRAINT_FILE = 'appearances.txt'
SUPPORT = '3'
CONFIDENCE = '40'

topics, words = utilities.preprocess_data(reuters_directory=REUTERS_DIRECTORY, num_files=NUM_FILES)

t_array = utilities.create_transactions_array(topics, words)
utilities.write_transactions(t_array, IN_FILE)
utilities.write_appearances(topics, CONSTRAINT_FILE)
open(OUT_FILE, 'w')

params = ['apriori', '-tr', '-c'+CONFIDENCE, '-s'+SUPPORT, '-R'+CONSTRAINT_FILE, IN_FILE, OUT_FILE]
subprocess.call(params)
