# Author: Zach Peugh #
#    CSE 5243 Lab2   #
#       2/22/2016     #

################ Utility Module ##############
import re
import nltk
import numpy as np

TOPICS_POSITION = 0
TITLE_POSITION = 1
BODY_POSITION = 2

# returns all content of string_to_parse between start and stop.
# returns "none" if nothing is found
def get_content_between_strings(start, stop, string_to_parse):
    content = re.search( "{0}(.*){1}".format(start, stop), string_to_parse )
    if content:
        return content.group(1)
    else:
        return "none"

def make_reuter_list_from_file( filename ):
    #Setup variables
    reuter_array = []
    current_reuter = ""
    inside_reuter = False

    #iterate through line by line and separate the file into <REUTERS></REUTERS> sections
    with open( filename ) as reuter:
        for line in reuter:
            if ("</REUTERS>" in line):
                inside_reuter = False
                reuter_array.append(current_reuter)
                current_reuter = ""
            elif inside_reuter:
                    # print("Inside")
                    current_reuter += line
            elif ("<REUTER" in line):
                inside_reuter = True
    return reuter_array


# Filters out newlines and some other markup jargon in a string.
def filter_string(str):
    str = str.replace('\n',' ')
    str = str.replace('&lt;','<')
    return str


def stringIsNumber( word ):
    try:
        float(word)
        return True
    except ValueError:
        return False


# eliminate stopwords from a list and then return the cleaned array
def remove_stopwords( wordset ):
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords = stopwords + ['&','#', '.', ',', '\'s', '\'\'', '\"', '\"\"', '\'' ,'reuter', 'reuters']

    new_wordset = []
    # stemmer = nltk.stem.porter.PorterStemmer()
    for word in wordset:
        word = word.lower()
        # word = stemmer.stem(word)
        if word not in stopwords:
            if not stringIsNumber(word):
                new_wordset.append(word)
    return new_wordset

# tokenize a string and clean it of stopwords, removing duplicates
# return a set of keywords
def tokenize_and_clean( str ):
    # use split if nltk package is not installed, or install punkt nltk package
    return list( remove_stopwords( nltk.word_tokenize( str ) ) )
    # return set( remove_stopwords( nltk.word_tokenize( str ) ) ) #remove duplicates
    # return list( remove_stopwords( str.split() ) )



# return a list of strings between <D></D> tags
def get_topics( strng ):
    strng = strng.replace("</D><D>",'~')
    strng = strng.strip("<D>")
    strng = strng.strip("</D>")
    return strng.split('~')


# Takes an array of string reuter bodies and returns a list of tuples in the form
# ( [topics], [locations], titles_string, body_string )
# where title and body are both the filtered string content between their respective tags
def get_entry_array( reuter_array ):
    tuple_array = []
    i = 0
    for reuter in reuter_array:
        reuter = filter_string(reuter)
        title = get_content_between_strings("<TITLE>", "</TITLE>", reuter)
        body = get_content_between_strings("<BODY>", "</BODY>", reuter)
        topics = get_topics(get_content_between_strings("<TOPICS>", "</TOPICS>", reuter))
        tuple_array.append( (topics, title, body) )
        i += 1
    return tuple_array

# Takes a dictionary of {word:frequency} tuples and returns a dictionary of words whose
# frequency was above the threshold of percentage seen in documents
# (i.e. 0.01 to retrieve all words occurring in 1% or more documents
def throw_out_below_frequency(dictionary, percent_occurance_lower_cutoff, percent_occurance_upper_cutoff):
    size = len(dictionary)
    lower_cutoff = percent_occurance_lower_cutoff * size
    upper_cutoff = percent_occurance_upper_cutoff * size
    sliced_dict = dict()
    while len(dictionary) > 0:
        tup = dictionary.popitem()
        word = tup[0]
        freq = tup[1]
        if freq > lower_cutoff and freq < upper_cutoff:
            sliced_dict[word] = freq

    return sliced_dict

def choose_most_likely_label(word_dict, words):
    if len(words) > 1:
        counts = dict()
        for word in words:
            counts[word] = word_dict[word]
        return sorted(counts, key=counts.get)[-1]
    else:
        return words[0]


# takes ordered lists of topic words, body words, and the full tuple array, and returns
# the final feature vector dictionary with 5 entries of equal length
# 'words_vectors' -> the body buzzwords feature vector
# 'important_words_vectors' -> the five important words from the bodies vectors
# 'topic_keyword_vectors' -> the topic keyword indices (if any) vectors
# 'topics_classes' -> the matching topics classes
def create_feature_vector(ordered_topic_words_list, sliced_ordered_body_words_list, tuple_array):
    num_documents = len(tuple_array)
    num_distinct_body_words = len( sliced_ordered_body_words_list )
    num_distinct_title_words = len( ordered_topic_words_list )
    topic_frequency_dict = dict()
    feature_vector_dict = dict()
    feature_vector_dict['words_vectors'] = []
    feature_vector_dict['topic_keyword_vectors'] = []
    feature_vector_dict['topics_classes'] = []
    feature_vector_dict['words_and_topics_vectors'] = []
    i = 0
    j = 21
    for tup in tuple_array:
        i +=1
        if i % 1000 == 0:
            j += 1
            print ("Step {0}/41".format(j) )

        important_words = dict()
        all_words = dict()
        topics = tup[TOPICS_POSITION]
        topic_keyword_vector = np.zeros(num_distinct_title_words)
        bodies_vector = np.zeros(num_distinct_body_words)

        for topic in topics:
            if topic not in topic_frequency_dict:
                topic_frequency_dict[topic] = 0
            else:
                topic_frequency_dict[topic] += 1

        for word in tup[TITLE_POSITION]:
            if word in ordered_topic_words_list:
                index = ordered_topic_words_list.index(word)
                topic_keyword_vector[index] += 1

        for word in tup[BODY_POSITION]:
            if word in sliced_ordered_body_words_list:
                index = sliced_ordered_body_words_list.index(word)
                bodies_vector[index] += 1


        feature_vector_dict['words_vectors'].append(bodies_vector)
        feature_vector_dict['topic_keyword_vectors'].append(topic_keyword_vector)
        feature_vector_dict['topics_classes'].append(topics)
        feature_vector_dict['words_and_topics_vectors'].append(np.concatenate((bodies_vector,topic_keyword_vector)))

    #TODO: find a better way to choose labels than just picking the first entry

    feature_vector_dict['topics_ints'] = [ordered_topic_words_list.index(choose_most_likely_label(topic_frequency_dict, label)) for label in feature_vector_dict['topics_classes']]


    return feature_vector_dict
