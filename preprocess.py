import re
import nltk
import numpy as np

TOPICS_POSITION = 0
PLACES_POSITION = 1
TITLE_POSITION = 2
BODY_POSITION = 3


#put this pointer in document: /home/0/peugh/5243/lab1
#set permissions

#
# #Load in all of the reuters from the archive
# for i in NUM_SUFFIXES:
#     url = "https://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-0{0}.sgm".format(i)
#     print url
#     urllib.urlretrieve (url, "reuters/{0}.sgm".format(i) )
#

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

# eliminate stopwords from a list and then return the cleaned array
def remove_stopwords( wordset ):
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords.append('&')
    stopwords.append('#')
    new_wordset = []
    # stemmer = nltk.stem.porter.PorterStemmer()
    for word in wordset:
        word = word.lower()
        # word = stemmer.stem(word)
        if word not in stopwords:
            new_wordset.append(word)
    return new_wordset

# tokenize a string and clean it of stopwords, removing duplicates
# return a set of keywords
def tokenize_and_clean( str ):
    return list( remove_stopwords( nltk.word_tokenize( str ) ) )
    # return set( remove_stopwords( nltk.word_tokenize( str ) ) )


# return a list of strings between <D></D> tags
def get_topics_or_places( strng ):
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
        topics = get_topics_or_places(get_content_between_strings("<TOPICS>", "</TOPICS>", reuter))
        places = get_topics_or_places(get_content_between_strings("<PLACES>", "</PLACES>", reuter))
        tuple_array.append( (topics, places, title, body) )
        i += 1
    return tuple_array

# Takes a dictionary of {word:frequency} tuples and returns a dictionary of words whose
# frequency was above the threshold of percentage seen in documents
# (i.e. 0.01 to retrieve all words occurring in 1% or more documents
def throw_out_below_frequency(dictionary, percent_occurance_lower_cutoff, percent_occurance_upper_cutoff):
    size = len(dictionary)
    print("THE LENGTH OF THE ORIGINAL DICTIONARY IS %i" % (size))
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


def create_feature_vector(ordered_topic_words_list, ordered_body_words_list, tuple_array):
    num_documents = len(tuple_array)
    num_distinct_body_words = len( ordered_body_words_list )
    num_distinct_title_words = len( ordered_topic_words_list )
    feature_vector_dict = dict()
    feature_vector_dict['words_vectors'] = []
    feature_vector_dict['topic_keyword_vectors'] = []
    feature_vector_dict['topics_classes'] = []
    feature_vector_dict['places_classes'] = []
    i = 0
    for tup in tuple_array:
        i +=1
        if i % 1000 == 0:
            print ("{0}%".format(int((i / num_documents) * 100) ) )
        topics = tup[TOPICS_POSITION]
        places = tup[PLACES_POSITION]
        topic_keyword_vector = []
        bodies_vector = []

        for word in tup[TITLE_POSITION]:
            if word in ordered_topic_words_list:
                index = ordered_topic_words_list.index(word)
                topic_keyword_vector.append(index)

        for word in tup[BODY_POSITION]:
            if word in ordered_body_words_list:
                index = ordered_body_words_list.index(word)
                bodies_vector.append(index)
            if word in ordered_topic_words_list:
                index = ordered_topic_words_list.index(word)
                topic_keyword_vector.append(index)

        feature_vector_dict['words_vectors'].append(bodies_vector)
        feature_vector_dict['topic_keyword_vectors'].append(topic_keyword_vector)
        feature_vector_dict['topics_classes'].append(topics)
        feature_vector_dict['places_classes'].append(places)

    return feature_vector_dict
