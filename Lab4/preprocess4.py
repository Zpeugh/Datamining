# Author: Zach Peugh #
#    CSE 5243 Lab2   #
#       2/22/2016     #

################ Utility Module ##############
import re
import nltk
import numpy as np
import binascii
import random

TOPICS_POSITION = 0
BODY_POSITION = 1

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
# Also converts all numbers to "NUMBER" for better document comparison
def filter_string(str):
    str = str.replace('\n',' ')
    str = str.replace('&lt;','<')
    #replace all numbers with the word NUMBER
    #str = re.sub(r"([\d]+([\.,]*[\d]+)+)+", "NUMBER", str)
    return str



# return a list of strings between <D></D> tags
def get_topics( strng ):
    strng = strng.replace("</D><D>",'~')
    strng = strng.strip("<D>")
    strng = strng.strip("</D>")
    return strng.split('~')


# Takes an array of string reuter bodies and returns a list of tuples in the form
# ( [topics], body_string )
# where title and body are both the filtered string content between their respective tags
def get_entry_array( reuter_array ):
    tuple_array = []
    i = 0
    for reuter in reuter_array:
        reuter = filter_string(reuter)

        body = filter_string( get_content_between_strings("<BODY>", "</BODY>", reuter) )
        #body = get_content_between_strings("<BODY>", "</BODY>", reuter)
        topics = get_topics(get_content_between_strings("<TOPICS>", "</TOPICS>", reuter))
        tuple_array.append( (topics, body) )
        i += 1
    return tuple_array


def choose_most_likely_label(word_dict, words):
    if len(words) > 1:
        counts = dict()
        for word in words:
            counts[word] = word_dict[word]
        return sorted(counts, key=counts.get)[-1]
    else:
        return words[0]



def preprocess_data(reuters_directory="/home/0/srini/WWW/674/public/reuters", num_files=21):

    full_tuple_list = []
    topics_set = set()
    topic_classes = []

    NUM_SUFFIXES = ["00","01","02","03","04","05","06","07","08","09","10","12","13","14","15","16","17","18","19","20","21"]

    for i in range(num_files):

        reuter_array = make_reuter_list_from_file( "{0}/reut2-0{1}.sgm".format(reuters_directory,NUM_SUFFIXES[i]) )
        split_array = get_entry_array( reuter_array )
        print("Processing article {0}".format(i))
        for article in split_array:
            body = list( nltk.word_tokenize( article[BODY_POSITION] ))
            topics = article[TOPICS_POSITION]

            hashed_shingles = set()
            if topics and topics != ['']:

                for topic in topics:
                    topics_set.add(topic)

                for i in range(len(body) - 2):
                    shingle = body[i] + " " + body[i+1] + " " + body[i+2]
                    hashed_shingles.add( binascii.crc32(shingle) & 0xffffffff )

                full_tuple_list.append( (topics, hashed_shingles) )

    return full_tuple_list
