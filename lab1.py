import numpy as np
import preprocess
# import matplotlib.pyplot as plt

TOPICS_POSITION = 0
PLACES_POSITION = 1
TITLE_POSITION = 2
BODY_POSITION = 3

BODY_LOWER_CUTOFF = .005
BODY_UPPER_CUTOFF = .10


#####################################BEGIN SCRIPT#########################################
full_tuple_list = []
body_word_frequency_dict = dict()
topic_word_frequency_dict = dict()
# title_word_frequency_dict = dict()
topics_set = set()
NUM_SUFFIXES = ["00","01","02","03","04","05","06","07","08","09","10","12","13","14","15","16","17","18","19","20","21"]

for i in range(0, len(NUM_SUFFIXES)):
# for i in range(0, 1):

    reuter_array = preprocess.make_reuter_list_from_file( "reuters/{0}.sgm".format(NUM_SUFFIXES[i]) )
    split_array = preprocess.get_entry_array( reuter_array )
    print("{0}/{1}".format(i,21))
    for article in split_array:
        title = preprocess.tokenize_and_clean(article[TITLE_POSITION])
        body = preprocess.tokenize_and_clean(article[BODY_POSITION])
        topics = article[TOPICS_POSITION]
        places = article[PLACES_POSITION]
        full_tuple_list.append( (topics, places, title, body) )
        for topic in topics:
            topics_set.add(topic)
            if topic in topic_word_frequency_dict:
                topic_word_frequency_dict[topic] += 1
            else:
                topic_word_frequency_dict[topic] = 1
        # for word in title:
        #     if word in title_word_frequency_dict:
        #         title_word_frequency_dict[word] += 1
        #     else:
        #         title_word_frequency_dict[word] = 1
        for word in body:
            if word in body_word_frequency_dict:
                body_word_frequency_dict[word] += 1
            else:
                body_word_frequency_dict[word] = 1



# length = len(body_word_frequency_dict)
#
# body_words = np.array(list(body_word_frequency_dict.values()) ) / length
# body_words = np.array(body_words)
#
# plt.figure(1)
# plt.scatter(np.arange(len(body_words)), body_words)
# plt.title("Body Word-Frequency")
# plt.xlabel("Individual Words")
# plt.ylabel("Normalized per document Frequency")
#
# body_words2 = np.array(list(body_word_frequency_dict.values())) / length
# body_words2 = np.array([x for x in body_words if x > BODY_LOWER_CUTOFF and x < BODY_UPPER_CUTOFF])
#
# plt.figure(2)
# plt.scatter(np.arange(len(body_words2)), body_words2)
# plt.title("Sliced Body Word-frequency")
# plt.xlabel("Individual Words")
# plt.ylabel("Normalized per document Frequency")
#
# plt.show()


#
# plt.scatter(np.arange(len(topic_word_frequency_dict)), list(topic_word_frequency_dict.values()) )
# plt.show()

# I will choose a threshold frequency as the inverse of the number of topics available
# to reduce the dimensionality of the word attributes
# sliced_body_dict = throw_out_below_frequency( body_word_frequency_dict, 1/len(topics_set) )
body_copy = body_word_frequency_dict.copy()
sliced_body_dict = preprocess.throw_out_below_frequency( body_word_frequency_dict, BODY_LOWER_CUTOFF, BODY_UPPER_CUTOFF )

print( "The length of the sliced dictionary is %i" %( len(sliced_body_dict)) )

# sliced_title_dict = throw_out_below_frequency( title_word_frequency_dict, 1/len(topics_set) )
# sliced_title_dict = preprocess.throw_out_below_frequency( title_word_frequency_dict, TITLE_LOWER_CUTOFF, TITLE_UPPER_CUTOFF )
#
# print( "The length of the sliced dictionary is %i" %( len(sliced_title_dict)) )

# ordered_title_words_list = list( sliced_title_dict.keys() )
ordered_body_words_list = list( sliced_body_dict.keys() )


final_vector_dataset = preprocess.create_feature_vector( list(topics_set), ordered_body_words_list, full_tuple_list )

words = final_vector_dataset["words_vectors"]
keywords = final_vector_dataset["topic_keyword_vectors"]
topics = final_vector_dataset["topics_classes"]
places = final_vector_dataset["places_classes"]


# # print (nltk.jaccard_distance(body1, body2))
