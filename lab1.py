import urllib
import re
import nltk

# s = 'asdf=5;iwantthis123jasd'
# result = re.search('asdf=5;(.*)123jasd', s)
# print result.group(1)

nums = ["00","01","02","03","04","05","06","07","08","09","10","12","13","14","15","16","17","18","19","20","21"]
#
# #Load in all of the reuters from the archive
# for i in nums:
#     url = "https://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-0{0}.sgm".format(i)
#     print url
#     urllib.urlretrieve (url, "reuters/{0}.sgm".format(i) )
#

#Loop through all 21 files and input them into arrays of strings between Reuters





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
    str = str.replace('\n','')
    str = str.replace('&lt;','<')
    return str



# Takes an array of string reuter bodies and returns a list of tuples in he form (title, body)
# where title and body are both the filtered string content between their respective tags
def get_title_body_entry_array( reuter_array ):
    tuple_array = []
    i = 0
    for reuter in reuter_array:
        reuter = filter_string(reuter)
        title = re.search( '<TITLE>(.*)</TITLE>', reuter )
        body = re.search( '<BODY>(.*)</BODY>', reuter )
        if not title:
            title = "NONE"
        else:
            title = title.group(1)
        if not body:
            body = "NONE"
        else:
            body = body.group(1)
        tuple_array.append( (title, body) )
        i += 1
    return tuple_array



# for i in range(0, 21):
#
#     reuter_array = make_reuter_list_from_file( "reuters/{0}.sgm".format(nums[i]) )
#     print( len( reuter_array ) )

reuter_array = make_reuter_list_from_file( "reuters/{0}.sgm".format(nums[0]) )
one = get_title_body_entry_array( reuter_array )

body1 = set(nltk.word_tokenize( one[1][1] ))
body2 = set(nltk.word_tokenize( one[6][1] ))
print (nltk.jaccard_distance(body1, body2))


#
#
# file_0 = open("reuters/{0}.sgm".format(nums[0]), 'r' )
#
# num_open_bodies = 0
# num_closed_bodies = 0
#
#
#
# monetary_hot_words = ["$", "cents", "dollars", "dollar", "cent"]
#
# profit_loss_buzzwords = ["vs"]
#
#
#
# for line in file_0:
#     # if ("<TITLE" in line):
#     #     num_open_bodies += 1
#     #
#     # if ("</TITLE>" in line):
#     #     num_closed_bodies += 1
#
#
#     if ("<REUTER" in line):
#         result = re.search('<TITLE>(.*)</TITLE>', line)
#         if result:
#             print( result.group(1) )
#
#
#     #find titles and add them to the array, writing "none" if there is no title
#     if ("<TITLE" in line):
#         result = re.search('<TITLE>(.*)</TITLE>', line)
#         if result:
#             print( result.group(1) )
#
#     # if ("</TITLE>" in line):
#     #     num_closed_bodies += 1
#
# #
# #
# # print( num_open_bodies )
# # print( num_closed_bodies )
