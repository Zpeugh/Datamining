##################### Running the script #####################
This is a python program that uses nltk and numpy imports.
In order to run the script to produce the output vectors, you
must have these python modules downloaded. Specifically inside of
the nltk module, you will need the "punkt" and "stopwords" packages.
These can be acquired with the nltk downloader.

To run the file simply run the command  "python lab1.py" and wait for all
steps to complete.

##################### Understanding the output #####################
Once the script has run, the output will be 5 different files, inside
of the Outputs directory.  All files have 20578 lines, and each line is one
documents feature vector. The following are the split of the text documents.
More information on each of the feature vectors can be read in the report.docx
file.

Feature Vectors:
    important_words_vectors.txt
    buzzword_vectors.txt
    topic_keyword_vectors.txt

Class Vectors:
    topics_classes.txt
    places_classes.txt

e.g. line 58 of 'topics_classes.txt' is the topic class vector for line 58
of each of the 3 feature vectors.


******
If for some reason permissions are unset, the GitHub repository for the assignment
can be cloned by using the command:
git clone https://github.com/Zpeugh/Datamining.git
