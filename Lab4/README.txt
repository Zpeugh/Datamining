stdlinux lab folder location:
    /home/3/peugh/cse5243/Datamining/lab3/


##################### Running the script #####################
In order to run the script to produce the output vectors, you
must have these python modules downloaded:
    nltk
    numpy
    scipy
    sklearn

Assuming all dependencies are met, run the file by typing the command
    python cluster_data.py


########################################## Files ##########################################

classify_data.py
    This script is set up to run KNN, Naive Bayes, and Decision Tree algorithms on a sammple
    of 8 reuters files at a 66/33 training split for the sake of time.  This should take
    less than 3 minutes to complete. The number of sample files can be adjusted by changing
    the num_files variable in the preprocessData() function to a number between 1-21.  The
    results of the classification will be printed to the console.

preprocess3.py
    This is the tweaked version of preprocess.py seen in lab 1.  Used to get do preliminary
    preprocessing of reuters articles.

utilities.py
    This file contains the functions to acquire the preprocessed feature vectors, and also
    contains all of the necessary functions to run and evaluate the classification of the
    feature vectors.

Lab3.docx
    This is the Lab report


**
If for some reason permissions are unset, the GitHub repository for the assignment
can be cloned by using the command:
git clone https://github.com/Zpeugh/Datamining.git
All files to run the lab are under /lab3
