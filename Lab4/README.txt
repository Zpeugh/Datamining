stdlinux lab folder location:
    /home/3/peugh/cse5243/Datamining/lab4/


##################### Running the script #####################
In order to run the script to produce the output vectors, you
must have these python modules downloaded:
    nltk
    numpy
    matplotlib (optional)

Assuming all dependencies are met, run the file by typing the command
    python minHash.py


########################################## Files ##########################################

minHash.py
    This script is set up to run MinHash on 4 documents for K values 16, 32, 64, 128, 256.
    The results will print to the screen.  They can be plotted if the matplotlib code is
    uncommented out, given that the matplotlib library is installed.

preprocess3.py
    This is the script to shingle and hash the document into feature vectors.

utilities.py
    This file contains all of the code to run the MinHash algorithm, and generate calculate
    the errors between similarity arrays.

Lab4.pdf
    This is the Lab report


**
If for some reason permissions are unset, the GitHub repository for the assignment
can be cloned by using the command:
git clone https://github.com/Zpeugh/Datamining.git
All files to run the lab are under /lab4
