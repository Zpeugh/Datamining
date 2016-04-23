stdlinux lab folder location:
    /home/3/peugh/cse5243/Datamining/lab5/


##################### Running the script #####################
In order to run the script to produce the output vectors, you
must have these python modules downloaded:
    nltk
    numpy

Assuming all dependencies are met, run the file by typing the command
    python lab5.py


########################################## Files ##########################################

lab5.py
    This script is set up to run Association Rule classification on 5 Reuters files.  You
    can change the number to anything between 1-21 files.  If a permission error rises,
    you must chmod 755 apriori to be executable.

apriori
    This is Christian Borgelt's 32-bit apriori executable.  Permissions for execution must
    be set on this for the lab to work.

preprocess5.py
    This is the script to convert the Reuters documents into feature vectors

utilities.py
    This file contains all of the code to run the Association Rule classification

Lab5.pdf
    This is the Lab report

############################# Files generated during execution #############################

transactions.txt
    This is the input file for apriori.  It contains all of the lines of transactions.

appearances.txt
    This contains the constraints on rule form.

rules.txt
    This is the output of the apriori file.  It contains all of the rules to be used to
    classify



**
If for some reason permissions are unset, the GitHub repository for the assignment
can be cloned by using the command:
git clone https://github.com/Zpeugh/Datamining.git
All files to run the lab are under /lab5
