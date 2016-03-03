stdlinux lab folder location:
    /home/3/peugh/cse5243/Datamining/lab2/


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

cluster_data.py
    This script is set up to run KMeans and hierarchical clustering on a sammple of 5
    files for times sake.  This should not take much longer than 3 minutes in total.
    The number of sample files can be adjusted by changing the num_files variable in
    the preprocessData() function to a number between 1-21.  The results of the
    clustering will be printed to the console and the clusters formed can be parsed
    through and compared by iterating through and looking at k_results['clusters']
    vs ground_truth_labels['clusters'] if you wish. This is best done inside an
    interactive console such as ipython.

preprocess2.py
    This is the tweaked version of preprocess.py seen in lab 1.  Used to get do preliminary
    preprocessing of reuters articles.

k_means.py
    This is the KMeans code found online

Lab2.docx
    This is the Lab report





**
If for some reason permissions are unset, the GitHub repository for the assignment
can be cloned by using the command:
git clone https://github.com/Zpeugh/Datamining.git
All files to run the lab are under /lab2

******
There was trouble with some imports not being configured correctly when trying to use
sklearn and numpy from your directory.  They worked fine from my local computer but
I could not get them to run on my sdlinux account due to these modules not being built
correctly (according to the console output).
