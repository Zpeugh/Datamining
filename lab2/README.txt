stdlinux lab folder location:
    /home/3/peugh/cse5243/Datamining/lab2/


##################### Running the script #####################
In order to run the script to produce the output vectors, you
must have these python modules downloaded:
    nltk
    numpy
    scipy
    sklearn

The main script to run the data is called cluster_data.py. This script is set up to
run KMeans and hierarchical clustering on a sammple of 5 files for times sake.  This
should not take much longer than 3 minutes in total.  The number of sample files can
be adjusted by changing the num_files variable in the preprocessData() function to a
number between 1-21.  The results of the clustering will be printed to the console
and the clusters formed can be parsed through and compared by looking at
results['clusters'] vs topics['clusters']

To run the file simply run the command  "python cluster_data.py" and wait for all
steps to complete.

##################### Understanding the output #####################





******
If for some reason permissions are unset, the GitHub repository for the assignment
can be cloned by using the command:
git clone https://github.com/Zpeugh/Datamining.git
All files to run the lab are under /lab2
