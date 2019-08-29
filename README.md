# ACL Author Disambiguation

Entity Disambiguation for the new ACL Anthology

Written by Gabriel Orlanski

# Requirements:
#### PDF Parsing
* [GROBID PDF parser](https://github.com/kermitt2/grobid)
    * I used GROBID and the [python client written by them as well](https://github.com/kermitt2/grobid-client-python)
    * You can use any PDF parser, but the results must be in XML files. Please check config.json for the XPaths you need
    
#### Modules
* PyYAML 5.1.2
* Unidecode 1.1.1
* fuzzysearch 0.6.2
* hurry.filesize 0.9
* lxml 4.4.0
* multiprocessing 2.6.2.1
* nltk 3.4.4
* numpy 1.17.0
* py-stringmatching 0.4.1
* scikit-learn 0.21.3 
* scipy 1.3.1
* textdistance 4.1.4
* tqdm 4.33.0

# How to use
### Basic use: 
1. Run GROBID and its python client on the pdfs
2. Run create_data.py to generate the information about the papers, organizations, and manual fixes needed
3. Training model (You can skip if you want to use pre-trained models)
    1. Run preprocess_data.py
    2. Run train.py
4. Create the targets you want to disambiguate (NOT IMPLEMENTED YET)
5. Run disambiguate.py (NOT IMPLEMENTED YET)
    1. If you would like to test the disambiguation program, run evaluate-disambiguation.py
6. Check the results.json file, and change any 'same' key to any changes you want to make
7. Run update_papers.py to update papers with their new correct authors(NOT IMPLEMENTED YET)

### Using your own model:
You can use your own model if you would like, but there are a few requirements to do so:
1. You __must__ have .predict() and .predict_proba() functions that takes in a 2d array of vectors, the shape of which will be [n,m]
    1. n is the number of samples to predict
    2. is the length of each vector
    3. .predict() __must__ return a np.array() of 1s and 0s, where 1 is the same and 0 is different
    4. .predict_proba() __must__ return a np.array() of length 2 arrays where the first element is the probability of that the pair are different authors and the second is the probability that the pair is the same author
2. For the time being, you __must__ have a .voting attribute, where it is either 'soft' or 'hard'

### Using your own custom CompareAuthors:
You can use your own CompareAuthors, please take a look at the compare_authors class for more information on what you need. If you would like to pass specific information to it, take a look at create_training_data.py's getAuthorInfo()* and change it accordingly

\* I will try to make it easier to override this function by passing it to the create_training_data
 

# Acknowledgments

# References
