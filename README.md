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
* nltk 3.4.4
* numpy 1.17.0
* py-stringmatching 0.4.1
* scikit-learn 0.21.3 
* scipy 1.3.1
* textdistance 4.1.4
* tqdm 4.33.0

# How to use

# Acknowledgments

# References
