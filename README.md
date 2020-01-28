# Differentially Private Set Union for Vocabulary Release
One Paragraph of project description goes here

## Prerequisites

```
python >= 3.6
numpy >= 1.14.3
pandas >= 0.22.0 
scipy >= 1.0.1
tqdm >= 4.23.4
```

##Dataset
```
unzip data/clean_askreddit.csv.zip
```
The default dataset used in this repo is clean_askreddit.csv.  
For your own input file, the file must contain an "author" and a "clean_text" column where the clean text has been 
preprocessed and tokenized. For an example, see utils.py. 

## Example: Generating a Histogram

```
python gen_histogram.py --save_histogram
```
This generates and saves a histogram with the Policy Gaussian algorithm and outputs the number of unigrams released. 

```
python gen_histogram.py --alg policy --noise laplace --n 2 --trials 3
```
To change which algorithm is used specify the --alg and --noise parameter. 
To change the n of ngrams in the histogram use: --ngrams For multiple shuffles 
of the dataset, use the --trials parameter. 


## Paper
tbd 

## Acknowledgments

