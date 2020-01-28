# Differentially Private Set Union for Vocabulary Release
This repository contains the code and dataset for the following paper:  
> **Differentially Private Set Union with Applications to Vocabulary
Generation**<br>
> Pankaj Gulhane, Sivakanth Gopi, Janardhan Kulkarni, Judy Hanwen Shen,
Milad Shokouhi, and Sergey Yekhanin<br>
> https://arxiv.org/
>
> **Abstract:** *Motivated by many applications in language modeling, we study the basic operation of set union in the differential privacy setting. In the set union problem, we are given a universe U of items, possibly of infinite size. Suppose we are given a database D of users. Each user i contributes a subset W_i of U. We want an (epsilon, delta)-differentially private Algorithm A which outputs a subset S such that the size of S is as large as possible.
In this paper, we design new algorithms for this problem, which substantially improve upon the algorithms that can be derived by known techniques in the literature.*

## Prerequisites

```
python >= 3.6
numpy >= 1.14.3
pandas >= 0.22.0 
scipy >= 1.0.1
tqdm >= 4.23.4
```

## Dataset
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
To change the n of ngrams in the histogram use: --ngrams . For multiple shuffles 
of the dataset, use the --trials parameter. 

## Acknowledgments

