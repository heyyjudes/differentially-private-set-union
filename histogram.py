import os
import json
import copy
import operator
import numpy as np
from enum import Enum
from tqdm import tqdm
from collections import Counter, defaultdict

# local libs
import utils as ut


class Noise(Enum):
    LAPLACE = 1
    GAUSSIAN = 2


class Algorithm(Enum):
    COUNT = 1
    WEIGHTED = 2
    POLICY = 3
    GREEDY = 4
    MAXSUM = 5


class Histogram:
    '''
    Histogram class for generating histograms from different policies
    self.ngram_hist contains the generated histogram
    '''
    def __init__(self, n, input_df, dataset_str, ngram_union):
        '''
        Initializing histogram class
        :param n: n in n-gram to build histogram from
        :param input_df: input dataframe with clean text data under "clean_text" column
        :param dataset_str: name of dataset in input_df
        '''
        self.n = n
        self.ngram_union = ngram_union
        self.ngram_hist = None 
        self.input_df = input_df
        self.Delta_0 = None
        self.Delta = None
        self.noise_dist = None # Noise Enum
        self.algorithm = None # Algorithm Enum
        self.dataset_str = dataset_str

    def generate_hist(self):
        '''
        Generate histogram with infinite delta_0 (each use can add all their tokens)
        :return: generated histogram
        '''
        self.ngram_hist = Counter() 
        for index, group in tqdm(self.input_df.groupby("author"), position=0, leave=True):
            posts = group["clean_text"]
            posts = [p.split(" ") for p in posts]

            if self.ngram_union:
                words = [tokens for p in posts for tokens in p]
                for i in range(2, self.n + 1):
                    posts_tokenized = [ut.tokens2ngram(p, i) for p in posts]
                    words = words + [tokens for p in posts_tokenized for tokens in p]
            else:
                if self.n > 1:
                    posts = [ut.tokens2ngram(p, self.n) for p in posts]
                words = [tokens for p in posts for tokens in p]
            # if self.n > 1:
            #     posts = [ut.tokens2ngram(p, self.n) for p in posts]
            # words = [tokens for p in posts for tokens in p]
            self.ngram_hist.update(list(set(words)))
        return self.ngram_hist

    def generate_delta_hist(self, delta_0):
        '''
        Generate histogram where each user can only contribute delta sampled ngrams
        :param delta_0: sensitivity parameter which limits number of ngrams each user can contribute
        :return: generated histogram
        '''
        self.ngram_hist = Counter() 
        self.Delta_0 = delta_0
        self.algorithm = Algorithm.COUNT

        for index, group in tqdm(self.input_df.groupby("author"), position=0, leave=True):
            posts = group["clean_text"]
            posts = [p.split(" ") for p in posts]
            if self.ngram_union:
                words = [tokens for p in posts for tokens in p]
                for i in range(2, self.n + 1):
                    posts_tokenized = [ut.tokens2ngram(p, i) for p in posts]
                    words = words + [tokens for p in posts_tokenized for tokens in p]
            else:
                if self.n > 1:
                    posts = [ut.tokens2ngram(p, self.n) for p in posts]
                words = [tokens for p in posts for tokens in p]

            all_grams = list(set(words))
            if len(all_grams) > self.Delta_0:
                selected_ngrams = np.random.choice(all_grams, size=self.Delta_0, replace=False).tolist()
            else:
                selected_ngrams = all_grams[:]
            self.ngram_hist.update(list(set(selected_ngrams)))
        return self.ngram_hist

    def generate_weighted_hist(self, delta_0, weighted_dist, delta=None):
        '''
        Generate histogram according to weighted Laplace or weighted Gaussian policy
        :param delta: sensitivity parameter which limits amount each user can contribute (this is NOT small delta)
        :param delta_0: parameter limiting number of unique ngrams each user can contribute
        :param weighted_dist: 'laplace' or 'gaussian'
        :type weighted_dist: str
        :return: generated histogram
        '''
        self.ngram_hist = defaultdict(float)
        assert(type(weighted_dist) == Noise)
        self.noise_dist = weighted_dist
        self.algorithm = Algorithm.WEIGHTED
        self.Delta_0 = delta_0
        self.Delta = delta if delta else 1

        for index, group in tqdm(self.input_df.groupby("author"), position=0, leave=True):
            posts = group["clean_text"]
            posts = [p.split(" ") for p in posts]

            if self.ngram_union:
                words = [tokens for p in posts for tokens in p]
                for i in range(2, self.n + 1):
                    posts_tokenized = [ut.tokens2ngram(p, i) for p in posts]
                    words = words + [tokens for p in posts_tokenized for tokens in p]
            else:
                if self.n > 1:
                    posts = [ut.tokens2ngram(p, self.n) for p in posts]
                words = [tokens for p in posts for tokens in p]

            all_grams = list(set(words))

            if len(all_grams) > self.Delta_0:
                selected_ngrams = np.random.choice(all_grams, size=self.Delta_0, replace=False).tolist()
            else:
                selected_ngrams = all_grams[:]

            for ngram in selected_ngrams:
                if self.noise_dist == Noise.LAPLACE:
                    self.ngram_hist[ngram] += self.Delta/len(selected_ngrams)
                elif self.noise_dist==Noise.GAUSSIAN: 
                    self.ngram_hist[ngram] += np.sqrt(self.Delta/len(selected_ngrams))
        return self.ngram_hist

    def update_budget_dict(self, rho_dict, update):
        '''
        Helper method to update uniformly update rho_dict
        :param rho_dict: input dictionary containing values to update
        :type rho_dict: dict
        :param update: amount to subtract each value in rho_dict by
        :return: update rho_dict
        '''
        for key in rho_dict: 
            rho_dict[key] -= update 
        return rho_dict

    def generate_policy_laplace_hist(self, delta_0, Gamma, passes=1, delta=None):
        '''
        Generate histogram according to Policy Laplace
        :param delta_0: parameter limiting number of unique ngrams each user can contribute
        :param passes: number of passes each user makes when contributing to histogram
        :param delta: sensitivity parameter which limits amount each user can contribute
        :param rho: threshold for cutoff
        :return: generated histogram
        '''
        self.ngram_hist = defaultdict(float)
        self.Delta_0 = delta_0
        self.Delta = delta if delta else 1
        # divide budget by number of passes that will be made
        self.Delta = self.Delta/passes

        self.algorithm = Algorithm.POLICY
        self.noise_dist = Noise.LAPLACE

        # keep track of selected ngrams over multiple passes
        selected_ngram_dict = {}

        for p in range(passes):
            for index, group in tqdm(self.input_df.groupby("author"), position=0, leave=True):
                posts = group["clean_text"]
                posts = [p.split(" ") for p in posts]

                if self.ngram_union:
                    words = [tokens for p in posts for tokens in p]
                    for i in range(2, self.n + 1):
                        posts_tokenized = [ut.tokens2ngram(p, i) for p in posts]
                        words = words + [tokens for p in posts_tokenized for tokens in p]
                else:
                    if self.n > 1:
                        posts = [ut.tokens2ngram(p, self.n) for p in posts]
                    words = [tokens for p in posts for tokens in p]

                all_grams = list(set(words))

                # sample delta_0 ngrams
                if len(all_grams) > self.Delta_0:
                    if p == 0:
                        selected_ngrams = np.random.choice(all_grams, size=self.Delta_0, replace=False).tolist()
                        selected_ngram_dict[index] = selected_ngrams
                    else:
                        selected_ngrams = selected_ngram_dict[index]
                else:
                    selected_ngrams = all_grams[:]

                gap_dict = {}

                for w in selected_ngrams:
                    if self.ngram_hist[w] < Gamma:
                        gap_dict[w] = Gamma - self.ngram_hist[w]
                # sort rho dict
                sorted_gap_dict = sorted(gap_dict.items(), key=operator.itemgetter(1))

                sorted_gap_keys = [k for k, v in sorted_gap_dict]

                budget = copy.copy(self.Delta)
                total_tokens = len(sorted_gap_keys)

                for i, w in enumerate(sorted_gap_keys):
                    cost = gap_dict[w]*(total_tokens-i)
                    if cost < budget:
                        for j in range(i, total_tokens):
                            add_gram = sorted_gap_keys[j]
                            self.ngram_hist[add_gram] += gap_dict[w]
                        # update remaining budget
                        budget -= cost
                        # update dictionary of values containing difference from gap
                        gap_dict = self.update_budget_dict(gap_dict, gap_dict[w])
                    else:
                        for j in range(i, total_tokens):
                            add_gram = sorted_gap_keys[j]
                            self.ngram_hist[add_gram] += budget/(total_tokens-i)
                        break

        return self.ngram_hist

    def generate_policy_greedy_hist(self, delta_0, Gamma, delta=None):
        '''
        Generate histogram according to Greedy Laplace Policy
        :param delta: sensitivity parameter which limits amount each user can contribute
        :param delta_0: parameter limiting number of unique ngrams each user can contribute
        :param Gamma: threshold for cutoff
        :return: generated histogram
        '''
        self.ngram_hist = defaultdict(float)
        self.Delta = delta if delta else 1
        self.Delta_0 = delta_0
        self.noise_dist = Noise.LAPLACE
        self.algorithm = Algorithm.GREEDY

        for index, group in tqdm(self.input_df.groupby("author"), position=0, leave=True):
            posts = group["clean_text"]
            posts = [p.split(" ") for p in posts]

            if self.ngram_union:
                words = [tokens for p in posts for tokens in p]
                for i in range(2, self.n + 1):
                    posts_tokenized = [ut.tokens2ngram(p, i) for p in posts]
                    words = words + [tokens for p in posts_tokenized for tokens in p]
            else:
                if self.n > 1:
                    posts = [ut.tokens2ngram(p, self.n) for p in posts]
                words = [tokens for p in posts for tokens in p]

            all_grams = list(set(words))

            if len(all_grams) > self.Delta_0:
                selected_ngrams = np.random.choice(all_grams, size=self.Delta_0, replace=False).tolist()
            else:
                selected_ngrams = all_grams[:]

            rho_dict = {}
            # for w in all_grams:
            for w in selected_ngrams:
                if self.ngram_hist[w] < Gamma:
                    rho_dict[w] = Gamma - self.ngram_hist[w]
            # sort rho dict
            sorted_rho_dict = sorted(rho_dict.items(), key=operator.itemgetter(1))

            sorted_rho_keys = [k for k, v in sorted_rho_dict]

            budget = copy.copy(self.Delta)

            for i, w in enumerate(sorted_rho_keys):
                cost = rho_dict[w]
                if cost < budget:
                    self.ngram_hist[w] += rho_dict[w]
                    # update remaining budget
                    budget -= cost
                else:
                    # not enough budget to meet threshold: add rest of budget to item
                    self.ngram_hist[w] += budget
                    break

        return self.ngram_hist

    def generate_policy_gaussian_hist(self, delta_0, Gamma, passes=1, delta=None):
        '''
        Generate histogram according to Policy Gaussian histogram
        :param delta_0: parameter limiting number of unique ngrams each user can contribute
        :param rho: threshold for cutoff
        :return: generated histogram
        '''
        self.ngram_hist = defaultdict(float)
        self.Delta_0 = delta_0
        self.Delta = delta if delta else 1
        # divide budget by number of passes that will be made
        self.Delta = self.Delta/passes
        self.noise_dist = Noise.GAUSSIAN
        self.algorithm = Algorithm.POLICY

        # keep track of selected ngrams over multiple passes
        selected_ngram_dict = {}

        for p in range(passes):
            for index, group in tqdm(self.input_df.groupby("author"), position=0, leave=True):
                posts = group["clean_text"]
                posts = [p.split(" ") for p in posts]

                if self.ngram_union:
                    words = [tokens for p in posts for tokens in p]
                    for i in range(2, self.n + 1):
                        posts_tokenized = [ut.tokens2ngram(p, i) for p in posts]
                        words = words + [tokens for p in posts_tokenized for tokens in p]
                else:
                    if self.n > 1:
                        posts = [ut.tokens2ngram(p, self.n) for p in posts]
                    words = [tokens for p in posts for tokens in p]

                all_grams = list(set(words))

                # sample delta_0 ngrams
                if len(all_grams) > self.Delta_0:
                    if p == 0:
                        selected_ngrams = np.random.choice(all_grams, size=self.Delta_0, replace=False).tolist()
                        selected_ngram_dict[index] = selected_ngrams
                    else:
                        selected_ngrams = selected_ngram_dict[index]
                else:
                    selected_ngrams = all_grams[:]

                # filter out ngrams over threshold
                selected_ngrams = [gram for gram in selected_ngrams if self.ngram_hist[gram] < Gamma]

                # calculate normalization constant
                diff_arr = np.asarray([Gamma - self.ngram_hist[gram] for gram in selected_ngrams])
                Z = np.linalg.norm(diff_arr, ord=2)

                sum_contrib = np.zeros(len(diff_arr))
                # add update to histogram proportional to distance to threshold
                for i, ngram in enumerate(selected_ngrams):
                    self.ngram_hist[ngram] += min(self.Delta, Z)*diff_arr[i]/Z
                    sum_contrib[i] = min(self.Delta, Z) * diff_arr[i] / Z

                assert(np.round(np.linalg.norm(sum_contrib, 2), 3) <= np.round(np.sqrt(self.Delta), 3))

        return self.ngram_hist

    def generate_maxsum_gaussian_hist(self, delta_0, Gamma, passes=1, delta=None):
        '''
        Generate histogram according to max-sum Gaussian policy
        :param Delta_0: parameter limiting number of unique ngrams each user can contribute
        :n_list: build histogram on range of ngrams for n \in n_list
        :return: generated histogram
        '''

        self.ngram_hist = defaultdict(float)
        self.Delta_0 = delta_0
        self.Delta = delta if delta else 1
        # divide budget by number of passes that will be made
        self.Delta = self.Delta/passes
        self.noise_dist = Noise.GAUSSIAN
        self.algorithm = Algorithm.MAXSUM
        self.num_tokens = []
        for index, group in tqdm(self.input_df.groupby("author"), position=0, leave=True):
            posts = group["clean_text"]
            posts = [p.split(" ") for p in posts]

            if self.ngram_union:
                user_ngrams = [tokens for p in posts for tokens in p]
                for i in range(2, self.n+1):
                    posts_tokenized = [ut.tokens2ngram(p, i) for p in posts]
                    user_ngrams = user_ngrams + [tokens for p in posts_tokenized for tokens in p]
            else:
                if self.n > 1:
                    posts = [ut.tokens2ngram(p, self.n) for p in posts]
                user_ngrams = [tokens for p in posts for tokens in p]

            unique_user_ngrams = list(set(user_ngrams))
            self.num_tokens.append(unique_user_ngrams)

            if len(unique_user_ngrams) > self.Delta_0:
                selected_ngrams = np.random.choice(unique_user_ngrams, size=self.Delta_0, replace=False).tolist()
            else:
                selected_ngrams = unique_user_ngrams

            gap_dict = {}

            for w in selected_ngrams:
                if self.ngram_hist[w] < Gamma:
                    gap_dict[w] = Gamma - self.ngram_hist[w]
            # sort rho dict
            sorted_gap_dict = sorted(gap_dict.items(), key=lambda x: x[0])

            sorted_gap_keys = [k for k, v in sorted_gap_dict]

            budget = 1
            total_tokens = len(sorted_gap_keys)

            for i, w in enumerate(sorted_gap_keys):
                cost = gap_dict[w] ** 2 * (total_tokens - i)
                if cost < budget:
                    self.ngram_hist[w] = Gamma
                    # update remaining budget
                    budget -= gap_dict[w] ** 2
                else:
                    for j in range(i, total_tokens):
                        add_gram = sorted_gap_keys[j]
                        self.ngram_hist[add_gram] += budget / np.sqrt(total_tokens - i)
                    break

        save_str = 'user_tokens_{}_union.npy'.format(self.n) if self.ngram_union else 'user_tokens_{}.npy'.format(self.n)
        with open(save_str, 'wb') as f:
            np.save(f, np.asarray(self.num_tokens))
            
        return self.ngram_hist


    def save_hist(self, histogram_str, end_str):
        '''
        save self.ngram_hist
        :param histogram_str: holder to store histogram in
        :param end_str: string specifying delta, delta_0, and n
        :return: None
        '''
        file_path = "hist/{}/{}/".format(self.dataset_str, histogram_str)
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        with open("hist/{}/{}/{}_{}gram.json".format(self.dataset_str, histogram_str, self.dataset_str + end_str, self.n), 'w') as f: 
            json.dump(dict(self.ngram_hist), f)