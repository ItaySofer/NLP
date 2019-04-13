#!/usr/local/bin/python

from data_utils import utils as du
import numpy as np
import pandas as pd
import csv
from collections import defaultdict

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                      index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)


def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)
    token_count = 0

    start_token = dataset[0][0]
    for sentence in dataset:
        unigram_counts[(start_token,)] += 1
        bigram_counts[(start_token, start_token,)] += 1
        for i in range(2, len(sentence)):
            w_i = sentence[i]
            w_i_minus_1 = sentence[i - 1]
            w_i_minus_2 = sentence[i - 2]
            unigram_counts[(w_i,)] += 1
            bigram_counts[(w_i, w_i_minus_1,)] += 1
            trigram_counts[(w_i, w_i_minus_1, w_i_minus_2,)] += 1
            token_count += 1

    return trigram_counts, bigram_counts, unigram_counts, token_count


def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    sum_log_2_p = 0
    lambda3 = 1 - lambda1 - lambda2
    test_token_count = 0
    for sentence in eval_dataset:
        for i in range(2, len(sentence)):
            w_i = sentence[i]
            w_i_minus_1 = sentence[i - 1]
            w_i_minus_2 = sentence[i - 2]
            unigram_p = float(unigram_counts[(w_i,)]) / train_token_count
            bigram_p = float(bigram_counts[(w_i, w_i_minus_1,)]) / unigram_counts[(w_i_minus_1,)] \
                if unigram_counts[(w_i_minus_1,)] != 0 else 0
            trigram_p = float(trigram_counts[(w_i, w_i_minus_1, w_i_minus_2,)]) / bigram_counts[(w_i_minus_1, w_i_minus_2,)] \
                if bigram_counts[(w_i_minus_1, w_i_minus_2,)] != 0 else 0
            p = lambda1 * trigram_p + lambda2 * bigram_p + lambda3 * unigram_p
            sum_log_2_p += np.log2(p)
            test_token_count += 1

    L = float(sum_log_2_p) / test_token_count
    perplexity = 2 ** (-L)
    return perplexity


def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    # Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    print "#perplexity: " + str(perplexity)
    ### YOUR CODE HERE
    ### END YOUR CODE


if __name__ == "__main__":
    test_ngram()
