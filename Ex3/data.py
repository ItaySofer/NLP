import os
import re
MIN_FREQ = 3


def invert_dict(d):
    res = {}
    for k, v in d.iteritems():
        res[v] = k
    return res


def read_conll_pos_file(path):
    """
        Takes a path to a file and returns a list of word/tag pairs
    """
    sents = []
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.append(curr)
                curr = []
            else:
                tokens = line.strip().split("\t")
                curr.append((tokens[1],tokens[3]))
    return sents


def increment_count(count_dict, key):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1


def compute_vocab_count(sents):
    """
        Takes a corpus and computes all words and the number of times they appear
    """
    vocab = {}
    for sent in sents:
        for token in sent:
            increment_count(vocab, token[0])
    return vocab


def replace_word(word):
    """
        Replaces rare words with categories (numbers, dates, etc...)
    """
    ### YOUR CODE HERE
    alpha = re.compile("[a-zA-Z]")
    numeric = re.compile("[0-9]")
    dash = re.compile("-")
    slash = re.compile("\\\\")
    comma = re.compile(",")
    period = re.compile(".")
    caps = re.compile("[A-Z]")
    panct = re.compile("[,.!?]")

    if word.isdigit():
        return "num"
    if numeric.search(word) is not None and alpha.search(word) is not None:
        return "containsDigitAndAlpha"
    if numeric.search(word) is not None and dash.search(word) is not None:
        return "containsDigitAndDash"
    if numeric.search(word) is not None and slash.search(word) is not None:
        return "containsDigitAndSlash"
    if numeric.search(word) is not None and comma.search(word) is not None:
        return "containsDigitAndComma"
    if numeric.search(word) is not None and period.search(word) is not None:
        return "containsDigitAndPeriod"
    if word.isupper():
        return "allCaps"
    if caps.match(word):
        return "initCap"
    if word.isalpha() and caps.match(word) is None:
        return "lowerCase"
    if panct.search(word):
        return "punct"
    ### END YOUR CODE
    return "UNK"


def preprocess_sent(vocab, sents):
    """
        return a sentence, where every word that is not frequent enough is replaced
    """
    res = []
    total, replaced = 0, 0
    for sent in sents:
        new_sent = []
        for token in sent:
            if token[0] in vocab and vocab[token[0]] >= MIN_FREQ:
                new_sent.append(token)
            else:
                new_sent.append((replace_word(token[0]), token[1]))
                replaced += 1
            total += 1
        res.append(new_sent)
    print "replaced: " + str(float(replaced)/total)
    return res







