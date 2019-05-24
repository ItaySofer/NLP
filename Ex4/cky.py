from collections import defaultdict

from PCFG import PCFG
import math


def load_sents_to_parse(filename):
    sents = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
    return sents


def get_probability(pcfg, x, wi):
    x_rules = pcfg._rules[x]
    x_sum_wight = pcfg._sums[x]
    for rhs, weight in x_rules:
        if pcfg.is_preterminal(rhs) and rhs[0] == wi:
            return float(weight) / x_sum_wight

    return 0


def build_initial_pi_bp(pcfg, sent_as_list):
    pi = defaultdict(lambda: defaultdict(int))
    bp = defaultdict(lambda: dict())

    for x in pcfg._rules.keys():
        for i in range(len(sent_as_list)):
            wi = sent_as_list[i]
            pi[(i, i)][x] = get_probability(pcfg, x, wi)
            if pi[(i, i)][x] != 0:
                bp[(i, i)][x] = (wi, )

    return pi, bp


def calculate_pi_pb(i, j, x, pcfg, pi):
    max_prob = 0
    bp = None
    sum_wight = pcfg._sums[x]
    for rhs, weight in pcfg._rules[x]:
        if pcfg.is_preterminal(rhs):
            continue
        y, z = rhs
        for s in range(i, j):
            q = float(weight) / sum_wight
            prob = q * pi[(i, s)][y] * pi[(s+1, j)][z]
            if prob > max_prob:
                max_prob = prob
                bp = (y, z, s)

    return max_prob, bp


def parse_bp(i, j, symbol, bp, sent_as_list):
    bp_symbol = bp[(i, j)][symbol]

    if len(bp_symbol) == 1:
        return "(" + symbol + " " + bp_symbol[0] + ")"

    y, z, s = bp_symbol
    return "(" + symbol + " " + parse_bp(i, s, y, bp, sent_as_list) + " " +\
           parse_bp(s + 1, j, z, bp, sent_as_list) + ")"


def cnf_cky(pcfg, sent):
    ### YOUR CODE HERE
    sent_as_list = sent.split()
    n = len(sent_as_list)
    pi, bp = build_initial_pi_bp(pcfg, sent_as_list)

    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            for x in pcfg._rules.keys():
                pi[(i, j)][x], bp[(i, j)][x] = calculate_pi_pb(i, j, x, pcfg, pi)

    if pi[(0, n - 1)]["ROOT"] != 0:
        return parse_bp(0, n - 1, "ROOT", bp, sent_as_list)
    ### END YOUR CODE
    return "FAILED TO PARSE!"


def non_cnf_cky(pcfg, sent):
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE
    return "FAILED TO PARSE!"


if __name__ == '__main__':
    import sys
    cnf_pcfg = PCFG.from_file_assert(sys.argv[1], assert_cnf=True)
    # non_cnf_pcfg = PCFG.from_file_assert(sys.argv[2])
    sents_to_parse = load_sents_to_parse(sys.argv[3])
    for sent in sents_to_parse:
        print cnf_cky(cnf_pcfg, sent)
        # print non_cnf_cky(non_cnf_pcfg, sent)
