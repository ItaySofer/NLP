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

def cnf_cky(pcfg, sent):
    ### YOUR CODE HERE
    raise NotImplementedError
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
    non_cnf_pcfg = PCFG.from_file_assert(sys.argv[2])
    sents_to_parse = load_sents_to_parse(sys.argv[3])
    for sent in sents_to_parse:
        print cnf_cky(cnf_pcfg, sent)
        print non_cnf_cky(non_cnf_pcfg, sent)
