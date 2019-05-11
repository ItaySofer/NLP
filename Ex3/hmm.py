from data import *
import time
import numpy as np
import itertools


def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print "Start training"
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts = {}, {}, {}, {}, {}
    ### YOUR CODE HERE
    for sentence in sents:
        y_prev2 = '*'
        y_prev1 = '*'

        bigram = (y_prev2, y_prev1)
        q_bi_counts[bigram] = q_bi_counts.get(bigram, 0) + 1
        q_uni_counts[y_prev2] = q_uni_counts.get(y_prev2, 0) + 1
        q_uni_counts[y_prev1] = q_uni_counts.get(y_prev1, 0) + 1

        for x, y in sentence:
            # q transitions
            trigram = (y_prev2, y_prev1, y)
            bigram = (y_prev1, y)
            unigram = y
            q_tri_counts[trigram] = q_tri_counts.get(trigram, 0) + 1
            q_bi_counts[bigram] = q_bi_counts.get(bigram, 0) + 1
            q_uni_counts[unigram] = q_uni_counts.get(unigram, 0) + 1

            # e emissions
            e_word_pair = (x, y)
            e_tag_counts[y] = e_tag_counts.get(y, 0) + 1
            e_word_tag_counts[e_word_pair] = e_word_tag_counts.get(e_word_pair, 0) + 1

            total_tokens += 1
            y_prev2 = y_prev1
            y_prev1 = y

        trigram = (y_prev2, y_prev1, 'STOP')
        bigram = (y_prev1, 'STOP')
        unigram = 'STOP'
        q_tri_counts[trigram] = q_tri_counts.get(trigram, 0) + 1
        q_bi_counts[bigram] = q_bi_counts.get(bigram, 0) + 1
        q_uni_counts[unigram] = q_uni_counts.get(unigram, 0) + 1
        total_tokens += 1

    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts

def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE

    def q(y, y_prev2, y_prev1):
        if (y_prev2, y_prev1) in q_bi_counts:
            trigram = float(q_tri_counts.get((y_prev2, y_prev1, y), 0.0)) / float(q_bi_counts[(y_prev2, y_prev1)])
        else:
            trigram = 0.0
        if y_prev1 in q_uni_counts:
            bigram = float(q_bi_counts.get((y_prev1, y), 0.0)) / float(q_uni_counts[y_prev1])
        else:
            bigram = 0.0
        unigram = float(q_uni_counts.get(y, 0.0)) / float(total_tokens)
        return lambda1 * trigram + lambda2 * bigram + (1 - lambda1 - lambda2) * unigram

    e = lambda x, y: float(e_word_tag_counts[(x, y)]) / float(e_tag_counts[y]) if (x, y) in e_word_tag_counts else 0.0

    trajectory = {(0, '*', '*'): 0.0}   # Work in log space
    bp = {}

    # Prune here - for each Sk use only tags which have prob > 0 for word k
    S = [['*']] + [[tag for w, tag in list(itertools.product([token], e_tag_counts.keys()))
                    if (w, tag) in e_word_tag_counts] for token in sent]

    for k in range(1, len(sent) + 1):
        for u in S[k - 1]:
            for v in S[k]:
                max_w = None
                max_w_val = -np.inf
                for w in S[max(k - 2, 0)]:
                    curr_w_val = trajectory[(k - 1, w, u)] + np.log(q(v, w, u)) + np.log(e(sent[k - 1], v))
                    if curr_w_val > max_w_val:
                        max_w_val = curr_w_val
                        max_w = w
                trajectory[(k, u, v)] = max_w_val
                bp[(k, u, v)] = max_w

    n = len(sent)
    highest_prob = -np.inf
    for u in S[n - 1]:
        for v in S[n]:
            prob = trajectory[(n, u, v)] + np.log(q('STOP', u, v))
            if prob > highest_prob:
                highest_prob = prob
                predicted_tags[n - 1] = v
                if n > 1:
                    predicted_tags[n - 2] = u

    for k in range(n - 2, 0, -1):
        y_k_plus_2 = predicted_tags[k + 1]
        y_k_plus_1 = predicted_tags[k]
        predicted_tags[k - 1] = bp[(k + 2, y_k_plus_1, y_k_plus_2)]

    ### END YOUR CODE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print "Start evaluation"
    acc_viterbi = 0.0
    ### YOUR CODE HERE
    lambda1 = 0.747
    lambda2 = 0.252
    correct = 0
    total = 0

    for entry in test_data:
        sent = [word_tag_pair[0] for word_tag_pair in entry]
        pred = hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                           e_word_tag_counts,e_tag_counts, lambda1, lambda2)
        for word_tag_pair, tag_pred in zip(entry, pred):
            total += 1
            if tag_pred == word_tag_pair[1]:
                correct += 1

    acc_viterbi = 100.0 * float(correct) / float(total)
    ### END YOUR CODE

    return acc_viterbi

if __name__ == "__main__":
    start_time = time.time()
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
    print "HMM DEV accuracy: " + str(acc_viterbi)

    train_dev_end_time = time.time()
    print "Train and dev evaluation elapsed: " + str(train_dev_end_time - start_time) + " seconds"

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                           e_word_tag_counts, e_tag_counts)
        print "HMM TEST accuracy: " + str(acc_viterbi)
        full_flow_end_time = time.time()
        print "Full flow elapsed: " + str(full_flow_end_time - start_time) + " seconds"