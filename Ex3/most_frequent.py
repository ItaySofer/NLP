from collections import defaultdict

from data import *

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    """
    ### YOUR CODE HERE
    word_to_tags_dict = defaultdict(lambda: defaultdict(int))
    for sent in train_data:
        for word_tag_pair in sent:
            word, tag = word_tag_pair
            word_to_tags_dict[word][tag] += 1

    result = dict()
    for word, tag_count_dict in word_to_tags_dict.items():
        tag_count_pairs_sorted = list(sorted(tag_count_dict.items(), key=lambda tag_count_pair: tag_count_pair[1], reverse=True))
        most_frequent_tag = tag_count_pairs_sorted[0][0]
        result[word] = most_frequent_tag

    return result
    ### END YOUR CODE


def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    ### YOUR CODE HERE
    correct = 0
    total = 0
    for sent in test_set:
        for word_tag_pair in sent:
            word, tag = word_tag_pair
            if word in pred_tags:
                if pred_tags[word] == tag:
                    correct += 1
        total += len(sent)

    return float(correct) / total
    ### END YOUR CODE

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    print "dev: most frequent acc: " + str(most_frequent_eval(dev_sents, model))

    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: " + str(most_frequent_eval(test_sents, model))