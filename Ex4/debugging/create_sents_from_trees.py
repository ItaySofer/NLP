def load_sents_to_parse(filename):
    sents = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
    return sents


sents = load_sents_to_parse("test_trees.txt")
for sent in sents:
    sent_as_list = sent.split()
    strip_sent = [word.strip("()") for word in sent_as_list]
    filtered_sent = [word for word in strip_sent if word.islower() or word == "!" or word == "."]
    print " ".join(filtered_sent)