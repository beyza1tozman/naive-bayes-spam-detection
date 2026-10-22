import os
from collections import Counter

import numpy as np


def read_mails(folder):
    mails_list = os.listdir(folder)
    mail_paths = [os.path.join(folder, mail) for mail in mails_list]

    texts = []

    for path in mail_paths:
        with open(path) as f:
            text = f.read()
            texts.append(text)

    labels = []

    for mail in mails_list:
        if mail.startswith('spmsg'):
            labels.append(1) # spam
        else:
            labels.append(0) # not spam

    return texts, labels


def build_dictionary(texts, size=2000):
    # only call this with train texts, otherwise its data leakage
    counter = Counter()

    for text in texts:
        words = text.split()
        counter.update(word for word in words if word.isalpha() and len(word) > 1)

    top_pairs = counter.most_common(size)
    return [word for word, count in top_pairs] # pairs are like ('order', 1414), we only need the word


def to_vectors(texts, word_to_index):
    X = np.zeros((len(texts), len(word_to_index))) # rows are mails, columns are dictionary words

    for i, text in enumerate(texts): # i is the row of the mail
        for word in text.split():
            if word in word_to_index: # skip words that are not in the dictionary (punctuation too)
                X[i, word_to_index[word]] += 1

    return X
