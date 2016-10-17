import numpy as np
import re
import glob
import itertools
from collections import Counter
import pickle
import pandas as pd

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"<br />", " ", string)
    return string.strip().lower()


def get_reviews(path, clean=True):
    complete_path = path + '/*.txt'
    files = glob.glob(complete_path)
    reviews = [str(open(rev).readlines()[0]).strip() for rev in files]
    if clean:
        reviews = [clean_str(rev) for rev in reviews]
    return reviews


def get_train_sets():
    train_positive_reviews = pickle.load(open("data/train_pos.p","rb"))
    train_negative_reviews = pickle.load(open("data/train_neg.p","rb"))
    return train_positive_reviews, train_negative_reviews


def get_test_sets():
    test_positive_reviews = pickle.load(open("data/test_pos.p","rb"))
    test_negative_reviews = pickle.load(open("data/test_neg.p","rb"))
    return test_positive_reviews, test_negative_reviews


# Loads the vocabulary
def load_vocabulary(file_path, num_words=10000):
    with open(file_path) as vocab:
        vocab_list = [next(vocab) for x in range(num_words)]
    vocab_list = [str(vocab).strip() for vocab in vocab_list]
    return vocab_list


def label_data(positive_revs, negative_revs):
    # Generate the labels
    positive_labels = [[0, 1] for _ in positive_revs]
    negative_labels = [[1, 0] for _ in negative_revs]

    # Concatenates the positive and negative labels for train and val
    y_labels = np.concatenate([positive_labels, negative_labels], 0)

    x_train = positive_revs + negative_revs

    return [x_train, y_labels]


def split_train_validation(x_train, y_train, amount_val=.25):
    x_train_shuffled = []
    y_train_shuffled = []
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    for i in shuffle_indices:
        x_train_shuffled.append(x_train[i])
        y_train_shuffled.append(y_train[i])

    total_reviews = len(x_train_shuffled)
    training_num = total_reviews - int(total_reviews * amount_val)

    x_t = x_train_shuffled[:training_num]
    y_t = y_train_shuffled[:training_num]

    x_dev = x_train_shuffled[training_num:]
    y_dev = y_train_shuffled[training_num:]

    return [x_t, y_t], [x_dev, y_dev]


def set_oov(reviews, vocabulary):
    updated_reviews = []
    for review in reviews:
        splitted_review = review.split(" ")
        for i, word in enumerate(splitted_review):
            if word not in vocabulary:
                splitted_review[i] = 'oov'
            else:
                splitted_review[i] = word
        new_review = ' '.join(splitted_review)
        updated_reviews.append(new_review)
    return updated_reviews


def set_oov_tag(reviews, vocabulary):
    updated_reviews = []
    set_vocabulary = set(vocabulary)
    for review in reviews:
        set_review = set(review.split(" "))
        oov_words = set_review - set_vocabulary
        #print(list(oov_words))

        dic_oov_words = {k:'oov' for k in oov_words}
        #print(dic_oov_words)
        if len(dic_oov_words) >= 1:
            rep = dict((re.escape(k), v) for k, v in dic_oov_words.items())
            pattern = re.compile("|".join(rep.keys()))
            oov_review = pattern.sub(lambda m: rep[re.escape(m.group(0))], review)
            updated_reviews.append(oov_review)
        else:
            updated_reviews.append(review)
    return updated_reviews


# Functions for the N-Grams
def parse_ngrams(text, n_g):
    ngrams_text = []
    for item in text:
        words = item.split(" ")
        words = find_ngrams(words, n_g)
        words =["-".join(word) for word in words]
        ngrams_text.append(" ".join(words[1:]))
    return ngrams_text


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def build_noov(ngram, vocabulary):
    aux_text = []
    for element in ngram:
        actual_word = []
        for word in element.split(" "):
            if word in vocabulary:
                actual_word.append(word)
            else:
                actual_word.append("noov")
        actual_word = " ".join(actual_word)
        aux_text.append(actual_word)
    return aux_text


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
