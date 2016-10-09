'''Code to load review data.'''

import collections
import numpy as np
import os
import pickle
import random
import re
import sklearn.cross_validation as cv

PICKLE_FILE = 'data.p'
OUT_OF_VOCAB = '<oov>'

def _load_reviews(review_dir):
    print('Loading reviews from "{}"'.format(review_dir))
    reviews = []    
    for filename in os.listdir(review_dir):
        with open(os.path.join(review_dir, filename)) as f:
            # NOTE: leaving in punctuation (periods, commas, etc).
            review_text = f.read().replace('<br />', '\n')  # Replace HTML breaks with newline chars.
            # Split on parens, whitespace, periods, hyphens, exclamation marks, question marks,
            # colons, and double quotes. Leaving in single quotes so contractions stay together.
            # Leaving in the punctutation that's split on.
            review = filter(None, [word.lower().strip()
                                   for word in re.split("([\(\)\s\.,\-!?:\"]+)", review_text)])
            reviews.append(review)
    return reviews

# Returns vocab set, unshuffled reviews and labels lists.
# Words not in the 10K words are replaced with "<oov>".
# Punctuation is left in.
# positive label = [0,1]
# negative label = [1,0]
def load(use_pickle=True):
    if use_pickle and os.path.isfile(PICKLE_FILE):
        print('Loading reviews from pickle file "{}"'.format(PICKLE_FILE))
        vocab, reviews, labels = pickle.load(open(PICKLE_FILE, 'r'))
    else:
        print('Loading from scratch...'.format(PICKLE_FILE))
        
        positive_reviews = _load_reviews('aclImdb/train/pos/')
        negative_reviews = _load_reviews('aclImdb/train/neg/')
        reviews = positive_reviews + negative_reviews

        # TODO Don't know why the labels are done like this, but that's what HW1 does...
        positive_labels = [[0, 1] for _ in positive_reviews]
        negative_labels = [[1, 0] for _ in negative_reviews]
        labels = np.concatenate([positive_labels, negative_labels], 0)    
        
        # Replace all words that aren't in the top 10k with <oov>.
        print('Filtering out top 10k words')
        word_counts = collections.Counter([word for review in reviews for word in review])
        vocab = set([pair[0] for pair in word_counts.most_common(10000)])
        reviews = [[word if word in vocab else OUT_OF_VOCAB for word in review]
                   for review in reviews]
        vocab.add(OUT_OF_VOCAB)

        print('Saving loaded data to file "{}"'.format(PICKLE_FILE))
        pickle.dump((vocab, reviews, labels), open(PICKLE_FILE, 'w'))

        print('Loaded {} data points'.format(len(reviews)))
        return vocab, reviews, labels    


# returns reviews_train, reviews_val, labels_train, labels_val
def shuffle_split_data(reviews, labels, val_size=0.2, seed=1):
    if seed:
        random.seed(seed)
    # Shuffle the reviews
    zipped_reviews = zip(reviews, labels)
    random.shuffle(zipped_reviews)
    shuffled_reviews, shuffled_labels = zip(*zipped_reviews)
    reviews_train, reviews_val, labels_train, labels_val = cv.train_test_split(
        shuffled_reviews, shuffled_labels, test_size=val_size)
    print('Split {} training data points, {} validation data points'.format(
        len(reviews_train), len(reviews_val)))
    return reviews_train, reviews_val, labels_train, labels_val

    
# Taken from
# https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
