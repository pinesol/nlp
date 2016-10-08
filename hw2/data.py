'''Code to load review data.'''

import collections
import numpy as np
import os
import pickle
import random
import re
import sklearn.cross_validation as cv

PICKLE_FILE = 'data.p'
VALIDATION_PROPORTION = 0.2

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

def load(use_pickle=True):
    if use_pickle and os.path.isfile(PICKLE_FILE):
        print('Loading reviews from pickle file "{}"'.format(PICKLE_FILE))
        reviews_train, reviews_val, labels_train, labels_val = pickle.load(open(PICKLE_FILE, 'r'))
    else:
        print('Loading from scratch...'.format(PICKLE_FILE))
        
        positive_reviews = _load_reviews('aclImdb/train/pos/')
        negative_reviews = _load_reviews('aclImdb/train/neg/')
        reviews = positive_reviews + negative_reviews

        # TODO Don't know why the labels are done like this, but that's what HW1 does...
        positive_labels = [[0, 1] for _ in positive_reviews]
        negative_labels = [[1, 0] for _ in negative_reviews]
        labels = np.concatenate([positive_labels, negative_labels], 0)    
        
        # Filter out all words but the top 10k
        print('Filtering out top 10k words')
        word_counts = collections.Counter([word for review in reviews for word in review])
        top_words = set([pair[0] for pair in word_counts.most_common(10000)])
        reviews = [set(review).intersection(top_words) for review in reviews]

        # Shuffle the reviews
        zipped_reviews = zip(reviews, labels)
        random.shuffle(zipped_reviews)
        shuffled_reviews, shuffled_labels = zip(*zipped_reviews)

        reviews_train, reviews_val, labels_train, labels_val = cv.train_test_split(
            shuffled_reviews, shuffled_labels, test_size=VALIDATION_PROPORTION)
        
        print('Saving loaded data to file "{}"'.format(PICKLE_FILE))
        pickle.dump((reviews_train, reviews_val, labels_train, labels_val),
                    open(PICKLE_FILE, 'w'))
        
        print('Loaded {} training data points, {} validation data points'.format(
            len(reviews_train), len(reviews_val)))
    return reviews_train, reviews_val, labels_train, labels_val
