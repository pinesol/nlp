'''Code to load review data.'''

import collections
import numpy as np
import os
import random
import re


PICKLE_FILE = 'data.p'


def _load_reviews(review_dir):
    print('Loading reviews from "{}"'.format(review_dir))
    reviews = []    
    for filename in os.listdir(review_dir):
        with open(os.path.join(review_dir, filename)) as f:
            # NOTE: leaving in punctuation.
            review = [word.strip().lower() for word in re.split('(\W+)', f.read())]
            # filter out empty strings
            review = filter(bool, review)
            reviews.append(review)
    return reviews

def load():
    if os.path.isfile(PICKLE_FILE):
        print('Loading reviews from pickle file "{}"'.format(PICKLE_FILE))
        shuffled_reviews, shuffled_labels = pickle.load(open(PICKLE_FILE, 'r'))
    else:
        print('Could\'t find saved file "{}", loading from scratch...'.format(PICKLE_FILE))
        
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
        top_words = word_counts.most_common(10000) # TODO this returns pairs, get the first one of each pair and make it a set
        reviews = [set(review).intersection(top_words) for review in reviews] # TODO always returns empty!

        # Shuffle the reviews
        zipped_reviews = zip(reviews, labels)
        random.shuffle(zipped_reviews)
        shuffled_reviews, shuffled_labels = zip(*zipped_reviews)

#        print('Saving loaded data to file "{}"'.format(PICKLE_FILE))
#        pickle.dump((shuffled_reviews, shuffled_labels), open(PICKLE_FILE, 'w'))
        
        print('Loaded {} data-labels pairs'.format(len(reviews)))
    return shuffled_reviews, shuffled_labels
