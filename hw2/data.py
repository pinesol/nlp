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
NUM_UNIGRAMS = 10000
NUM_BIGRAMS = 10000


def _load_reviews(review_dir, max_reviews=None):
    print('Loading reviews from "{}"'.format(review_dir))
    reviews = []    
    for filename in os.listdir(review_dir):
        if max_reviews and len(reviews) > max_reviews:
            break
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


def make_pickle_filename(use_bigrams=False, max_words=None, max_reviews=None):
    pickle_filename = ''
    if use_bigrams:
        pickle_filename += 'bigram-'
    if max_words:
        pickle_filename += 'words' + str(max_words) + '-'
    if max_reviews:
        pickle_filename += 'reviews' + str(max_reviews) + '-'
    pickle_filename += PICKLE_FILE
    return pickle_filename
    

# Returns vocab map, unshuffled reviews and labels lists.
# Words not in the 10K words are replaced with "<oov>".
# The vocab map is indexed from one, zero indicates EOD.
# Punctuation is left in.
# positive label = [0,1]
# negative label = [1,0]
def load(use_pickle=True, use_bigrams=False, max_words=None, max_reviews=None):
    pickle_filename = make_pickle_filename(use_bigrams, max_words, max_reviews)
    if use_pickle and os.path.isfile(pickle_filename):
        print('Loading reviews from pickle file "{}"'.format(pickle_filename))
        vocab, reviews, labels = pickle.load(open(pickle_filename, 'r'))
    else:
        print('Loading from scratch...'.format(pickle_filename))
        
        positive_reviews = _load_reviews('aclImdb/train/pos/', max_reviews)
        negative_reviews = _load_reviews('aclImdb/train/neg/', max_reviews)
        reviews = positive_reviews + negative_reviews

        # TODO Don't know why the labels are done like this, but that's what HW1 does...
        positive_labels = [[0, 1] for _ in positive_reviews]
        negative_labels = [[1, 0] for _ in negative_reviews]
        labels = np.concatenate([positive_labels, negative_labels], 0)    
        
        # Replace all words that aren't in the top 10k with <oov>.
        print('Filtering out top 10k words')
        word_counts = collections.Counter([word for review in reviews for word in review])
        vocab = set([pair[0] for pair in word_counts.most_common(NUM_UNIGRAMS)])
        reviews = [[word if word in vocab else OUT_OF_VOCAB for word in review]
                   for review in reviews]
        # Add the OUT_OF_VOCAB symbol to the vocabulary.
        vocab.add(OUT_OF_VOCAB)

        if use_bigrams:
            print('Adding bigrams...')
            bigram_reviews = [[review[i]+'|'+review[i+1] for i in range(len(review)-1)]
                              for review in reviews]
            bigram_counts = collections.Counter([bigram for review in bigram_reviews for bigram in review])
            # Get the most commonly occuring bigrams. Don't add the useless <oov>|<oov> bigram.
            OOVOOV = OUT_OF_VOCAB + '|' + OUT_OF_VOCAB
            bigrams_vocab = set([pair[0] for pair in bigram_counts.most_common(NUM_BIGRAMS+1) if pair[0] != OOVOOV])
            print('Adding {} bigrams to the vocab'.format(len(bigrams_vocab)))
            # Remove infrequent words
            bigram_reviews = [[word for word in review if word in bigrams_vocab]
                              for review in bigram_reviews]
            # Merge the bigram reviews and vocab with the unigram reviews and vocab.
            reviews = [uni + bi for uni, bi in zip(reviews, bigram_reviews)]
            vocab |= bigrams_vocab

        if max_words:
            print('Truncating the reviews to have a maximum of {} words'.format(max_words))
            reviews = [review[:max_words] for review in reviews]
            
        # Make vocab into a word, index dictionary.
        # Indexing the words from one so that a value of zero indicates
        # the end of the review has already been passed.
        vocab = {word:(i+1) for i, word in enumerate(sorted(vocab))}

        print('Saving loaded data to file "{}"'.format(pickle_filename))
        pickle.dump((vocab, reviews, labels), open(pickle_filename, 'w'))

        print('Loaded {} data points'.format(len(reviews)))
    return vocab, reviews, labels    


# Coverts the reviews list into a 2D numpy array of vocab IDs.
# Each vocab_id review is padded with zeros to the length of the longest review.
def make_vocab_id_reviews(vocab, reviews):
    max_length = max([len(review) for review in reviews])
    vocab_id_reviews = np.zeros((len(reviews), max_length), dtype=np.int64)
    for i, review in enumerate(reviews):
        for j, word in enumerate(review):
            vocab_id_reviews[i][j] = vocab.get(word)
    return vocab_id_reviews

    
# returns reviews_train, reviews_val, labels_train, labels_val
def shuffle_split_data(reviews, labels, val_size=0.2, seed=1):
    if seed:
        np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(reviews)))
    reviews = reviews[shuffle_indices]
    labels = labels[shuffle_indices]
    reviews_train, reviews_val, labels_train, labels_val = cv.train_test_split(
        reviews, labels, test_size=val_size)
    print('Split {} training data points, {} validation data points'.format(
        len(reviews_train), len(reviews_val)))
    return reviews_train, reviews_val, labels_train, labels_val

    
# Taken from
# https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
#Num batches Per Epoch: 626
#Num total batches: 5008
def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    print 'Num batches Per Epoch: {}'.format(num_batches_per_epoch)
    print 'Num total batches: {}'.format(num_batches_per_epoch*num_epochs)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def create_fasttext_data(train_filepath, test_filepath):
    pos_dir = 'aclImdb/train/pos/'
    neg_dir = 'aclImdb/train/neg/'

    def create_fasttext_review_list(review_dir, positive):
        reviews = []
        for filename in os.listdir(review_dir):
            with open(os.path.join(review_dir, filename)) as f:
                review_text = f.read().replace('<br />', ' ').replace('\n', ' ')
                if positive:
                    label = '__label__1 , '
                else:
                    label = '__label__0 , '
                reviews.append(label + review_text)
        return reviews

    print('Reading positive reviews')
    pos_reviews = create_fasttext_review_list(pos_dir, positive=True)
    print('Reading negative reviews')
    neg_reviews = create_fasttext_review_list(neg_dir, positive=False)

    reviews = pos_reviews + neg_reviews
    random.shuffle(reviews)
    train_reviews = reviews[:int(0.8*len(reviews))]
    test_reviews = reviews[len(train_reviews):]
    
    train_review_str = '\n'.join(train_reviews)
    print('Writing train output to {}'.format(train_filepath))
    with open(train_filepath, 'w') as f:
        f.write(train_review_str)
    test_review_str = '\n'.join(test_reviews)
    print('Writing test output to {}'.format(test_filepath))
    with open(test_filepath, 'w') as f:
        f.write(test_review_str)

    
                    



                
